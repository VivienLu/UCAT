import numpy as np
import argparse, os, time, random
from tqdm import tqdm
import logging
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import *
from replace import clip
from models import prompters
from models.prompters import TokenPrompter,NullPrompter
from models.model import *
from attacks_multilabel import *
import copy
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32
from utils import load_train_dataset, load_val_datasets, get_text_prompts_train, \
    get_text_prompts_val

import torch.nn as nn
from attention_map import *
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as kl_div
from tensorboardX import SummaryWriter
import shutil
from dpn_losses import *
import sys
from datetime import datetime

torch.set_num_threads(10)
torch.set_num_interop_threads(10)

def parse_option():
    parser = argparse.ArgumentParser('Adapting CLIP for zero-shot adv robustness')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=1, help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    # optimization
    parser.add_argument('--Method', type=str, default='UCAT')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # adversarial attack
    parser.add_argument('--train_eps', type=float, default=1, help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=2)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=1, help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=100)
    parser.add_argument('--test_stepsize', type=int, default=1)
    
    # model
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch',
                        choices=['padding', 'random_patch', 'fixed_patch', 'null_patch'],
                        help='choose visual prompting method')
    # parser.add_argument('--prompt_size', type=int, default=30, help='size for visual prompts')
    # parser.add_argument('--add_prompt_size', type=int, default=10, help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data', 
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='tinyImageNet',
                        choices=['cifar100', 'ImageNet', 'cifar10', 'tinyImageNet'], help='Data set for training')
    parser.add_argument('--image_size', type=int, default=224, help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0, help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='/work1/wenjing/1_CLIP_Uncertainty/results/CEEDL/models', 
                        help='path to save models')
    parser.add_argument('--filename', type=str, default=None, 
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1, help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')

    parser.add_argument('--gpu', type=int, default=1, help='gpu to use')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--VPbaseline', action='store_true')
    parser.add_argument('--attack', choices=['pgd', 'autoattack', 'CW'], default='pgd')
    parser.add_argument('--noimginprop', action='store_true')
    
    #FT
    parser.add_argument('--last_num_ft', type=int, default=0)
    parser.add_argument('--adaptation_method', type=str, default='FT', choices=['VPT','FT'],
                        help='choose visual adaptation method')
    parser.add_argument('--Distance_metric', type=str, default='l2', choices=['cos', 'l2', 'l1'],
                        help='Select the distance measure in the loss function')
    parser.add_argument('--atten_methods',type=str,default='text',choices=['text','visual'])
    parser.add_argument('--testdata', type=str, nargs='+')
    parser.add_argument('--target_concentration', type=float, default=0.07, help='gpu to use')
    parser.add_argument('--mode', type=str, default='train',choices=['test','train'])
    parser.add_argument('--save_dir', type=str, default='/work1/wenjing/1_CLIP_Uncertainty/results/MultiLabel/', 
                        help='path to save models')
    parser.add_argument('--tau', type=int, default=5, help='gpu to use')


    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_lr-{}_decay-{}_bsz-{}_warmup-{}'. \
        format(args.Method, args.dataset, args.model, args.arch, args.learning_rate, 
               args.weight_decay, args.batch_size, args.warmup)
    return args

def compute_multilabel_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
    top_k: int = None,
    eps: float = 1e-12,
):
    assert outputs.shape == targets.shape, f"shape mismatch: {outputs.shape} vs {targets.shape}"
    device = outputs.device
    B, C = outputs.shape

    # probs = torch.sigmoid(outputs/100) if from_logits else outputs.clamp(0, 1)
    probs = outputs
    probs_np = probs.detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(np.int32)

    if top_k is not None:
        k = min(top_k, C)
        pred_bin = torch.zeros_like(probs, dtype=torch.int)
        topk = torch.topk(probs, k=k, dim=1, largest=True, sorted=False).indices
        pred_bin.scatter_(1, topk, 1)
    else:
        pred_bin = (probs >= threshold).to(torch.int)

    y_pred = pred_bin.detach().cpu().numpy().astype(np.int32)

    # ---------- AP / mAP ----------
    def _ap_per_class(y_true_c, y_score_c):
        # y_true_c, y_score_c shape [B]
        if y_true_c.max() == y_true_c.min():
            return np.nan
        try:
            from sklearn.metrics import average_precision_score
            return float(average_precision_score(y_true_c, y_score_c))
        except Exception:
            order = np.argsort(-y_score_c)
            y_true_sorted = y_true_c[order]
            tp = np.cumsum(y_true_sorted == 1)
            fp = np.cumsum(y_true_sorted == 0)
            recall = tp / max(tp[-1], 1e-12)
            precision = tp / np.maximum(tp + fp, 1e-12)
            
            recall_points = np.linspace(0, 1, 101)
            prec_at_r = []
            for r in recall_points:
                mask = recall >= r
                prec_at_r.append(precision[mask].max() if np.any(mask) else 0.0)
            return float(np.mean(prec_at_r))

    per_class_ap = np.array([_ap_per_class(y_true[:, c], probs_np[:, c]) for c in range(C)], dtype=float)
    mAP = float(np.nanmean(per_class_ap))

    # ---------- micro / macro P/R/F1 ----------
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    tp = (y_true_flat * y_pred_flat).sum()
    fp = ((1 - y_true_flat) * y_pred_flat).sum()
    fn = (y_true_flat * (1 - y_pred_flat)).sum()

    micro_precision = float(tp / max(tp + fp, eps))
    micro_recall    = float(tp / max(tp + fn, eps))
    micro_f1        = float(2 * micro_precision * micro_recall / max(micro_precision + micro_recall, eps))

    precisions, recalls, f1s = [], [], []
    for c in range(C):
        yt, yp = y_true[:, c], y_pred[:, c]
        tp_c = (yt & yp).sum()
        fp_c = ((1 - yt) & yp).sum()
        fn_c = (yt & (1 - yp)).sum()
        p_c = tp_c / max(tp_c + fp_c, eps)
        r_c = tp_c / max(tp_c + fn_c, eps)
        f1_c = 2 * p_c * r_c / max(p_c + r_c, eps)
        precisions.append(float(p_c))
        recalls.append(float(r_c))
        f1s.append(float(f1_c))

    macro_precision = float(np.mean(precisions))
    macro_recall    = float(np.mean(recalls))
    macro_f1        = float(np.mean(f1s))

    return {
        "per_class_ap": per_class_ap,   # [C]
        "mAP": mAP,
        "micro/precision": micro_precision,
        "micro/recall": micro_recall,
        "micro/f1": micro_f1,
        "macro/precision": macro_precision,
        "macro/recall": macro_recall,
        "macro/f1": macro_f1,
        "threshold": None if top_k is not None else float(threshold),
        "top_k": int(top_k) if top_k is not None else None,
    }

def main():
    global best_acc1, device, logger
    args = parse_option()
    device = torch.device("cuda:{}".format(args.gpu))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if args.mode == 'train':
        log_dir = os.path.join(args.save_dir, args.Method, 'log')
    else:
        log_dir = os.path.join(args.save_dir, args.Method, f'{args.attack}-testeps{args.test_eps}-testnum{args.test_numsteps}')
    args.model_dir = os.path.join(args.save_dir, args.Method, 'model')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    current_file = os.path.abspath(sys.argv[0])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    basename = os.path.basename(current_file)
    target_file = os.path.join(log_dir, f"{timestamp}-{basename}")
    shutil.copy(current_file, target_file)
    
    if args.mode == 'train':
        file_handler = logging.FileHandler(os.path.join(log_dir,f'{args.filename}.log'))
    else:
        file_handler = logging.FileHandler(os.path.join(log_dir,f'{args.testdata[0]}.log'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    args.train_eps = args.train_eps / 255.
    args.test_eps = args.test_eps / 255.
    args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = args.test_stepsize / 255.

    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f'{key}: {value}')
        logger.info(f'{key}: {value}')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    """create model"""
    if args.adaptation_method == 'VPT':
        add_prompt_len = args.add_prompt_size
    else:
        add_prompt_len = 0
    print(" create model")
    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)

    convert_models_to_fp32(model)
    model = model.to(device)
    frozen_model = copy.deepcopy(model).to(device)
    
    model.eval()
    frozen_model.eval() 
    
    """define criterion and optimizer"""
    if args.adaptation_method == 'VPT':
        prompter = prompters.__dict__[args.method](args).to(device)
        add_prompter = TokenPrompter(args.add_prompt_size).to(device)
        optimizer = torch.optim.SGD(list(prompter.parameters()) + list(add_prompter.parameters()),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        prompter = NullPrompter().to(device)
        add_prompter = TokenPrompter(0).to(device)
        if args.last_num_ft == 0:
            optimizer = torch.optim.SGD(model.visual.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(list(model.visual.parameters())[-args.last_num_ft:],
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    
    """Load the pre-trained model"""
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if 'epoch' in checkpoint.keys():
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
            else:
                args.start_epoch = 0
                best_acc1 = 0

            if 'vision_encoder_state_dict' in checkpoint.keys():
                model.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
            elif 'state_dict' in checkpoint.keys():
                prompter.load_state_dict(checkpoint['state_dict'])
                add_prompter.load_state_dict(checkpoint['add_prompter'])
            else:
                model.visual.load_state_dict(checkpoint, strict=False)

            if 'epoch' in checkpoint.keys():
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                logger.info("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, 0))
                logger.info("loaded checkpoint '{}' (epoch {})".format(args.resume, 0))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    template = 'This is a photo of a {}'
    print(f'template: {template}')


    """load training dataset"""
    train_dataset = load_train_dataset(args)
    
    """load val dataset(s)"""
    if args.testdata is None:
        val_dataset_name = ['tinyImageNet','cifar10', 'cifar100','STL10','Food101','oxfordpet','flowers102','dtd','EuroSAT',\
                            'fgvc_aircraft','Caltech101','Caltech256','StanfordCars','PCAM','ImageNet','SUN397']
    else:
        val_dataset_name = args.testdata
    val_dataset_list = load_val_datasets(args, val_dataset_name)


    """create dataloaders"""
    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                               shuffle=True, sampler=train_sampler)

    val_loader_list = [DataLoader(each, batch_size=args.batch_size*2, pin_memory=True,
                                   shuffle=False, sampler=val_sampler) for each in val_dataset_list]

    """get text prompts for training/val"""
    texts_train = get_text_prompts_train(args, train_dataset, template=template)
    texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=template)
    
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True
    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    epochs_since_improvement = 0
    best_acc1 = 0

    if args.mode == 'train':

        """training"""
        for epoch in range(args.epochs):
            # train for one epoch
            train(train_loader, texts_train, model,frozen_model, prompter, add_prompter, optimizer, scheduler,
                scaler, epoch,  args, writer)
            
            # evaluate on validation set
            if epoch % args.validate_freq == 0:
                acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model,frozen_model,optimizer, device,
                                    prompter, add_prompter, args, writer, epoch)
                
            # remember best acc@1 and save checkpoint
            is_best = acc1_mean > best_acc1
            best_acc1 = max(acc1_mean, best_acc1)

            save_checkpoint({
                'epoch': args.start_epoch + epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'vision_encoder_state_dict':model.visual.state_dict(),
            }, args, is_best=is_best)

            if is_best:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                print(f"There's no improvement for {epochs_since_improvement} epochs.")
                logger.info(f"There's no improvement for {epochs_since_improvement} epochs.")
                if epochs_since_improvement >= args.patience:
                    print("The training halted by early stopping criterion.")
                    logger.info("The training halted by early stopping criterion.")
                    break
            writer.flush()
    else:
        acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model,frozen_model,optimizer, device,
                                prompter, add_prompter, args, writer, epoch=0)


"""train function"""
def train(train_loader, texts, model, frozen_model, prompter, add_prompter,
          optimizer, scheduler, scaler, epoch,  args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(args.start_epoch + epoch))

    """switch to train mode"""
    prompter.train()
    add_prompter.train()
    model.visual.train()
    num_batches_per_epoch = len(train_loader)

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    end = time.time()
    iter_num = num_batches_per_epoch*(args.start_epoch + epoch)


    logit_scale = model.logit_scale.exp().detach().item() # *args.target_concentration
    target_concentration = torch.exp(nn.Softplus()(torch.Tensor([2/args.target_concentration])))
    lambda_rob = torch.Tensor([1e2/target_concentration]).to(device)
    CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
    criterion_kl = nn.KLDivLoss(reduction="mean").to(device)

    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)
        BATCH_SIZE = images.size(0)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)
        
        # with automatic mixed precision
        with autocast():
            """Build adversarial example"""
            if not args.VPbaseline:
                delta = attack_pgd(prompter, model,add_prompter,images,
                                target, text_tokens, alpha, attack_iters, 'l_inf',
                                device=device, args=args, epsilon=args.train_eps)
                tmp = clip_img_preprocessing(images + delta,device)
            else:
                tmp = clip_img_preprocessing(images,device)

            prompted_images = prompter(tmp)
            clean_images = prompter(clip_img_preprocessing(images,device))
            prompt_token = add_prompter()

            output, _ , text_features= multiGPU_CLIP(model, prompted_images, text_tokens, target, device, prompt_token)
            alpha_adv = 1e-6 + torch.exp((output/logit_scale + 1)/args.target_concentration)

            output_clean, _ , _ = multiGPU_CLIP(model, clean_images, text_tokens, target, device, prompt_token)

            with torch.no_grad():
                output_ori, _ , _ = multiGPU_CLIP(frozen_model, clean_images, text_tokens, target, device, prompt_token)
            alpha_ori = 1e-6 + torch.exp((output_ori/logit_scale+1)/args.target_concentration)
            kl_adv2ori = (kl_div(Dirichlet(alpha_adv), Dirichlet(alpha_ori))).mean()

            iter_num += 1
            loss_TeCoA = CrossEntropyLoss(output, target)
            loss = loss_TeCoA + lambda_rob*kl_adv2ori

            logger.info(f'Iter num: {iter_num}, loss {loss.item()}, TeCoA: {loss_TeCoA.item()}, kl_adv2ori: {kl_adv2ori}, lambda_rob: {(lambda_rob*kl_adv2ori).item()}')
            print(f'Iter num: {iter_num}, loss {loss.item()}, TeCoA: {loss_TeCoA.item()}, kl_adv2ori: {kl_adv2ori}, lambda_rob: {(lambda_rob*kl_adv2ori).item()}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)   
        # measure accuracy
        acc1_org = accuracy(output_ori, target, topk=(1,))
        acc1 = accuracy(output, target, topk=(1,))
        acc1_clean = accuracy(output_clean, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('Train/Acc/acc_org', acc1_org[0], iter_num)
        writer.add_scalar('Train/Acc/acc_clean', acc1_clean[0], iter_num)
        writer.add_scalar('Train/Acc/acc_adv', acc1[0], iter_num)
        writer.add_scalar('Loss/TeCoA', loss_TeCoA, iter_num)
        writer.add_scalar('Loss/kl-adv2ori', kl_adv2ori, iter_num)
        

        if i % args.print_freq == 0:
            entries = progress.display(i)
            logger.info(entries)
            if args.debug:
                break
    save_checkpoint({
        'epoch': args.start_epoch + epoch + 1,
        'state_dict': prompter.state_dict(),
        'add_prompter': add_prompter.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        'vision_encoder_state_dict':model.visual.state_dict(),
        }, args)
    return losses.avg, top1.avg

def build_metric_meters(prefix: str, topk: int):
    return [
        AverageMeter(f'{prefix}@{topk}mAP', ':6.2f'),
        AverageMeter(f'{prefix}@{topk}Micro_P', ':6.2f'),
        AverageMeter(f'{prefix}@{topk}Micro_R', ':6.2f'),
        AverageMeter(f'{prefix}@{topk}Micro_F1', ':6.2f'),
        AverageMeter(f'{prefix}@{topk}Macro_P', ':6.2f'),
        AverageMeter(f'{prefix}@{topk}Macro_R', ':6.2f'),
        AverageMeter(f'{prefix}@{topk}Macro_F1', ':6.2f'),
    ]

def log_metrics(logger, prefix, meters):
    logger.info(
        f"{prefix}: "
        f"mAP {meters[0].avg*100:.2f} | "
        f"Micro P {meters[1].avg*100:.2f} R {meters[2].avg*100:.2f} F1 {meters[3].avg*100:.2f} | "
        f"Macro P {meters[4].avg*100:.2f} R {meters[5].avg*100:.2f} F1 {meters[6].avg*100:.2f}"
    )

def validate(val_loader_list, val_dataset_name, texts_list, model,frozen_model,optimizer, device,
                prompter, add_prompter, args, writer, epoch):
    dataset_num = len(val_loader_list)
    acc_all = []

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        binary = ['PCAM', 'hateful_memes']
        attacks_to_run=['apgd-ce', 'apgd-dlr']

        if dataset_name in binary:
            attacks_to_run=['apgd-ce']
            
        batch_time = AverageMeter('Time', ':6.3f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, ],
            prefix=dataset_name + '_Validate: ')

        # Clean
        clean_topk3_meters = build_metric_meters('Clean', 3)
        clean_topk5_meters = build_metric_meters('Clean', 5)

        # Org
        org_topk3_meters = build_metric_meters('Org', 3)
        org_topk5_meters = build_metric_meters('Org', 5)

        # Adv
        adv_topk3_meters = build_metric_meters('Adv', 3)
        adv_topk5_meters = build_metric_meters('Adv', 5)


        adv_clip_topk3_meters = build_metric_meters('AdvCLIP', 3)
        adv_clip_topk5_meters = build_metric_meters('AdvCLIP', 5)

        all_meters = (
            clean_topk3_meters + clean_topk5_meters +
            org_topk3_meters + org_topk5_meters +
            adv_topk3_meters + adv_topk5_meters +
            adv_clip_topk3_meters + adv_clip_topk5_meters
        )

        progress.meters.extend(all_meters)

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()
        model.zero_grad()
        frozen_model.eval()

        logit_scale = model.logit_scale.exp().detach().item()*args.target_concentration
        iter_num = len(val_loader) * (args.start_epoch + epoch)

        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            with autocast():

                # compute output
                with torch.no_grad():
                    """clean images"""
                    prompt_token = None
                    output_org = multiGPU_CLIP(model, clip_img_preprocessing(images,device),text_tokens,target, device, None)[0]
                    
                    # acc1 = accuracy(output_org, target, topk=(1,))
                    # outputs: [B, 80], targets: [B, 80]
                    metrics_clean_topk3 = compute_multilabel_metrics(output_org, target, from_logits=True, top_k=3)
                    metrics_clean_topk5 = compute_multilabel_metrics(output_org, target, from_logits=True, top_k=5)

                    batch_size = images.size(0)
                    clean_topk3_meters[0].update(metrics_clean_topk3['mAP'], batch_size),
                    clean_topk3_meters[1].update(metrics_clean_topk3['micro/precision'], batch_size),
                    clean_topk3_meters[2].update(metrics_clean_topk3['micro/recall'], batch_size),
                    clean_topk3_meters[3].update(metrics_clean_topk3['micro/f1'], batch_size),
                    clean_topk3_meters[4].update(metrics_clean_topk3['macro/precision'], batch_size),
                    clean_topk3_meters[5].update(metrics_clean_topk3['macro/recall'], batch_size),
                    clean_topk3_meters[6].update(metrics_clean_topk3['macro/f1'], batch_size)

                    clean_topk5_meters[0].update(metrics_clean_topk5['mAP'], batch_size),
                    clean_topk5_meters[1].update(metrics_clean_topk5['micro/precision'], batch_size),
                    clean_topk5_meters[2].update(metrics_clean_topk5['micro/recall'], batch_size),
                    clean_topk5_meters[3].update(metrics_clean_topk5['micro/f1'], batch_size),
                    clean_topk5_meters[4].update(metrics_clean_topk5['macro/precision'], batch_size),
                    clean_topk5_meters[5].update(metrics_clean_topk5['macro/recall'], batch_size),
                    clean_topk5_meters[6].update(metrics_clean_topk5['macro/f1'], batch_size)

                    log_metrics(logger, "Clean@3", clean_topk3_meters)
                    log_metrics(logger, "Clean@5", clean_topk5_meters)

                    output_ori = multiGPU_CLIP(frozen_model, clip_img_preprocessing(images,device),text_tokens,target, device, None)[0]


                    metrics_org_topk3 = compute_multilabel_metrics(output_ori, target, from_logits=True, top_k=3)
                    metrics_org_topk5 = compute_multilabel_metrics(output_ori, target, from_logits=True, top_k=5)


                    org_topk3_meters[0].update(metrics_org_topk3['mAP'], batch_size),
                    org_topk3_meters[1].update(metrics_org_topk3['micro/precision'], batch_size),
                    org_topk3_meters[2].update(metrics_org_topk3['micro/recall'], batch_size),
                    org_topk3_meters[3].update(metrics_org_topk3['micro/f1'], batch_size),
                    org_topk3_meters[4].update(metrics_org_topk3['macro/precision'], batch_size),
                    org_topk3_meters[5].update(metrics_org_topk3['macro/recall'], batch_size),
                    org_topk3_meters[6].update(metrics_org_topk3['macro/f1'], batch_size)

                    org_topk5_meters[0].update(metrics_org_topk5['mAP'], batch_size),
                    org_topk5_meters[1].update(metrics_org_topk5['micro/precision'], batch_size),
                    org_topk5_meters[2].update(metrics_org_topk5['micro/recall'], batch_size),
                    org_topk5_meters[3].update(metrics_org_topk5['micro/f1'], batch_size),
                    org_topk5_meters[4].update(metrics_org_topk5['macro/precision'], batch_size),
                    org_topk5_meters[5].update(metrics_org_topk5['macro/recall'], batch_size),
                    org_topk5_meters[6].update(metrics_org_topk5['macro/f1'], batch_size)

                    log_metrics(logger, "Org@3", org_topk3_meters)
                    log_metrics(logger, "Org@5", org_topk5_meters)



                """adv images"""
                if args.attack == 'pgd':
                    delta_noprompt = attack_pgd(None, model, None, images, target, text_tokens,
                                        test_stepsize, args.test_numsteps,'l_inf',device, args, epsilon=args.test_eps)
                    delta_noprompt_ori = attack_pgd(None, frozen_model, None, images, target, text_tokens,
                                        test_stepsize, args.test_numsteps,'l_inf',device, args, epsilon=args.test_eps)
                    attacked_images = images + delta_noprompt
                    attacked_images_ori = images + delta_noprompt_ori
                elif args.attack == "CW":
                    delta_prompt = attack_CW(prompter, model, add_prompter, images, target, text_tokens, test_stepsize,
                                                    args.test_numsteps, 'l_inf', device, epsilon=args.test_eps, multilabel=True)
                    delta_prompt_ori = attack_CW(prompter, frozen_model, add_prompter, images, target, text_tokens, test_stepsize,
                                                    args.test_numsteps, 'l_inf', device, epsilon=args.test_eps, multilabel=True)
                    attacked_images = images + delta_prompt
                    attacked_images_ori = images + delta_prompt_ori
                else:
                    attacked_images  = attack_auto(model, images, target, text_tokens, None, None, device,
                                            attacks_to_run=attacks_to_run, epsilon=args.test_eps)
                    attacked_images_ori  = attack_auto(frozen_model, images, target, text_tokens, None, None, device,
                                            attacks_to_run=attacks_to_run, epsilon=args.test_eps)

                # torch.cuda.empty_cache()
                with torch.no_grad():
                    output_org_adv, _, text_features= multiGPU_CLIP(model, clip_img_preprocessing(attacked_images,device),
                                                        text_tokens, target, device, None)
                    output_ori_adv = multiGPU_CLIP(frozen_model, clip_img_preprocessing(attacked_images_ori,device),text_tokens,target, device, None)[0]

                    metrics_adv_topk3 = compute_multilabel_metrics(output_org_adv, target, from_logits=True, top_k=3)
                    metrics_adv_topk5 = compute_multilabel_metrics(output_org_adv, target, from_logits=True, top_k=5)

                    adv_topk3_meters[0].update(metrics_adv_topk3['mAP'], batch_size),
                    adv_topk3_meters[1].update(metrics_adv_topk3['micro/precision'], batch_size),
                    adv_topk3_meters[2].update(metrics_adv_topk3['micro/recall'], batch_size),
                    adv_topk3_meters[3].update(metrics_adv_topk3['micro/f1'], batch_size),
                    adv_topk3_meters[4].update(metrics_adv_topk3['macro/precision'], batch_size),
                    adv_topk3_meters[5].update(metrics_adv_topk3['macro/recall'], batch_size),
                    adv_topk3_meters[6].update(metrics_adv_topk3['macro/f1'], batch_size)

                    adv_topk5_meters[0].update(metrics_adv_topk5['mAP'], batch_size),
                    adv_topk5_meters[1].update(metrics_adv_topk5['micro/precision'], batch_size),
                    adv_topk5_meters[2].update(metrics_adv_topk5['micro/recall'], batch_size),
                    adv_topk5_meters[3].update(metrics_adv_topk5['micro/f1'], batch_size),
                    adv_topk5_meters[4].update(metrics_adv_topk5['macro/precision'], batch_size),
                    adv_topk5_meters[5].update(metrics_adv_topk5['macro/recall'], batch_size),
                    adv_topk5_meters[6].update(metrics_adv_topk5['macro/f1'], batch_size)

                    log_metrics(logger, "Adv@3", adv_topk3_meters)
                    log_metrics(logger, "Adv@5", adv_topk5_meters)

                    metrics_adv_clip_topk3 = compute_multilabel_metrics(output_ori_adv, target, from_logits=True, top_k=3)
                    metrics_adv_clip_topk5 = compute_multilabel_metrics(output_ori_adv, target, from_logits=True, top_k=5)

                    adv_clip_topk3_meters[0].update(metrics_adv_clip_topk3['mAP'], batch_size),
                    adv_clip_topk3_meters[1].update(metrics_adv_clip_topk3['micro/precision'], batch_size),
                    adv_clip_topk3_meters[2].update(metrics_adv_clip_topk3['micro/recall'], batch_size),
                    adv_clip_topk3_meters[3].update(metrics_adv_clip_topk3['micro/f1'], batch_size),
                    adv_clip_topk3_meters[4].update(metrics_adv_clip_topk3['macro/precision'], batch_size),
                    adv_clip_topk3_meters[5].update(metrics_adv_clip_topk3['macro/recall'], batch_size),
                    adv_clip_topk3_meters[6].update(metrics_adv_clip_topk3['macro/f1'], batch_size)

                    adv_clip_topk5_meters[0].update(metrics_adv_clip_topk5['mAP'], batch_size),
                    adv_clip_topk5_meters[1].update(metrics_adv_clip_topk5['micro/precision'], batch_size),
                    adv_clip_topk5_meters[2].update(metrics_adv_clip_topk5['micro/recall'], batch_size),
                    adv_clip_topk5_meters[3].update(metrics_adv_clip_topk5['micro/f1'], batch_size),
                    adv_clip_topk5_meters[4].update(metrics_adv_clip_topk5['macro/precision'], batch_size),
                    adv_clip_topk5_meters[5].update(metrics_adv_clip_topk5['macro/recall'], batch_size),
                    adv_clip_topk5_meters[6].update(metrics_adv_clip_topk5['macro/f1'], batch_size)


                    log_metrics(logger, "Adv@3", adv_clip_topk3_meters)
                    log_metrics(logger, "Adv@5", adv_clip_topk5_meters)


                    print('\n\n\n', torch.mean(torch.abs(output_org - output_org_adv)).item())
            batch_time.update(time.time() - end)
            end = time.time()

            entries = progress.display(i)
            logger.info(entries)

            if i % args.print_freq == 0:
                entries = progress.display(i)
                logger.info(entries)
                if args.debug:
                    break
        torch.cuda.empty_cache()
        
        logger.info("="*60)
        log_final_results(logger, clean_topk3_meters, "Clean@3")
        log_final_results(logger, clean_topk5_meters, "Clean@5")
        log_final_results(logger, org_topk3_meters,   "Org@3")
        log_final_results(logger, org_topk5_meters,   "Org@5")
        log_final_results(logger, adv_topk3_meters,   "Adv@3")
        log_final_results(logger, adv_topk5_meters,   "Adv@5")
        log_final_results(logger, adv_clip_topk3_meters,   "AdvCLIP@3")
        log_final_results(logger, adv_clip_topk5_meters,   "AdvCLIP@5")
        logger.info("="*60)
        acc_all.append(adv_topk3_meters[0].avg)

    return np.mean(acc_all)

def log_final_results(logger, meters, tag):
    """
    meters: [mAP, micro_P, micro_R, micro_F1, macro_P, macro_R, macro_F1]
    tag: 'Clean@3', 'Org@5', 'Adv@3' ...
    """
    logger.info(
        f"{tag}: "
        f"mAP {meters[0].avg*100:.2f} | "
        f"Micro P {meters[1].avg*100:.2f} R {meters[2].avg*100:.2f} F1 {meters[3].avg*100:.2f} | "
        f"Macro P {meters[4].avg*100:.2f} R {meters[5].avg*100:.2f} F1 {meters[6].avg*100:.2f}"
    )

if __name__ == '__main__':
    main()
