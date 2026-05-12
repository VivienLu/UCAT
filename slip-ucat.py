from __future__ import print_function
import cv2
import numpy as np
import argparse, os, time, random
from tqdm import tqdm
import logging
import torch, torchvision
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import *
from replace import clip
from models import prompters
from models.prompters import TokenPrompter,NullPrompter
from models.model import *
from attacks import *
import copy
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from utils import load_train_dataset, load_val_datasets, get_text_prompts_train, \
    get_text_prompts_val

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from attention_map import *
import math
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as kl_div
from tensorboardX import SummaryWriter
import shutil
import sys
from datetime import datetime
from slip import slip_utils
from slip import slip_models
from collections import OrderedDict


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
    parser.add_argument('--Method', type=str, default='TGA-ZSR')
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
    parser.add_argument('--attack', choices=['pgd', 'autoattack', 'CW', 'CAA', 'a3'], default='pgd')
    parser.add_argument('--noimginprop', action='store_true')
    
    #FT
    parser.add_argument('--last_num_ft', type=int, default=0)
    parser.add_argument('--adaptation_method', type=str, default='FT', choices=['VPT','FT'],
                        help='choose visual adaptation method')
    parser.add_argument('--Distance_metric', type=str, default='l2', choices=['cos', 'l2', 'l1'],
                        help='Select the distance measure in the loss function')
    parser.add_argument('--atten_methods',type=str,default='text',choices=['text','visual'])
    parser.add_argument('--Alpha', type=float, default=0.08, help='L_AR in Equ.6')
    parser.add_argument('--Beta', type=float, default=0.05, help='L_AMC in Equ.7')
    parser.add_argument('--testdata', type=str, nargs='+')
    parser.add_argument('--target_concentration', type=float, default=0.07, help='gpu to use')
    parser.add_argument('--mode', type=str, default='train',choices=['test','train'])
    parser.add_argument('--save_dir', type=str, default='/work1/wenjing/1_CLIP_Uncertainty/results/ceedl/', 
                        help='path to save models')
    parser.add_argument('--tau', type=float, default=0.05, help='gpu to use')


    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_lr-{}_decay-{}_bsz-{}_warmup-{}_trial-{}_Alpha-{}_Beta-{}_distance-{}_atten_methods-{}-new'. \
        format(args.Method, args.dataset, args.model, args.arch, args.learning_rate, 
               args.weight_decay, args.batch_size, args.warmup, args.trial, args.Alpha, 
               args.Beta, args.Distance_metric, args.atten_methods)
    return args

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
    # model, preprocess = clip.load('/work1/wenjing/1_CLIP_Uncertainty/CLIP_ckp/slip_base_25ep.pt', device, jit=False, prompt_len=add_prompt_len)
    ckpt_path = '/work1/wenjing/1_CLIP_Uncertainty/CLIP_ckp/slip_base_25ep.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(slip_models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

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

def au_dirichlet(alpha: torch.Tensor, C: int, eps: float = 1e-12) -> torch.Tensor:
    alpha = torch.clamp(alpha, min=eps)
    alpha0 = alpha.sum(dim=1, keepdim=True)                 # [B,1]
    psi_a1  = torch.special.digamma(alpha + 1.0)            # [B,C]
    psi_a01 = torch.special.digamma(alpha0 + 1.0)           # [B,1]
    w = alpha / alpha0                                       # [B,C]
    AU = -torch.sum(w * (psi_a1 - psi_a01), dim=1)          # [B]
    return AU /math.log(alpha.size(1))

def eu_dirichlet(alpha: torch.Tensor, C: int, eps: float = 1e-12) -> (torch.Tensor, torch.Tensor):
    alpha = torch.clamp(alpha, min=eps)
    # EU = C /  (alpha+1).sum(dim=1)
    EU = C /  torch.logsumexp(alpha+1, dim=1)
    return EU /math.log(alpha.size(1))

def entropy_from_logits(logits):
    p = torch.softmax(logits, dim=1)
    return -(p * (p+1e-12).log()).sum(1) /math.log(logits.size(1))


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

    logit_scale = torch.Tensor([100.0]).to(device)
    target_concentration = torch.exp(nn.Softplus()(torch.Tensor([2/args.target_concentration])))
    lambda_rob = torch.Tensor([args.tau/target_concentration]).to(device)
    CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)

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
                                device=device, args=args, epsilon=args.train_eps, slip=True)
                tmp = clip_img_preprocessing(images + delta,device)
            else:
                tmp = clip_img_preprocessing(images,device)

            prompted_images = prompter(tmp)
            clean_images = prompter(clip_img_preprocessing(images,device))
            prompt_token = add_prompter()

            output, _ , text_features= multiGPU_CLIP(model, prompted_images, text_tokens, target, device, prompt_token, slip=True)
            alpha_adv = 1e-6 + torch.exp((output/logit_scale + 1)/args.target_concentration)

            output_clean, _ , _ = multiGPU_CLIP(model, clean_images, text_tokens, target, device, prompt_token, slip=True)

            with torch.no_grad():
                output_ori, _ , _ = multiGPU_CLIP(frozen_model, clean_images, text_tokens, target, device, prompt_token, slip=True)
            alpha_ori = 1e-6 + torch.exp((output_ori/logit_scale+1)/args.target_concentration)
            kl_adv2ori = (kl_div(Dirichlet(alpha_adv), Dirichlet(alpha_ori))).mean()
            iter_num += 1
            loss_TeCoA = CrossEntropyLoss(output, target)
            loss = loss_TeCoA + lambda_rob*kl_adv2ori

            logging.info(f'Iter num: {iter_num}, loss {loss.item()}, TeCoA: {loss_TeCoA.item()}, kl_adv2ori: {kl_adv2ori}, lambda_rob*kl_adv2ori: {(lambda_rob*kl_adv2ori).item()}')
            print(f'Iter num: {iter_num}, loss {loss.item()}, TeCoA: {loss_TeCoA.item()}, kl_adv2ori: {kl_adv2ori}, lambda_rob*kl_adv2ori: {(lambda_rob*kl_adv2ori).item()}')
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
        # losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Clean Acc@1', ':6.2f')
        top1_ori = AverageMeter('Original Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv FT Acc@1', ':6.2f')
        top1_adv_oriclip = AverageMeter('Adv CLIP Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1_org, top1_adv_org],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()
        model.zero_grad()
        frozen_model.eval()

        # logit_scale = model.logit_scale.exp().detach().item()*args.target_concentration
        logit_scale = torch.Tensor([100.0]).to(device)*args.target_concentration
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
                    output_org = multiGPU_CLIP(model, clip_img_preprocessing(images,device),text_tokens,target, device, None, slip=True)[0]
                    acc1 = accuracy(output_org, target, topk=(1,))
                    top1_org.update(acc1[0].item(), images.size(0))

                    output_ori = multiGPU_CLIP(frozen_model, clip_img_preprocessing(images,device),text_tokens,target, device, None, slip=True)[0]
                    acc1_ori = accuracy(output_ori, target, topk=(1,))
                    top1_ori.update(acc1_ori[0].item(), images.size(0))

                """adv images"""
                if args.attack == 'pgd':
                    delta_noprompt = attack_pgd(None, model, None, images, target, text_tokens,
                                        test_stepsize, args.test_numsteps,'l_inf',device, args, epsilon=args.test_eps, slip=True)
                    attacked_images = images + delta_noprompt

                    delta_noprompt_oriclip = attack_pgd(None, frozen_model, None, images, target, text_tokens,
                                        test_stepsize, args.test_numsteps,'l_inf',device, args, epsilon=args.test_eps, slip=True)
                    attacked_images_oriclip = images + delta_noprompt_oriclip

                elif args.attack == "CW":
                    delta_prompt = attack_CW(prompter, model, add_prompter, images, target, text_tokens, test_stepsize,
                                                    args.test_numsteps, 'l_inf', device, epsilon=args.test_eps, slip=True)
                    attacked_images = images + delta_prompt
                    delta_noprompt_oriclip = attack_CW(prompter, frozen_model, add_prompter, images, target, text_tokens, test_stepsize,
                                                    args.test_numsteps, 'l_inf', device, epsilon=args.test_eps, slip=True)
                    attacked_images_oriclip = images + delta_noprompt_oriclip

                elif args.attack == 'CAA':
                    attacked_images = attack_caa(
                        model, images, target, text_tokens, prompter, add_prompter, device,
                        enabled_attack=[5],
                        epsilon=args.test_eps,
                        inner_steps=args.test_numsteps,
                        order_schedule='fixed',   
                        start_num=1,
                        multiple_rand_start=False,
                        dataset_hint='imagenet'     
                    )
                    attacked_images_oriclip = attack_caa(
                        frozen_model, images, target, text_tokens, prompter, add_prompter, device,
                        enabled_attack=[5],
                        epsilon=args.test_eps,
                        inner_steps=args.test_numsteps,
                        order_schedule='fixed',      
                        start_num=1,
                        multiple_rand_start=False,
                        dataset_hint='imagenet'   
                    )
                elif args.attack == 'a3': 
                    attacked_images = attack_a3(
                        model, images, target, text_tokens, prompter, add_prompter, device,
                        epsilon=args.test_eps,     
                        num_steps=args.test_numsteps,
                        num_restarts=1,           
                        step_size=test_stepsize,
                        osd_start=0.0, osd_inc=0.1,
                        early_stop=True, rand_init=True, use_adi=True
                    )
                    attacked_images_oriclip = attack_a3(
                        frozen_model, images, target, text_tokens, prompter, add_prompter, device,
                        epsilon=args.test_eps,    
                        num_steps=args.test_numsteps, 
                        num_restarts=1,           
                        step_size=test_stepsize,
                        osd_start=0.0, osd_inc=0.1,
                        early_stop=True, rand_init=True, use_adi=True
                    )
                else:
                    attacked_images  = attack_auto(model, images, target, text_tokens, None, None, device,
                                            attacks_to_run=attacks_to_run, epsilon=args.test_eps, slip=True)
                    attacked_images_oriclip  = attack_auto(frozen_model, images, target, text_tokens, None, None, device,
                                            attacks_to_run=attacks_to_run, epsilon=args.test_eps, slip=True)

                # torch.cuda.empty_cache()
                with torch.no_grad():
                    output_org_adv, _, text_features= multiGPU_CLIP(model, clip_img_preprocessing(attacked_images,device),
                                                        text_tokens, target, device, None, slip=True)
                    output_org_adv_oriclip, _, text_features_oriclip= multiGPU_CLIP(frozen_model, clip_img_preprocessing(attacked_images_oriclip,device),
                                                        text_tokens, target, device, None, slip=True)
                    
                    acc1 = accuracy(output_org_adv, target, topk=(1,))
                    acc1_oriclip = accuracy(output_org_adv_oriclip, target, topk=(1,))
                    top1_adv_org.update(acc1[0].item(), images.size(0))
                    top1_adv_oriclip.update(acc1_oriclip[0].item(), images.size(0))
                # torch.cuda.empty_cache()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            writer.add_scalar(f'{dataset_name}/Acc/acc_org', top1_ori.avg, iter_num)
            writer.add_scalar(f'{dataset_name}/Acc/acc_clean', top1_org.avg, iter_num)
            writer.add_scalar(f'{dataset_name}/Acc/acc_adv', acc1[0], iter_num)


            if i % args.print_freq == 0:
                entries = progress.display(i)
                logger.info(entries)
                if args.debug:
                    break
        torch.cuda.empty_cache()
        print(dataset_name + ' * Adv FT Acc@1 {top1_adv_org.avg:.3f}' '* Clean Acc@1 {top1_org.avg:.3f}'  '* Original Acc@1 {top1_ori.avg:.3f}' '* Adv CLIP Acc@1 {top1_adv_oriclip.avg:.3f}'
              .format(top1_adv_org=top1_adv_org, top1_org=top1_org, top1_ori=top1_ori, top1_adv_oriclip=top1_adv_oriclip))
        logger.info(dataset_name + ' * Adv FT Acc@1 {top1_adv_org.avg:.3f} ' '* Clean Acc@1 {top1_org.avg:.3f}'  '* Original Acc@1 {top1_ori.avg:.3f}' '* Adv CLIP Acc@1 {top1_adv_oriclip.avg:.3f}'
              .format(top1_adv_org=top1_adv_org, top1_org=top1_org, top1_ori=top1_ori, top1_adv_oriclip=top1_adv_oriclip))
        
        acc_all.append(top1_adv_org.avg)
    return np.mean(acc_all)


if __name__ == '__main__':
    main()
