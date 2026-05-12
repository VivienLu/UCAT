import torch
from utils import one_hot_embedding
from models.model import *
import torch.nn.functional as F
from autoattack import AutoAttack
# from torchattacks.attacks.autoattack import AutoAttack
import functools
import gc
import torch.nn as nn


lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(prompter, model, add_prompter, X, target, text_tokens, alpha,
               attack_iters, norm, device, args, restarts=1, early_stop=True, epsilon=0):

    criterion = nn.BCEWithLogitsLoss(reduction="sum").to(device)

    delta = torch.zeros_like(X, device=device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / (n + 1e-10) * epsilon
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    for _ in range(attack_iters):
        _images = clip_img_preprocessing(X + delta, device)
        prompted_images = prompter(_images) if prompter is not None else _images
        prompt_token = add_prompter() if add_prompter is not None else None

        logits, _, _ = multiGPU_CLIP(model, prompted_images, text_tokens, target, device, prompt_token)

        # 多标签损失：BCE
        loss = criterion(logits, target.float())
        # PGD是“最大化”损失（让模型犯错），所以对delta的梯度上升
        loss.backward()
        grad = delta.grad.detach()

        if norm == "l_inf":
            delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g = grad
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            delta.data = (delta + scaled_g * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(delta)

        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.grad.zero_()

    return delta.detach()

def attack_CW(prompter, model, add_prompter, X, target, text_tokens, alpha,
              attack_iters, norm, device, restarts=1, early_stop=True, epsilon=0, multilabel=False):
    delta = torch.zeros_like(X).to(device)
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta, device))
        prompt_token = add_prompter()

        # output, _, _, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)
        output, _, _ = multiGPU_CLIP(model, prompted_images, text_tokens,target, device, prompt_token)
        num_class = output.size(1)

        if multilabel:
            label_mask = target
        else:
            label_mask = one_hot_embedding(target, num_class, device)
        label_mask = label_mask.to(device)

        # correct_logit = torch.sum(label_mask * output, dim=1)
        # wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)
        
        # avoid div0 if some sample accidentally has 0 positives
        pos_counts = label_mask.sum(dim=1).clamp(min=1.0)   # shape (B,)

        # correct_logit: per-sample mean over positive logits (stabilizes varying #positives)
        correct_logit = (label_mask * output).sum(dim=1) / pos_counts    # (B,)

        # wrong_logit: strongest non-positive logit (use masked_fill for numeric stability)
        neg_logits = output.clone()
        mask = label_mask.bool()
        fill_val = torch.finfo(neg_logits.dtype).min  # float16: -65504.0, float32: -3.4e38
        neg_logits = neg_logits.masked_fill(mask, fill_val)
        # neg_logits = neg_logits.masked_fill(mask, float('-inf'))


        wrong_logit = neg_logits.max(dim=1).values    # (B,)

        # loss = criterion(output, target)
        loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

from autoattack.autopgd_base import APGDAttack

class APGD_BCE(APGDAttack):
    """
    Auto-PGD with BCE loss (for multi-label classification).
    """

    def __init__(self, model, n_restarts=1, n_iter=100, eps=8/255, 
                 norm='Linf', seed=0, loss='bce', eot_iter=1,
                 rho=.75, device='cuda'):
        super().__init__(model, n_restarts=n_restarts, n_iter=n_iter,
                         eps=eps, norm=norm, seed=seed,
                         loss=loss, eot_iter=eot_iter, rho=rho, device=device)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def compute_loss(self, x, y):
        """
        Override to use BCEWithLogitsLoss for multi-label one-hot target.
        x: logits [B, C]
        y: multi-hot labels [B, C]
        """
        return self.criterion(x, y.float())

class MultiLabelDLRLoss(torch.nn.Module):
    """
    target: [B, C], 0/1 multi-hot
    logits: [B, C]
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        # logits: [B, C]
        # target: [B, C] (0/1)
        B, C = logits.shape
        device = logits.device

        # 最大正类 logit
        pos_mask = target.bool()
        neg_mask = ~pos_mask

        # 防止全 0 的情况（某张图没有正类标签）
        pos_logits = torch.where(pos_mask, logits, torch.full_like(logits, -1e9))
        neg_logits = torch.where(neg_mask, logits, torch.full_like(logits, -1e9))

        z_pos = pos_logits.max(dim=1).values   # [B]
        z_neg = neg_logits.max(dim=1).values   # [B]

        # 全局最大/最小
        z_max = logits.max(dim=1).values
        z_min = logits.min(dim=1).values
        denom = (z_max - z_min + 1e-12)

        dlr = - (z_pos - z_neg) / denom
        return dlr.mean()

from autoattack.autopgd_base import APGDAttack

class APGD_DLR_ML(APGDAttack):
    """
    Auto-PGD with MultiLabel DLR loss.
    """
    def __init__(self, model, n_restarts=1, n_iter=100, eps=8/255,
                 norm='Linf', seed=0, eot_iter=1, rho=.75, device='cuda'):
        super().__init__(model, n_restarts=n_restarts, n_iter=n_iter,
                         eps=eps, norm=norm, seed=seed,
                         loss='dlr-ml', eot_iter=eot_iter, rho=rho, device=device)
        self.criterion = MultiLabelDLRLoss()

    def compute_loss(self, x, y):
        return self.criterion(x, y)


class AutoAttackML(AutoAttack):
    def __init__(self, model, norm='Linf', eps=1/255, version='rand', device='cuda', verbose=True):
        super().__init__(model, norm=norm, eps=eps, version=version, device=device, verbose=verbose)

        self.attacks['apgd-bce'] = APGD_BCE(model, n_iter=100, eps=eps, norm=norm, device=device)
        self.attacks['apgd-dlr-ml'] = APGD_DLR_ML(model, n_iter=100, eps=eps, norm=norm, device=device)

        self.attacks_to_run = ['apgd-bce', 'apgd-dlr-ml']


def attack_auto(model, images, target, text_tokens, prompter, add_prompter,device,
                         attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):

    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens, target=target, device=device,
        prompter=prompter, add_prompter=add_prompter
    )

    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False, device=device)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv
