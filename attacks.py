import torch
from utils import one_hot_embedding
from models.model import *
import torch.nn.functional as F
from autoattack import AutoAttack
# from torchattacks.attacks.autoattack import AutoAttack
import functools
import gc
import torch.nn as nn
from composite_attack import CompositeAttack


lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(prompter, model, add_prompter,  X, target, text_tokens, alpha,
               attack_iters, norm,device, args, restarts=1, early_stop=True, epsilon=0,
               eva_clip=False, slip=False):
    delta = torch.zeros_like(X).cuda(device)
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
    for i in range(attack_iters):
        _images = clip_img_preprocessing(X + delta,device)
        if prompter is not None:
            prompted_images = prompter(_images)
        else:
            prompted_images = _images
        prompt_token = add_prompter() if add_prompter is not None else None

        output, _, _= multiGPU_CLIP(model, prompted_images, text_tokens,target,device, prompt_token, eva_clip=eva_clip, slip=slip)
        CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
        # print('output', output.shape, output.dtype)
        # print('target', target.shape, target.dtype)
        loss = CrossEntropyLoss(output, target)
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


def attack_CW(prompter, model, add_prompter, X, target, text_tokens, alpha,
              attack_iters, norm, device, restarts=1, 
              early_stop=True, epsilon=0, multilabel=False,
              eval_clip=False, slip=False):
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
        # output, _, _ = multiGPU_CLIP(model, prompted_images, text_tokens,target, device, prompt_token, eva_clip=eva_clip, slip=slip)
        output, _, _ = multiGPU_CLIP(model, prompted_images, text_tokens,target, device, prompt_token)
        num_class = output.size(1)

        if multilabel:
            label_mask = target
        else:
            label_mask = one_hot_embedding(target, num_class, device)
        label_mask = label_mask.to(device)

        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, axis=1)

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

def attack_auto(model, images, target, text_tokens, prompter, add_prompter,device,
                         attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0, eva_clip=False, slip=False):

    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens, target=target, device=device,
        prompter=prompter, add_prompter=add_prompter, eva_clip=eva_clip, slip=slip,
    )

    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False, device=device)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    # adversary.apgd.n_iter = 2

    # wrapped = CLIPLogitsWrapper(model, text_tokens, device, prompter, add_prompter)
    # atk = AutoAttack(wrapped, norm='Linf', eps=epsilon, version='rand', n_classes=text_tokens.shape[0], seed=0, verbose=False)
    # x_adv = atk(images, target)
    return x_adv


class CLIPLogitsWrapper(nn.Module):
    def __init__(self, base_model, text_tokens, device, prompter=None, add_prompter=None):
        super().__init__()
        self.base_model = base_model          # 真正的模型，保证 .eval/.train/parameters 可用
        self.text_tokens = text_tokens
        self.device = device
        self.prompter = prompter
        self.add_prompter = add_prompter

    def forward(self, images):
        return multiGPU_CLIP_image_logits(
            images,
            model=self.base_model,
            text_tokens=self.text_tokens,
            target=None,
            device=self.device,
            prompter=self.prompter,
            add_prompter=self.add_prompter
        )
        

class VLMForwardModule(nn.Module):
    """
    与 AutoAttack 的 forward_pass 保持一致：
    forward(images) -> logits (B, C)
    """
    def __init__(self, forward_pass_callable):
        super().__init__()
        self.forward_pass = forward_pass_callable

    @torch.no_grad()  # 这里只返回 logits，不保留图；CAA 内部会对图像做 requires_grad
    def forward(self, images):
        # multiGPU_CLIP_image_logits 返回 logits_per_image, 只取第一个
        logits = self.forward_pass(images)
        return logits

def attack_caa(model, images, target, text_tokens, prompter, add_prompter, device,
               enabled_attack=None, epsilon=0.0, inner_steps=10,
               order_schedule='fixed', start_num=1, multiple_rand_start=False,
               dataset_hint='imagenet'):
    """
    参数说明：
      - enabled_attack: 选择哪些算子参与组合（索引见下）
          0=hue, 1=saturation, 2=rotation, 3=brightness, 4=contrast, 5=linf(加性噪声)
        例如：[5] 只做 L_inf（最贴近 PGD/CW 评测），或 [5,2,3,4] 做组合。
      - epsilon: L_inf 半径（例如 8/255）
      - inner_steps: 每个算子的 PGD 内迭代步数（与 args.test_numsteps 对齐）
      - order_schedule: 'fixed' | 'random' | 'scheduled'
      - dataset_hint: 仅影响 CAA 默认 eps 池；我们会手动覆盖 linf 的 eps
    """
    if enabled_attack is None:
        enabled_attack = [5]  # 默认只用 L_inf

    # 复用你 AutoAttack 的封装方式：images -> logits
    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits_caa,
        model=model, text_tokens=text_tokens, target=target, device=device,
        prompter=prompter, add_prompter=add_prompter
    )
    # vlm_module = VLMForwardModule(forward_pass).to(device)
    # print(vlm_module.forward.__closure__)  # 若是 functools.partial，看看闭包里有无 no_grad

    # 构造 CAA 攻击器
    attacker = CompositeAttack(
        model=forward_pass,
        enabled_attack=enabled_attack,
        mode='eval',  # 评测模式，命中即 early-stop
        dataset=dataset_hint,
        start_num=start_num,
        iter_num=5,                 # 仅在 'scheduled' 顺序下生效；可保留默认
        inner_iter_num=inner_steps, # 每个算子的步数
        multiple_rand_start=multiple_rand_start,
        order_schedule=order_schedule,
        device=device,
    ).to(device)

    # 覆盖 L_inf 通道的 eps（索引 5）
    eps = float(epsilon)
    if 5 in enabled_attack:
        attacker.eps_pool[5] = torch.tensor([-eps, eps], device=device)

    # 如你启用了几何/颜色算子，也可在此手动限制其幅度范围（可选）：
    # 例如：旋转最大 ±5 度、亮度/对比度幅度 0.1
    # if 2 in enabled_attack:  # rotation
    #     attacker.eps_pool[2] = torch.tensor([-5.0, 5.0], device=device)
    # if 3 in enabled_attack:  # brightness
    #     attacker.eps_pool[3] = torch.tensor([-0.1, 0.1], device=device)
    # if 4 in enabled_attack:  # contrast
    #     attacker.eps_pool[4] = torch.tensor([-0.1, 0.1], device=device)

    # 运行攻击（CAA: forward(images, labels) -> adv_images）
    target = target.to(device).long()
    images = images.to(device)
    adv_images = attacker(images, target)  # 已裁剪到 [0,1]
    return adv_images


import torch
import torch.nn.functional as F
import functools

# ---- 你的现有封装（保持使用）----
# def multiGPU_CLIP_image_logits(images, model, text_tokens, target, device, prompter=None, add_prompter=None):
#     ...
#     return logits_per_image, logits_per_text, scale_text_embed

@torch.no_grad()
def _zerolike(x):
    return torch.zeros_like(x)

def _project_linf(x, x0, eps):
    # clamp to L∞ ball around x0
    eta = torch.clamp(x - x0, min=-eps, max=eps)
    return torch.clamp(x0 + eta, 0., 1.)

# --- 工具函数：返回 per-sample loss 和 logits ---
def _get_loss(forward_pass, x, y, *, per_sample=False):
    logits = forward_pass(x)
    if per_sample:
        loss_vec = F.cross_entropy(logits, y, reduction='none')  # (B,)
        return loss_vec, logits
    else:
        loss = F.cross_entropy(logits, y, reduction='mean')      # 标量
        return loss, logits

def attack_a3(
    model, images, target, text_tokens, prompter, add_prompter, device,
    epsilon=8/255,
    num_steps=20,             # PGD 步数（A³里的 inner steps）
    num_restarts=5,           # 重启次数
    step_size=None,           # 步长；默认设成 2/255 或 ε/4
    osd_start=0.0,            # 每次重启前丢弃比例起点（例如 0.0 / 0.1）
    osd_inc=0.1,              # 每轮重启丢弃比例增量
    early_stop=True,          # 成功样本可早停（非严格 PGD-k，但更贴近 A³ 加速策略）
    rand_init=True,           # 随机起点
    use_adi=True,             # 自适应方向初始化
):
    """
    返回: x_adv, 形状与 images 相同, 落在 [0,1]
    """
    images = images.to(device)
    target = target.to(device)

    if step_size is None:
        step_size = min(1/255, epsilon/4)

    # 你的 VLM 前向，保持与 AutoAttack 一致： images -> logits
    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens, target=target, device=device,
        prompter=prompter, add_prompter=add_prompter
    )

    B = images.size(0)
    x0 = images.detach()
    best_adv = x0.clone()
    best_succ = torch.zeros(B, dtype=torch.bool, device=device)
    best_loss = torch.full((B,), -1e9, device=device)

    # ADI: 累积成功方向的符号偏置（简单实现）
    # dir_sum 形状与 images 相同；存每次重启成功样本的 sign(adv - x0) 的和
    dir_sum = torch.zeros_like(images)

    # --- 开始前：得到用于 OSD 排序的 per-sample 基线损失 ---
    with torch.enable_grad():
        # 注意：x0 需要 requires_grad=True 才能在相同图里再用（即便这里只取 detach 出去排序）
        loss_vec, _ = _get_loss(forward_pass, x0.detach().requires_grad_(True), target, per_sample=True)
    base_loss = loss_vec.detach()  # 形状 (B,)

    for r in range(num_restarts):
        # === OSD：决定这一轮哪些样本参与 ===
        drop_ratio = min(1.0, osd_start + r * osd_inc)
        k_keep = int(B * (1.0 - drop_ratio))
        k_keep = max(1, min(B, k_keep))            # <-- 防越界

        keep_idx = torch.topk(base_loss, k_keep, largest=True).indices   # <-- 现在 OK
        active_mask = torch.zeros(B, dtype=torch.bool, device=device)
        active_mask[keep_idx] = True

        # === 初始化（随机起点 + ADI 偏置） ===
        if rand_init:
            delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        else:
            delta = torch.zeros_like(images)

        if use_adi and r > 0:
            # 用累计方向的符号偏置做一个小幅度的启动偏置（不超过 eps 的 25%）
            bias = torch.sign(dir_sum).detach()
            delta = torch.clamp(delta + 0.25 * epsilon * bias, min=-epsilon, max=epsilon)

        x = torch.clamp(x0 + delta, 0., 1.)
        # 标记这一轮仍未成功的样本
        this_alive = active_mask.clone()

        # === PGD 循环 ===
        for t in range(num_steps):
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                loss, logits = _get_loss(forward_pass, x, target)

            # 只对仍在攻的样本反传（可选优化）
            if this_alive.any():
                grad = torch.autograd.grad(loss.sum(), x, retain_graph=False, create_graph=False)[0]
            else:
                grad = _zerolike(x)

            # 对“已成功”的样本禁用更新（早停）
            if early_stop:
                # 判断成功（非定向攻击，预测≠标签）
                pred = logits.argmax(1)
                succ_now = (pred != target)
                # 只在未成功的样本上更新
                update_mask = torch.logical_and(this_alive, ~succ_now)
                this_alive = torch.logical_and(this_alive, ~succ_now)
            else:
                update_mask = this_alive

            # PGD 步
            if update_mask.any():
                step = step_size * torch.sign(grad)
                step = torch.where(update_mask.view(B, 1, 1, 1), step, torch.zeros_like(step))
                x = x + step
                x = _project_linf(x, x0, epsilon)
            else:
                # 没有活跃样本需要更新，也要保证数值域
                x = _project_linf(x, x0, epsilon)

            # 早停：这一轮活跃样本全成功就跳出
            if early_stop and (~this_alive).all():
                break

        # === 记录这一轮的成功情况，用于 ADI ===
        with torch.no_grad():
            logits = forward_pass(x)
            pred = logits.argmax(1)
            succ = (pred != target)

            # 重新计算 per-sample loss 用于挑更“容易”的样本
            with torch.enable_grad():
                l_final_vec, _ = _get_loss(forward_pass, x.detach().requires_grad_(True), target, per_sample=True)

            # 维护“历史最佳” per-sample loss（谁更大留谁）
            better = l_final_vec.detach() > best_loss    # best_loss 也需是 (B,)，初始化时用 -1e9 * torch.ones(B)
            best_loss = torch.where(better, l_final_vec.detach(), best_loss)
            best_adv  = torch.where(better.view(B,1,1,1), x.detach(), best_adv)
            best_succ = torch.logical_or(best_succ, succ)

            # 仅在本轮参与的样本上刷新 base_loss，其他样本保持原值
            base_loss = base_loss.clone()
            base_loss[keep_idx] = best_loss[keep_idx]

            # ADI：累积成功方向（只对这轮成功的样本）
            if succ.any():
                d = torch.sign(x.detach() - x0)
                d = torch.where(succ.view(B,1,1,1), d, torch.zeros_like(d))
                dir_sum = dir_sum + d

        # 更新 OSD 的基准损失（下一轮排序用）
        # 这里用“当前对抗样本的损失”更新“活跃样本”的基础值，其余保持原值
        base_loss = base_loss.clone()
        base_loss[keep_idx] = best_loss[keep_idx]

    # 返回最终最坏的对抗样本（若没成功也返回最后的 x0+δ）
    # 若你希望“没成功时就返回原图”，可以改成：best_adv = torch.where(best_succ.view(B,1,1,1), best_adv, x0)
    return best_adv
