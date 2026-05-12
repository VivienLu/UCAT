# replace/clip.py  （新的实现）

import torch
from transformers import AlignModel, AlignProcessor

_align_processor = None   # 全局保存，给 tokenize 用

def load(model_name_or_path: str, device, jit=False, prompt_len=0):
    """
    模拟 openai.clip.load 的返回：
    返回 (model, preprocess_fn)
    """
    global _align_processor
    _align_processor = AlignProcessor.from_pretrained(model_name_or_path)
    model = AlignModel.from_pretrained(model_name_or_path)
    model.to(device)

    # preprocess：把一张 PIL image → clip 风格的 tensor（C,H,W）
    def preprocess(pil_img):
        # AlignProcessor 会返回 shape [1, C, H, W]
        inputs = _align_processor(images=pil_img, return_tensors="pt")
        # pixel_values = inputs["pixel_values"][0]
        pixel_values = inputs["pixel_values"]
        return pixel_values

    return model, preprocess


def tokenize(texts):
    """
    模拟 openai.clip.tokenize，返回 input_ids（[B, L]）
    """
    assert _align_processor is not None, "call load() before tokenize()"
    if isinstance(texts, str):
        texts = [texts]
    inputs = _align_processor(text=texts, return_tensors="pt", padding=True)
    return inputs["input_ids"]
