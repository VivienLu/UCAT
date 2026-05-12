import os
import json
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

try:
    from pycocotools.coco import COCO
except Exception as e:
    COCO = None


class COCOMultiLabel2017(VisionDataset):
    """COCO 2017 Multi-Label dataset (instances annotations → multi-hot labels).

    Directory layout (example):
        root/
          coco2017/
            annotations/
              instances_train2017.json
              instances_val2017.json
            train2017/
            val2017/

    Args:
        root (str): dataset root directory (the parent of `annotations/`, `train2017/`, `val2017/`).
        split (str): "train" or "val".
        target_types (str or list[str]): keep for compatibility; only "category" is used here.
        transforms / transform / target_transform: same semantics as torchvision VisionDataset.
        prompt_template (str): template for text prompts if needed.
        filter_crowd (bool): whether to ignore iscrowd=1 annotations.
        min_area (float): ignore instances with area <= min_area if >0.
    """

    _VALID_SPLITS = ("train", "val")
    _VALID_TARGET_TYPES = ("category",)

    def __init__(
        self,
        root: str,
        split: str = "val",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        prompt_template: str = "a photo of a {}.",
        filter_crowd: bool = False,
        min_area: float = 0.0,
    ):
        if COCO is None:
            raise ImportError("pycocotools is required. Install with `pip install pycocotools`.")

        split = split.lower()
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}, got {split}")

        if isinstance(target_types, str):
            target_types = [target_types]
        for tt in target_types:
            if tt not in self._VALID_TARGET_TYPES:
                raise ValueError(f"Unsupported target_types: {tt}")

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

        self.split = split
        self.target_types = list(target_types)
        self.prompt_template = prompt_template
        self.filter_crowd = filter_crowd
        self.min_area = float(min_area) if min_area is not None else 0.0

        # Paths
        self._ann_file = os.path.join(self.root, "annotations", f"instances_{split}2017.json")
        self._img_dir = os.path.join(self.root, f"{split}2017")

        if not os.path.isfile(self._ann_file):
            raise FileNotFoundError(f"COCO annotation not found: {self._ann_file}")
        if not os.path.isdir(self._img_dir):
            raise FileNotFoundError(f"COCO image folder not found: {self._img_dir}")

        # Build COCO api & category mappings
        coco = COCO(self._ann_file)
        self.coco = coco

        cats = coco.loadCats(coco.getCatIds())
        # COCO category ids are not 0..79; we remap to contiguous [0..C-1]
        self.cat_id_to_name = {c["id"]: c["name"] for c in cats}
        self.cat_ids_sorted = sorted(self.cat_id_to_name.keys())
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids_sorted)}
        self.idx_to_name = [self.cat_id_to_name[cid] for cid in self.cat_ids_sorted]
        self.num_classes = len(self.idx_to_name)

        # Gather image ids
        self.img_ids = coco.getImgIds()
        # Build (path, multi-hot) lists
        self._images: list[str] = []
        self._labels: list[torch.Tensor] = []

        for img_id in self.img_ids:
            img_info = coco.loadImgs([img_id])[0]
            file_name = img_info["file_name"]
            impath = os.path.join(self._img_dir, file_name)

            # Collect all anns for this image
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            # Build a multi-hot vector
            y = torch.zeros(self.num_classes, dtype=torch.float32)
            for a in anns:
                if self.filter_crowd and a.get("iscrowd", 0) == 1:
                    continue
                if self.min_area > 0.0 and float(a.get("area", 0.0)) <= self.min_area:
                    continue
                cid = a["category_id"]
                if cid in self.cat_id_to_idx:
                    y[self.cat_id_to_idx[cid]] = 1.0

            self._images.append(impath)
            self._labels.append(y)

        # （可选）如果你需要类名生成 prompts：
        self.class_prompts = [self.prompt_template.format(name) for name in self.idx_to_name]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = self._images[index]
        image = Image.open(img_path).convert("RGB")

        target = self._labels[index].clone()

        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target

    # 方便外部使用的辅助接口
    @property
    def classes(self):
        return self.idx_to_name

    @property
    def class_prompts_template(self):
        return self.class_prompts
