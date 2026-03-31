import os
import random
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


CLIP_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ULTRASOUND_DATASETS = {"BUSI", "BUSBRA", "BUSUC", "BUID", "UDIAT"}


def _cfg_get(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
        else:
            return default
    return cur


def to_long_tensor(pic):
    if isinstance(pic, Image.Image):
        pic = np.array(pic, dtype=np.uint8)
    return torch.from_numpy(pic).long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    return corr_images


def random_rotate(image, label, max_deg=20):
    angle = random.uniform(-max_deg, max_deg)
    image = image.rotate(angle, resample=Image.BILINEAR)
    label = label.rotate(angle, resample=Image.NEAREST)
    return image, label


def _base_task_name(task_name: str) -> str:
    return str(task_name).split("_")[0].upper()


def _is_ultrasound(task_name: str) -> bool:
    return _base_task_name(task_name) in ULTRASOUND_DATASETS


def _percentile_normalize(gray: np.ndarray, low=1.0, high=99.0) -> np.ndarray:
    lo = np.percentile(gray, low)
    hi = np.percentile(gray, high)
    if hi <= lo:
        return np.clip(gray / 255.0, 0.0, 1.0).astype(np.float32)
    gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return gray.astype(np.float32)


def _apply_clahe(gray_float: np.ndarray, clip_limit=2.0, tile_size=8) -> np.ndarray:
    gray_u8 = np.clip(gray_float * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    out = clahe.apply(gray_u8).astype(np.float32) / 255.0
    return out


def _augment_ultrasound_gray(gray_float: np.ndarray, cfg) -> np.ndarray:
    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.CLAHE_PROB", 0.3)):
        gray_float = _apply_clahe(
            gray_float,
            clip_limit=float(_cfg_get(cfg, "DATASET.AUG.CLAHE_CLIP_LIMIT", 2.0)),
            tile_size=int(_cfg_get(cfg, "DATASET.AUG.CLAHE_TILE", 8)),
        )

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.GAMMA_PROB", 0.3)):
        gamma_min, gamma_max = _cfg_get(cfg, "DATASET.AUG.GAMMA_RANGE", [0.7, 1.5])
        gamma = random.uniform(float(gamma_min), float(gamma_max))
        gray_float = np.power(np.clip(gray_float, 1e-6, 1.0), gamma)

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.BLUR_PROB", 0.15)):
        k = random.choice([3, 5])
        gray_float = cv2.GaussianBlur(gray_float, (k, k), 0)

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.NOISE_PROB", 0.2)):
        std = float(_cfg_get(cfg, "DATASET.AUG.NOISE_STD", 0.02))
        gray_float = gray_float + np.random.normal(0.0, std, size=gray_float.shape).astype(np.float32)

    return np.clip(gray_float, 0.0, 1.0)


def _prepare_image(image: np.ndarray, task_name: str, cfg, is_train: bool) -> Image.Image:
    use_us_pre = bool(_cfg_get(cfg, "DATASET.ULTRASOUND_PREPROCESS", True)) and _is_ultrasound(task_name)
    if use_us_pre:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.squeeze()
        p_low, p_high = _cfg_get(cfg, "DATASET.PREPROCESS.PERCENTILES", [1.0, 99.0])
        gray = _percentile_normalize(gray.astype(np.float32), float(p_low), float(p_high))
        if is_train and bool(_cfg_get(cfg, "DATASET.AUG.ENABLE", True)):
            gray = _augment_ultrasound_gray(gray, cfg)
        image = np.stack([gray, gray, gray], axis=-1)
        image = (image * 255.0).astype(np.uint8)
        return Image.fromarray(image)

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image.astype(np.uint8))


def _random_resized_crop_pair(image: Image.Image, mask: Image.Image, scale=(0.85, 1.0)):
    w, h = image.size
    s = random.uniform(float(scale[0]), float(scale[1]))
    crop_w = max(8, int(w * s))
    crop_h = max(8, int(h * s))
    if crop_w >= w or crop_h >= h:
        return image, mask
    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)
    image = F.resized_crop(image, top, left, crop_h, crop_w, (h, w), interpolation=transforms.InterpolationMode.BILINEAR)
    mask = F.resized_crop(mask, top, left, crop_h, crop_w, (h, w), interpolation=transforms.InterpolationMode.NEAREST)
    return image, mask


class ValGenerator(object):
    def __init__(self, output_size, task_name=None, cfg=None):
        self.output_size = tuple(output_size)
        self.task_name = task_name or ""
        self.cfg = cfg

    def __call__(self, sample):
        image, mask = sample["image"], sample["ground_truth_mask"]
        text = sample["text_prompt"]

        image = _prepare_image(image, self.task_name, self.cfg, is_train=False)
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        sample["image"] = image
        sample["ground_truth_mask"] = mask
        sample["text_prompt"] = text
        return sample


class RandomGenerator(object):
    def __init__(self, output_size, task_name=None, cfg=None):
        self.output_size = tuple(output_size)
        self.task_name = task_name or ""
        self.cfg = cfg

    def __call__(self, sample):
        image, mask = sample["image"], sample["ground_truth_mask"]
        text = sample["text_prompt"]

        image = _prepare_image(image, self.task_name, self.cfg, is_train=True)
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if bool(_cfg_get(self.cfg, "DATASET.AUG.ENABLE", True)):
            crop_scale = _cfg_get(self.cfg, "DATASET.AUG.RANDOM_RESIZED_CROP_SCALE", [0.85, 1.0])
            if random.random() < float(_cfg_get(self.cfg, "DATASET.AUG.RANDOM_RESIZED_CROP_PROB", 0.35)):
                image, mask = _random_resized_crop_pair(image, mask, scale=crop_scale)
            if random.random() < float(_cfg_get(self.cfg, "DATASET.AUG.HFLIP_PROB", 0.5)):
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() < float(_cfg_get(self.cfg, "DATASET.AUG.ROTATE_PROB", 0.5)):
                image, mask = random_rotate(image, mask, max_deg=float(_cfg_get(self.cfg, "DATASET.AUG.ROTATE_DEG", 15)))

        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        sample["image"] = image
        sample["ground_truth_mask"] = mask
        sample["text_prompt"] = text
        return sample


class DatasetSegmentation(Dataset):
    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: list,
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        image_size: int = 224,
    ) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, "img")
        self.output_path = os.path.join(dataset_path, "label")
        self.one_hot_mask = one_hot_mask
        self.task_name = task_name
        self.data_pairs = [(row["Image"], row["Ground Truth"], row["Description"]) for row in row_text]
        self.data_pairs = sorted(self.data_pairs, key=lambda x: x[0])
        self.joint_transform = joint_transform if joint_transform else (lambda x: x)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_filename, mask_filename, text = self.data_pairs[idx]
        image = cv2.imread(os.path.join(self.input_path, image_filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(os.path.join(self.input_path, image_filename))
        if mask is None:
            raise FileNotFoundError(os.path.join(self.output_path, mask_filename))

        mask[mask < 127] = 0
        mask[mask >= 127] = 1

        image, mask = correct_dims(image, mask)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, "one_hot_mask must be nonnegative"
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        inputs = {
            "image": image,
            "ground_truth_mask": mask,
            "image_name": image_filename,
            "mask_name": mask_filename,
            "text_prompt": text,
            "dataset_name": self.task_name,
        }
        return self.joint_transform(inputs)
