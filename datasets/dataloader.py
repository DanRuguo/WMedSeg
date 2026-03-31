import os
import random
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

CLIP_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

ENDO_DATASETS = {"kvasir", "colondb", "clinicdb", "cvc300", "bkai"}
ULTRASOUND_DATASETS = {"busi", "busbra", "busuc", "buid", "udiat"}
DERM_DATASETS = {"isic", "uwaterlooskincancer"}
MRI_DATASETS = {"btmri", "brisc"}


def _cfg_get(cfg, path, default=None):
    if cfg is None:
        return default
    cur = cfg
    for key in path.split("."):
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _base_task_name(task_name: str) -> str:
    return str(task_name).split("_")[0].lower()


def infer_modality(task_name: str, cfg=None) -> str:
    modality = _cfg_get(cfg, "DATASET.MODALITY", None)
    if modality is not None:
        return str(modality).lower()

    name = _base_task_name(task_name)
    if name in ENDO_DATASETS:
        return "endo"
    if name in ULTRASOUND_DATASETS:
        return "ultrasound"
    if name in DERM_DATASETS:
        return "derm"
    if name in MRI_DATASETS:
        return "mri"
    return "rgb"


def to_long_tensor(pic):
    if isinstance(pic, Image.Image):
        pic = np.array(pic, dtype=np.uint8)
    return torch.from_numpy(np.array(pic, dtype=np.uint8)).long()


def correct_dims(*images):
    out = []
    for img in images:
        if img.ndim == 2:
            out.append(np.expand_dims(img, axis=2))
        else:
            out.append(img)
    return out[0] if len(out) == 1 else out


def percentile_clip(gray: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo = np.percentile(gray, low)
    hi = np.percentile(gray, high)
    gray = np.clip(gray.astype(np.float32), lo, hi)
    gray = (gray - lo) / max(hi - lo, 1e-6)
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return gray


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
    if not bool(_cfg_get(cfg, "DATASET.AUG.ENABLE", True)):
        return np.clip(gray_float, 0.0, 1.0)

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


def preprocess_image_by_modality(image: np.ndarray, modality: str, cfg=None, is_train: bool = False) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if modality == "ultrasound":
        use_us_pre = bool(_cfg_get(cfg, "DATASET.ULTRASOUND_PREPROCESS", True))
        if use_us_pre:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            p_low, p_high = _cfg_get(cfg, "DATASET.PREPROCESS.PERCENTILES", [1.0, 99.0])
            gray = _percentile_normalize(gray.astype(np.float32), float(p_low), float(p_high))
            if is_train:
                gray = _augment_ultrasound_gray(gray, cfg)
            image = np.stack([gray, gray, gray], axis=-1)
            image = (image * 255.0).astype(np.uint8)
            return image
        return image.astype(np.uint8)

    if modality == "mri":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        low, high = _cfg_get(cfg, "DATASET.PREPROCESS.PERCENTILES", [1.0, 99.0])
        gray = percentile_clip(gray, float(low), float(high))
        image = np.stack([gray, gray, gray], axis=-1)
        return image

    return image.astype(np.uint8)


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask >= threshold).astype(np.uint8)


def letterbox_image_and_mask(image: np.ndarray, mask: np.ndarray, output_size: Sequence[int]):
    target_h, target_w = int(output_size[0]), int(output_size[1])
    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    canvas_img[top:top + new_h, left:left + new_w] = image_resized
    canvas_mask[top:top + new_h, left:left + new_w] = mask_resized
    return canvas_img, canvas_mask


def direct_resize_image_and_mask(image: np.ndarray, mask: np.ndarray, output_size: Sequence[int]):
    target_h, target_w = int(output_size[0]), int(output_size[1])
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def resize_image_and_mask(image: np.ndarray, mask: np.ndarray, output_size: Sequence[int], modality: str, cfg=None):
    resize_mode = _cfg_get(cfg, "DATASET.RESIZE_MODE", None)
    if resize_mode is None:
        resize_mode = "direct" if modality == "ultrasound" else "letterbox"
    resize_mode = str(resize_mode).lower()

    if resize_mode == "letterbox":
        return letterbox_image_and_mask(image, mask, output_size)
    return direct_resize_image_and_mask(image, mask, output_size)


def random_resized_crop_pair(image: np.ndarray, mask: np.ndarray, scale=(0.85, 1.0)):
    h, w = image.shape[:2]
    ratio = random.uniform(scale[0], scale[1])
    crop_h = max(1, int(h * ratio))
    crop_w = max(1, int(w * ratio))
    if crop_h >= h or crop_w >= w:
        return image, mask
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    image = image[top:top + crop_h, left:left + crop_w]
    mask = mask[top:top + crop_h, left:left + crop_w]
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def rotate_pair(image: np.ndarray, mask: np.ndarray, max_deg: float = 15.0):
    angle = random.uniform(-max_deg, max_deg)
    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_endo_aug(image: np.ndarray, mask: np.ndarray, cfg=None):
    if not bool(_cfg_get(cfg, "DATASET.AUG.ENABLE", True)):
        return image, mask

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.HFLIP_PROB", 0.5)):
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.ROTATE_PROB", 0.5)):
        image, mask = rotate_pair(image, mask, float(_cfg_get(cfg, "DATASET.AUG.ROTATE_DEG", 15)))

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.RANDOM_RESIZED_CROP_PROB", 0.35)):
        scale = _cfg_get(cfg, "DATASET.AUG.RANDOM_RESIZED_CROP_SCALE", [0.85, 1.0])
        image, mask = random_resized_crop_pair(image, mask, scale=(float(scale[0]), float(scale[1])))

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.COLOR_JITTER_PROB", 0.5)):
        contrast = float(_cfg_get(cfg, "DATASET.AUG.CONTRAST", 0.15))
        brightness = float(_cfg_get(cfg, "DATASET.AUG.BRIGHTNESS", 0.15))
        alpha = random.uniform(max(0.7, 1.0 - contrast), 1.0 + contrast)
        beta = random.uniform(-255.0 * brightness * 0.25, 255.0 * brightness * 0.25)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = float(_cfg_get(cfg, "DATASET.AUG.SATURATION", 0.15))
        hue = float(_cfg_get(cfg, "DATASET.AUG.HUE", 0.02))
        hsv[..., 1] *= random.uniform(max(0.7, 1.0 - sat), 1.0 + sat)
        hsv[..., 0] += random.uniform(-180.0 * hue, 180.0 * hue)
        hsv[..., 0] %= 180.0
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.GAMMA_PROB", 0.2)):
        gr = _cfg_get(cfg, "DATASET.AUG.GAMMA_RANGE", [0.85, 1.15])
        image = adjust_gamma(image, random.uniform(float(gr[0]), float(gr[1])))

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.BLUR_PROB", 0.10)):
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.NOISE_PROB", 0.05)):
        std = 255.0 * float(_cfg_get(cfg, "DATASET.AUG.NOISE_STD", 0.01))
        noise = np.random.normal(0.0, std, size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, mask


def apply_ultrasound_aug(image: np.ndarray, mask: np.ndarray, cfg=None):
    if not bool(_cfg_get(cfg, "DATASET.AUG.ENABLE", True)):
        return image, mask

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.RANDOM_RESIZED_CROP_PROB", 0.35)):
        scale = _cfg_get(cfg, "DATASET.AUG.RANDOM_RESIZED_CROP_SCALE", [0.85, 1.0])
        image, mask = random_resized_crop_pair(image, mask, scale=(float(scale[0]), float(scale[1])))

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.HFLIP_PROB", 0.5)):
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < float(_cfg_get(cfg, "DATASET.AUG.ROTATE_PROB", 0.5)):
        image, mask = rotate_pair(image, mask, float(_cfg_get(cfg, "DATASET.AUG.ROTATE_DEG", 15)))

    return image, mask


def apply_derm_or_mri_aug(image: np.ndarray, mask: np.ndarray, cfg=None):
    if not bool(_cfg_get(cfg, "DATASET.AUG.ENABLE", True)):
        return image, mask
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < 0.4:
        image, mask = rotate_pair(image, mask, 20.0)
    return image, mask


class ValGenerator(object):
    def __init__(self, output_size, cfg=None, task_name: Optional[str] = None):
        self.output_size = output_size
        self.cfg = cfg
        self.task_name = task_name

    def __call__(self, sample):
        image, mask = sample["image"], sample["ground_truth_mask"]
        task_name = sample.get("dataset_name", self.task_name)
        modality = infer_modality(task_name, self.cfg)

        image = preprocess_image_by_modality(image, modality, self.cfg, is_train=False)
        if mask.ndim == 3:
            mask = mask[..., 0]
        image, mask = resize_image_and_mask(image, mask, self.output_size, modality, self.cfg)

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        sample["image"] = image
        sample["ground_truth_mask"] = mask
        return sample


class RandomGenerator(object):
    def __init__(self, output_size, cfg=None, task_name: Optional[str] = None):
        self.output_size = output_size
        self.cfg = cfg
        self.task_name = task_name

    def __call__(self, sample):
        image, mask = sample["image"], sample["ground_truth_mask"]
        task_name = sample.get("dataset_name", self.task_name)
        modality = infer_modality(task_name, self.cfg)

        image = preprocess_image_by_modality(image, modality, self.cfg, is_train=True)
        if mask.ndim == 3:
            mask = mask[..., 0]
        image, mask = resize_image_and_mask(image, mask, self.output_size, modality, self.cfg)

        if modality == "endo":
            image, mask = apply_endo_aug(image, mask, self.cfg)
        elif modality == "ultrasound":
            image, mask = apply_ultrasound_aug(image, mask, self.cfg)
        else:
            image, mask = apply_derm_or_mri_aug(image, mask, self.cfg)

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        sample["image"] = image
        sample["ground_truth_mask"] = mask
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
        cfg=None,
    ) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, "img")
        self.output_path = os.path.join(dataset_path, "label")
        self.one_hot_mask = one_hot_mask
        self.task_name = task_name
        self.cfg = cfg
        self.data_pairs = [(row["Image"], row["Ground Truth"], row["Description"]) for row in row_text]
        self.data_pairs = sorted(self.data_pairs, key=lambda x: x[0])
        self.joint_transform = joint_transform if joint_transform is not None else (lambda x: x)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_filename, mask_filename, text = self.data_pairs[idx]
        image_path = os.path.join(self.input_path, image_filename)
        mask_path = os.path.join(self.output_path, mask_filename)

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(image_path)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)

        threshold = int(_cfg_get(self.cfg, "DATASET.MASK_THRESHOLD", 127))
        mask = binarize_mask(mask, threshold=threshold)
        image, mask = correct_dims(image, mask)

        inputs = {
            "image": image,
            "ground_truth_mask": mask,
            "image_name": image_filename,
            "mask_name": mask_filename,
            "text_prompt": text,
            "dataset_name": self.task_name,
        }
        inputs = self.joint_transform(inputs)

        if self.one_hot_mask:
            m = inputs["ground_truth_mask"]
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            inputs["ground_truth_mask"] = torch.zeros(
                (self.one_hot_mask, m.shape[-2], m.shape[-1]),
                dtype=torch.float32,
            ).scatter_(0, m.long(), 1)

        return inputs
