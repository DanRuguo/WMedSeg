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


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask >= threshold).astype(np.uint8)


def percentile_clip(gray: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo = np.percentile(gray, low)
    hi = np.percentile(gray, high)
    gray = np.clip(gray.astype(np.float32), lo, hi)
    gray = (gray - lo) / max(hi - lo, 1e-6)
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


def _largest_bbox(binary: np.ndarray):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]
    return x, y, w, h


def crop_valid_fov(image: np.ndarray, mask: Optional[np.ndarray] = None, gray_thr: int = 8, min_keep_ratio: float = 0.55):
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    valid = (gray > gray_thr).astype(np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    bbox = _largest_bbox(valid)
    if bbox is None:
        return image, mask

    x, y, w, h = bbox
    keep_ratio = (w * h) / float(max(1, image.shape[0] * image.shape[1]))
    if keep_ratio < min_keep_ratio:
        return image, mask

    image = image[y:y + h, x:x + w]
    if mask is not None:
        mask = mask[y:y + h, x:x + w]
    return image, mask


def enhance_endo_luminance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = percentile_clip(l, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image_by_modality(image: np.ndarray, modality: str, is_train: bool = False, cfg=None) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if modality == "ultrasound":
        return image.astype(np.uint8)

    if modality == "mri":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = percentile_clip(gray, 1.0, 99.0)
        return np.stack([gray, gray, gray], axis=-1)

    if modality == "endo":
        image = enhance_endo_luminance(image)
        return image.astype(np.uint8)

    return image.astype(np.uint8)


def resize_image_and_mask(image: np.ndarray, mask: np.ndarray, output_size: Sequence[int]):
    target_h, target_w = int(output_size[0]), int(output_size[1])
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def random_resized_crop_pair(image: np.ndarray, mask: np.ndarray, scale=(0.80, 1.0)):
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


def rotate_pair(image: np.ndarray, mask: np.ndarray, max_deg: float = 10.0):
    angle = random.uniform(-max_deg, max_deg)
    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return image, mask


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_endo_aug(image: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.35:
        image, mask = rotate_pair(image, mask, 10.0)

    if random.random() < 0.40:
        image, mask = random_resized_crop_pair(image, mask, scale=(0.80, 1.0))

    if random.random() < 0.50:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12.0, 12.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.90, 1.15)
        hsv[..., 0] += random.uniform(-4.0, 4.0)
        hsv[..., 0] %= 180.0
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if random.random() < 0.25:
        image = adjust_gamma(image, random.uniform(0.90, 1.10))

    if random.random() < 0.10:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if random.random() < 0.05:
        noise = np.random.normal(0.0, 3.0, size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, mask


def apply_default_aug(image: np.ndarray, mask: np.ndarray):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < 0.25:
        image, mask = rotate_pair(image, mask, 15.0)
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

        if mask.ndim == 3:
            mask = mask[..., 0]

        if modality == "endo":
            image, mask = crop_valid_fov(image, mask, gray_thr=8, min_keep_ratio=0.55)

        image = preprocess_image_by_modality(image, modality, is_train=False, cfg=self.cfg)
        image, mask = resize_image_and_mask(image, mask, self.output_size)

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

        if mask.ndim == 3:
            mask = mask[..., 0]

        if modality == "endo":
            image, mask = crop_valid_fov(image, mask, gray_thr=8, min_keep_ratio=0.55)

        image = preprocess_image_by_modality(image, modality, is_train=True, cfg=self.cfg)
        image, mask = resize_image_and_mask(image, mask, self.output_size)

        if modality == "endo":
            image, mask = apply_endo_aug(image, mask)
        else:
            image, mask = apply_default_aug(image, mask)

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
