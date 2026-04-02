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
OPENCLIP_NORMALIZE = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

ENDO_DATASETS = {"kvasir", "colondb", "clinicdb", "cvc300", "bkai"}
ULTRASOUND_DATASETS = {"busi", "busbra", "busuc", "buid", "udiat"}
DERM_DATASETS = {"isic", "uwaterlooskincancer"}
MRI_DATASETS = {"btmri", "brisc"}


def _cfg_get(cfg, path, default=None):
    if cfg is None:
        return default
    cur = cfg
    for key in path.split('.'):
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _pp_get(cfg, key, default=None):
    return _cfg_get(cfg, f'DATASET.PREPROCESS.{key}', default)


def _aug_get(cfg, key, default=None):
    return _cfg_get(cfg, f'TRAIN.AUG.{key}', default)


def _base_task_name(task_name: str) -> str:
    return str(task_name).split('_')[0].lower()


def infer_modality(task_name: str, cfg=None) -> str:
    modality = _cfg_get(cfg, 'DATASET.MODALITY', None)
    if modality is not None:
        return str(modality).lower()

    name = _base_task_name(task_name)
    if name in ENDO_DATASETS:
        return 'endo'
    if name in ULTRASOUND_DATASETS:
        return 'ultrasound'
    if name in DERM_DATASETS:
        return 'derm'
    if name in MRI_DATASETS:
        return 'mri'
    return 'rgb'


def get_normalize_transform(cfg=None):
    mode = str(_pp_get(cfg, 'NORMALIZE', 'imagenet_clip')).lower()
    if mode in {'open_clip', 'openai_clip', 'openai'}:
        return OPENCLIP_NORMALIZE
    return CLIP_NORMALIZE


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


def crop_valid_fov(image: np.ndarray, mask: Optional[np.ndarray] = None, gray_thr: int = 8, min_keep_ratio: float = 0.55, return_meta: bool = False):
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    crop_meta = {
        'crop_xywh': np.array([0, 0, orig_w, orig_h], dtype=np.int32),
        'crop_applied': np.array([0], dtype=np.int32),
    }
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    valid = (gray > gray_thr).astype(np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    bbox = _largest_bbox(valid)
    if bbox is None:
        return (image, mask, crop_meta) if return_meta else (image, mask)

    x, y, w, h = bbox
    keep_ratio = (w * h) / float(max(1, image.shape[0] * image.shape[1]))
    if keep_ratio < min_keep_ratio:
        return (image, mask, crop_meta) if return_meta else (image, mask)

    crop_meta['crop_xywh'] = np.array([x, y, w, h], dtype=np.int32)
    crop_meta['crop_applied'] = np.array([1], dtype=np.int32)
    image = image[y:y + h, x:x + w]
    if mask is not None:
        mask = mask[y:y + h, x:x + w]
    return (image, mask, crop_meta) if return_meta else (image, mask)


def crop_dark_frame(image: np.ndarray, mask: Optional[np.ndarray] = None, gray_thr: int = 12, min_keep_ratio: float = 0.60, return_meta: bool = False):
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    crop_meta = {
        'crop_xywh': np.array([0, 0, orig_w, orig_h], dtype=np.int32),
        'crop_applied': np.array([0], dtype=np.int32),
    }
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    valid = (gray > gray_thr).astype(np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    bbox = _largest_bbox(valid)
    if bbox is None:
        return (image, mask, crop_meta) if return_meta else (image, mask)

    x, y, w, h = bbox
    keep_ratio = (w * h) / float(max(1, image.shape[0] * image.shape[1]))
    if keep_ratio < min_keep_ratio:
        return (image, mask, crop_meta) if return_meta else (image, mask)

    crop_meta['crop_xywh'] = np.array([x, y, w, h], dtype=np.int32)
    crop_meta['crop_applied'] = np.array([1], dtype=np.int32)
    image = image[y:y + h, x:x + w]
    if mask is not None:
        mask = mask[y:y + h, x:x + w]
    return (image, mask, crop_meta) if return_meta else (image, mask)


def enhance_endo_luminance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = percentile_clip(l, 1.0, 99.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def derm_color_constancy(image: np.ndarray, power: float = 6.0, eps: float = 1e-6) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    img = np.maximum(img, eps)
    illum = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    illum = illum / max(np.linalg.norm(illum), eps)
    illum = illum * np.sqrt(3.0)
    img = img / illum.reshape(1, 1, 3)
    img = img / max(np.percentile(img, 99.0), eps)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def enhance_derm_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image_by_modality(image: np.ndarray, modality: str, is_train: bool = False, cfg=None) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if modality == 'ultrasound':
        return image.astype(np.uint8)

    if modality == 'mri':
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = percentile_clip(gray, 1.0, 99.0)
        return np.stack([gray, gray, gray], axis=-1)

    if modality == 'endo':
        do_enhance = bool(_pp_get(cfg, 'ENDO_ENHANCE_TRAIN', False)) if is_train else bool(_pp_get(cfg, 'ENDO_ENHANCE_EVAL', False))
        if do_enhance:
            image = enhance_endo_luminance(image)
        return image.astype(np.uint8)

    if modality == 'derm':
        if bool(_pp_get(cfg, 'DERM_COLOR_CONSTANCY', True)):
            image = derm_color_constancy(image, power=float(_pp_get(cfg, 'DERM_COLOR_CONSTANCY_POWER', 6.0)))
        do_enhance = bool(_pp_get(cfg, 'DERM_ENHANCE_TRAIN', False)) if is_train else bool(_pp_get(cfg, 'DERM_ENHANCE_EVAL', False))
        if do_enhance:
            image = enhance_derm_contrast(image)
        return image.astype(np.uint8)

    return image.astype(np.uint8)


def _cv2_border_type(pad_mode: str):
    pad_mode = str(pad_mode).lower()
    if pad_mode == 'reflect':
        return cv2.BORDER_REFLECT_101
    if pad_mode == 'edge':
        return cv2.BORDER_REPLICATE
    return cv2.BORDER_CONSTANT


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: Sequence[int],
    keep_aspect_ratio: bool = False,
    pad_mode: str = 'edge',
    return_meta: bool = False,
):
    target_h, target_w = int(output_size[0]), int(output_size[1])
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    meta = {
        'orig_hw': np.array([orig_h, orig_w], dtype=np.int32),
        'resized_hw': np.array([target_h, target_w], dtype=np.int32),
        'pad_tblr': np.array([0, 0, 0, 0], dtype=np.int32),
        'keep_aspect_ratio': np.array([1 if keep_aspect_ratio else 0], dtype=np.int32),
    }

    if not keep_aspect_ratio:
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        if return_meta:
            return image, mask, meta
        return image, mask

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f'Invalid image size: {(h, w)}')

    scale = min(target_h / float(h), target_w / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    border_type = _cv2_border_type(pad_mode)
    image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, border_type, value=0)
    mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    meta['resized_hw'] = np.array([new_h, new_w], dtype=np.int32)
    meta['pad_tblr'] = np.array([pad_top, pad_bottom, pad_left, pad_right], dtype=np.int32)
    if return_meta:
        return image, mask, meta
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


def random_focus_crop_pair(image: np.ndarray, mask: np.ndarray, scale=(0.75, 1.0), min_fg_keep: float = 0.90, tries: int = 10):
    h, w = image.shape[:2]
    fg = np.argwhere(mask > 0)
    if fg.size == 0:
        return random_resized_crop_pair(image, mask, scale=scale)

    ys, xs = fg[:, 0], fg[:, 1]
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    fg_sum = max(1, int(mask.sum()))

    for _ in range(max(1, int(tries))):
        ratio = random.uniform(scale[0], scale[1])
        crop_h = max(int(round(h * ratio)), (y1 - y0 + 1) + 4)
        crop_w = max(int(round(w * ratio)), (x1 - x0 + 1) + 4)
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        min_top = max(0, y1 - crop_h + 1)
        max_top = min(y0, h - crop_h)
        min_left = max(0, x1 - crop_w + 1)
        max_left = min(x0, w - crop_w)

        top = min_top if max_top < min_top else random.randint(min_top, max_top)
        left = min_left if max_left < min_left else random.randint(min_left, max_left)

        crop_img = image[top:top + crop_h, left:left + crop_w]
        crop_mask = mask[top:top + crop_h, left:left + crop_w]
        if crop_mask.sum() < min_fg_keep * fg_sum:
            continue

        crop_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return crop_img, crop_mask

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


def apply_rect_dropout(image: np.ndarray, max_regions: int = 2, area_ratio=(0.01, 0.05)) -> np.ndarray:
    h, w = image.shape[:2]
    mean_color = tuple(int(x) for x in image.reshape(-1, image.shape[-1]).mean(axis=0))
    num_regions = random.randint(1, max_regions)
    for _ in range(num_regions):
        region_area = random.uniform(area_ratio[0], area_ratio[1]) * h * w
        rect_h = max(4, int(round(np.sqrt(region_area))))
        rect_w = max(4, int(round(region_area / max(rect_h, 1))))
        rect_h = min(rect_h, h)
        rect_w = min(rect_w, w)
        top = random.randint(0, max(0, h - rect_h))
        left = random.randint(0, max(0, w - rect_w))
        image[top:top + rect_h, left:left + rect_w] = mean_color
    return image


def apply_random_endoscopic_viewport(image: np.ndarray, mask: np.ndarray, fov_ratio=(0.84, 0.98), vignette_p: float = 0.5):
    h, w = image.shape[:2]
    cy = 0.5 * h + random.uniform(-0.03, 0.03) * h
    cx = 0.5 * w + random.uniform(-0.03, 0.03) * w
    ry = 0.5 * h * random.uniform(fov_ratio[0], fov_ratio[1])
    rx = 0.5 * w * random.uniform(fov_ratio[0], fov_ratio[1])

    yy, xx = np.ogrid[:h, :w]
    norm = ((yy - cy) / max(ry, 1e-6)) ** 2 + ((xx - cx) / max(rx, 1e-6)) ** 2
    valid = norm <= 1.0

    out = image.copy()
    out[~valid] = 0

    if random.random() < vignette_p:
        falloff = np.clip(1.15 - np.sqrt(np.clip(norm, 0.0, None)), 0.65, 1.0).astype(np.float32)
        out = np.clip(out.astype(np.float32) * falloff[..., None], 0, 255).astype(np.uint8)

    if mask is not None:
        mask = mask.copy()
        mask[~valid] = 0
    return out, mask


def apply_random_derm_frame(image: np.ndarray, mask: np.ndarray, border_ratio=(0.04, 0.18), pad_value=0):
    h, w = image.shape[:2]
    bw = int(round(random.uniform(border_ratio[0], border_ratio[1]) * min(h, w)))
    bw = max(2, min(bw, min(h, w) // 4))
    mode = random.choice(['all', 'tb', 'lr', 'corner'])
    out = image.copy()
    if mode in {'all', 'tb'}:
        out[:bw, :] = pad_value
        out[-bw:, :] = pad_value
    if mode in {'all', 'lr'}:
        out[:, :bw] = pad_value
        out[:, -bw:] = pad_value
    if mode == 'corner':
        out[:bw, :] = pad_value
        out[:, :bw] = pad_value
    return out, mask


def apply_random_derm_zoom_out(
    image: np.ndarray,
    mask: np.ndarray,
    scale=(0.68, 0.92),
    center_jitter: float = 0.10,
    black_pad_p: float = 0.70,
    pad_mode: str = 'edge',
    pad_value: int = 0,
):
    # Shrink the lesion canvas and paste it back onto a same-size image.
    # This simulates smaller lesions, more surrounding skin context, and
    # occasional dark borders without changing the final tensor size.
    h, w = image.shape[:2]
    s = float(random.uniform(scale[0], scale[1]))
    new_h = max(8, min(h, int(round(h * s))))
    new_w = max(8, min(w, int(round(w * s))))

    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    small_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if random.random() < black_pad_p:
        canvas = np.full((h, w, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        if str(pad_mode).lower() == 'reflect':
            canvas = np.full((h, w, image.shape[2]), pad_value, dtype=image.dtype)
        else:
            mean_color = small_img.reshape(-1, small_img.shape[-1]).mean(axis=0).astype(image.dtype)
            canvas = np.tile(mean_color.reshape(1, 1, -1), (h, w, 1))

    canvas_mask = np.zeros((h, w), dtype=mask.dtype)

    max_off_y = max(0, h - new_h)
    max_off_x = max(0, w - new_w)
    base_y = max_off_y // 2
    base_x = max_off_x // 2
    jitter_y = int(round(center_jitter * max_off_y))
    jitter_x = int(round(center_jitter * max_off_x))
    top = int(np.clip(base_y + (random.randint(-jitter_y, jitter_y) if jitter_y > 0 else 0), 0, max_off_y))
    left = int(np.clip(base_x + (random.randint(-jitter_x, jitter_x) if jitter_x > 0 else 0), 0, max_off_x))

    canvas[top:top + new_h, left:left + new_w] = small_img
    canvas_mask[top:top + new_h, left:left + new_w] = small_mask
    return canvas, canvas_mask


def apply_random_hair_occlusion(image: np.ndarray, num_hairs=(3, 12), thickness=(1, 3), darkness=(5, 60)):
    h, w = image.shape[:2]
    out = image.copy()
    n = random.randint(int(num_hairs[0]), int(num_hairs[1]))
    for _ in range(n):
        color = int(random.uniform(darkness[0], darkness[1]))
        start = (random.randint(0, w - 1), random.randint(0, h - 1))
        length = random.uniform(0.15, 0.55) * min(h, w)
        angle = random.uniform(0, np.pi)
        dx = int(np.cos(angle) * length)
        dy = int(np.sin(angle) * length)
        ctrl = (
            int(np.clip(start[0] + dx * 0.5 + random.uniform(-0.12, 0.12) * w, 0, w - 1)),
            int(np.clip(start[1] + dy * 0.5 + random.uniform(-0.12, 0.12) * h, 0, h - 1)),
        )
        end = (int(np.clip(start[0] + dx, 0, w - 1)), int(np.clip(start[1] + dy, 0, h - 1)))
        curve = np.array([start, ctrl, end], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [curve], False, (color, color, color), thickness=random.randint(int(thickness[0]), int(thickness[1])), lineType=cv2.LINE_AA)
    return out


def apply_endo_aug(image: np.ndarray, mask: np.ndarray, cfg=None):
    if random.random() < float(_aug_get(cfg, 'HFLIP_P', 0.5)):
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < float(_aug_get(cfg, 'ENDO_ROTATE_P', 0.20)):
        image, mask = rotate_pair(image, mask, float(_aug_get(cfg, 'ENDO_ROTATE_DEG', 10.0)))

    focus_p = float(_aug_get(cfg, 'ENDO_FOCUS_CROP_P', 0.35))
    if random.random() < focus_p:
        scale = _aug_get(cfg, 'ENDO_FOCUS_SCALE', [0.75, 1.0])
        image, mask = random_focus_crop_pair(
            image,
            mask,
            scale=(float(scale[0]), float(scale[1])),
            min_fg_keep=float(_aug_get(cfg, 'ENDO_MIN_FG_KEEP', 0.90)),
        )

    if random.random() < float(_aug_get(cfg, 'ENDO_COLOR_P', 0.35)):
        alpha = random.uniform(0.90, 1.10)
        beta = random.uniform(-10.0, 10.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.92, 1.12)
        hsv[..., 0] += random.uniform(-3.0, 3.0)
        hsv[..., 0] %= 180.0
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if random.random() < float(_aug_get(cfg, 'ENDO_GAMMA_P', 0.20)):
        image = adjust_gamma(image, random.uniform(0.92, 1.08))

    if random.random() < float(_aug_get(cfg, 'ENDO_BLUR_P', 0.10)):
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if random.random() < float(_aug_get(cfg, 'ENDO_NOISE_P', 0.05)):
        noise = np.random.normal(0.0, float(_aug_get(cfg, 'ENDO_NOISE_STD', 3.0)), size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < float(_aug_get(cfg, 'ENDO_OCCLUSION_P', 0.12)):
        image = apply_rect_dropout(
            image,
            max_regions=int(_aug_get(cfg, 'ENDO_OCCLUSION_REGIONS', 2)),
            area_ratio=tuple(float(x) for x in _aug_get(cfg, 'ENDO_OCCLUSION_AREA', [0.01, 0.05])),
        )

    if random.random() < float(_aug_get(cfg, 'ENDO_VIEWPORT_P', 0.20)):
        image, mask = apply_random_endoscopic_viewport(
            image,
            mask,
            fov_ratio=tuple(float(x) for x in _aug_get(cfg, 'ENDO_VIEWPORT_RATIO', [0.84, 0.98])),
            vignette_p=float(_aug_get(cfg, 'ENDO_VIEWPORT_VIGNETTE_P', 0.50)),
        )

    return image, mask


def apply_derm_aug(image: np.ndarray, mask: np.ndarray, cfg=None):
    if random.random() < float(_aug_get(cfg, 'HFLIP_P', 0.5)):
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < float(_aug_get(cfg, 'VFLIP_P', 0.25)):
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    if random.random() < float(_aug_get(cfg, 'DERM_ROTATE_P', 0.35)):
        image, mask = rotate_pair(image, mask, float(_aug_get(cfg, 'DERM_ROTATE_DEG', 25.0)))

    focus_p = float(_aug_get(cfg, 'DERM_FOCUS_CROP_P', 0.35))
    if random.random() < focus_p:
        scale = _aug_get(cfg, 'DERM_FOCUS_SCALE', [0.72, 1.0])
        image, mask = random_focus_crop_pair(
            image,
            mask,
            scale=(float(scale[0]), float(scale[1])),
            min_fg_keep=float(_aug_get(cfg, 'DERM_MIN_FG_KEEP', 0.92)),
        )

    if random.random() < float(_aug_get(cfg, 'DERM_COLOR_P', 0.35)):
        alpha = random.uniform(0.92, 1.10)
        beta = random.uniform(-8.0, 8.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.92, 1.15)
        hsv[..., 0] += random.uniform(-4.0, 4.0)
        hsv[..., 0] %= 180.0
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if random.random() < float(_aug_get(cfg, 'DERM_GAMMA_P', 0.20)):
        image = adjust_gamma(image, random.uniform(0.90, 1.12))

    if random.random() < float(_aug_get(cfg, 'DERM_BLUR_P', 0.08)):
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if random.random() < float(_aug_get(cfg, 'DERM_NOISE_P', 0.04)):
        noise = np.random.normal(0.0, float(_aug_get(cfg, 'DERM_NOISE_STD', 2.0)), size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < float(_aug_get(cfg, 'DERM_HAIR_P', 0.20)):
        image = apply_random_hair_occlusion(
            image,
            num_hairs=tuple(int(x) for x in _aug_get(cfg, 'DERM_HAIR_COUNT', [3, 10])),
            thickness=tuple(int(x) for x in _aug_get(cfg, 'DERM_HAIR_THICKNESS', [1, 3])),
            darkness=tuple(float(x) for x in _aug_get(cfg, 'DERM_HAIR_DARKNESS', [5, 55])),
        )

    if random.random() < float(_aug_get(cfg, 'DERM_FRAME_P', 0.12)):
        image, mask = apply_random_derm_frame(
            image,
            mask,
            border_ratio=tuple(float(x) for x in _aug_get(cfg, 'DERM_FRAME_RATIO', [0.04, 0.18])),
            pad_value=int(_aug_get(cfg, 'DERM_FRAME_VALUE', 0)),
        )

    if random.random() < float(_aug_get(cfg, 'DERM_ZOOM_OUT_P', 0.0)):
        image, mask = apply_random_derm_zoom_out(
            image,
            mask,
            scale=tuple(float(x) for x in _aug_get(cfg, 'DERM_ZOOM_OUT_SCALE', [0.68, 0.92])),
            center_jitter=float(_aug_get(cfg, 'DERM_ZOOM_OUT_CENTER_JITTER', 0.10)),
            black_pad_p=float(_aug_get(cfg, 'DERM_ZOOM_OUT_BLACK_PAD_P', 0.70)),
            pad_mode=str(_aug_get(cfg, 'DERM_ZOOM_OUT_PAD_MODE', 'edge')),
            pad_value=int(_aug_get(cfg, 'DERM_FRAME_VALUE', 0)),
        )

    return image, mask


def apply_default_aug(image: np.ndarray, mask: np.ndarray, cfg=None):
    if random.random() < float(_aug_get(cfg, 'HFLIP_P', 0.5)):
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < float(_aug_get(cfg, 'DEFAULT_ROTATE_P', 0.25)):
        image, mask = rotate_pair(image, mask, float(_aug_get(cfg, 'DEFAULT_ROTATE_DEG', 15.0)))
    return image, mask


class ValGenerator(object):
    def __init__(self, output_size, cfg=None, task_name: Optional[str] = None):
        self.output_size = output_size
        self.cfg = cfg
        self.task_name = task_name

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        task_name = sample.get('dataset_name', self.task_name)
        modality = infer_modality(task_name, self.cfg)

        if mask.ndim == 3:
            mask = mask[..., 0]

        crop_meta = {
            'crop_xywh': np.array([0, 0, int(image.shape[1]), int(image.shape[0])], dtype=np.int32),
            'crop_applied': np.array([0], dtype=np.int32),
        }
        if modality == 'endo' and bool(_pp_get(self.cfg, 'ENDO_CROP_VALID_FOV', False)):
            image, mask, crop_meta = crop_valid_fov(
                image,
                mask,
                gray_thr=int(_pp_get(self.cfg, 'ENDO_CROP_GRAY_THR', 8)),
                min_keep_ratio=float(_pp_get(self.cfg, 'ENDO_CROP_MIN_KEEP_RATIO', 0.55)),
                return_meta=True,
            )
        if modality == 'derm' and bool(_pp_get(self.cfg, 'DERM_CROP_DARK_FRAME', False)):
            image, mask, crop_meta = crop_dark_frame(
                image,
                mask,
                gray_thr=int(_pp_get(self.cfg, 'DERM_CROP_GRAY_THR', 12)),
                min_keep_ratio=float(_pp_get(self.cfg, 'DERM_CROP_MIN_KEEP_RATIO', 0.60)),
                return_meta=True,
            )

        image = preprocess_image_by_modality(image, modality, is_train=False, cfg=self.cfg)
        image, mask, resize_meta = resize_image_and_mask(
            image,
            mask,
            self.output_size,
            keep_aspect_ratio=bool(_pp_get(self.cfg, 'KEEP_ASPECT_RATIO', modality in {'endo', 'derm'})),
            pad_mode=str(_pp_get(self.cfg, 'PAD_MODE', 'edge')),
            return_meta=True,
        )

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        image = F.to_tensor(image)
        image = get_normalize_transform(self.cfg)(image)
        mask = to_long_tensor(mask)

        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['meta_orig_hw'] = torch.as_tensor(resize_meta['orig_hw'], dtype=torch.int32)
        sample['meta_resized_hw'] = torch.as_tensor(resize_meta['resized_hw'], dtype=torch.int32)
        sample['meta_pad_tblr'] = torch.as_tensor(resize_meta['pad_tblr'], dtype=torch.int32)
        sample['meta_keep_ar'] = torch.as_tensor(resize_meta['keep_aspect_ratio'], dtype=torch.int32)
        sample['meta_crop_xywh'] = torch.as_tensor(crop_meta['crop_xywh'], dtype=torch.int32)
        sample['meta_crop_applied'] = torch.as_tensor(crop_meta['crop_applied'], dtype=torch.int32)
        return sample


class RandomGenerator(object):
    def __init__(self, output_size, cfg=None, task_name: Optional[str] = None):
        self.output_size = output_size
        self.cfg = cfg
        self.task_name = task_name

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        task_name = sample.get('dataset_name', self.task_name)
        modality = infer_modality(task_name, self.cfg)

        if mask.ndim == 3:
            mask = mask[..., 0]

        crop_meta = {
            'crop_xywh': np.array([0, 0, int(image.shape[1]), int(image.shape[0])], dtype=np.int32),
            'crop_applied': np.array([0], dtype=np.int32),
        }
        if modality == 'endo' and bool(_pp_get(self.cfg, 'ENDO_CROP_VALID_FOV', False)):
            image, mask, crop_meta = crop_valid_fov(
                image,
                mask,
                gray_thr=int(_pp_get(self.cfg, 'ENDO_CROP_GRAY_THR', 8)),
                min_keep_ratio=float(_pp_get(self.cfg, 'ENDO_CROP_MIN_KEEP_RATIO', 0.55)),
                return_meta=True,
            )
        if modality == 'derm' and bool(_pp_get(self.cfg, 'DERM_CROP_DARK_FRAME', False)):
            image, mask, crop_meta = crop_dark_frame(
                image,
                mask,
                gray_thr=int(_pp_get(self.cfg, 'DERM_CROP_GRAY_THR', 12)),
                min_keep_ratio=float(_pp_get(self.cfg, 'DERM_CROP_MIN_KEEP_RATIO', 0.60)),
                return_meta=True,
            )

        image = preprocess_image_by_modality(image, modality, is_train=True, cfg=self.cfg)
        image, mask, resize_meta = resize_image_and_mask(
            image,
            mask,
            self.output_size,
            keep_aspect_ratio=bool(_pp_get(self.cfg, 'KEEP_ASPECT_RATIO', modality in {'endo', 'derm'})),
            pad_mode=str(_pp_get(self.cfg, 'PAD_MODE', 'edge')),
            return_meta=True,
        )

        if modality == 'endo':
            image, mask = apply_endo_aug(image, mask, cfg=self.cfg)
        elif modality == 'derm':
            image, mask = apply_derm_aug(image, mask, cfg=self.cfg)
        else:
            image, mask = apply_default_aug(image, mask, cfg=self.cfg)

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        image = F.to_tensor(image)
        image = get_normalize_transform(self.cfg)(image)
        mask = to_long_tensor(mask)

        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['meta_orig_hw'] = torch.as_tensor(resize_meta['orig_hw'], dtype=torch.int32)
        sample['meta_resized_hw'] = torch.as_tensor(resize_meta['resized_hw'], dtype=torch.int32)
        sample['meta_pad_tblr'] = torch.as_tensor(resize_meta['pad_tblr'], dtype=torch.int32)
        sample['meta_keep_ar'] = torch.as_tensor(resize_meta['keep_aspect_ratio'], dtype=torch.int32)
        sample['meta_crop_xywh'] = torch.as_tensor(crop_meta['crop_xywh'], dtype=torch.int32)
        sample['meta_crop_applied'] = torch.as_tensor(crop_meta['crop_applied'], dtype=torch.int32)
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
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'label')
        self.one_hot_mask = one_hot_mask
        self.task_name = task_name
        self.cfg = cfg
        self.data_pairs = self._build_pairs(row_text)
        self.data_pairs = sorted(self.data_pairs, key=lambda x: x[0])
        self.joint_transform = joint_transform if joint_transform is not None else (lambda x: x)

    def _build_pairs(self, row_text):
        pairs = []
        for row in row_text:
            image_name = row.get('Image') or row.get('image')
            text = row.get('Description') or row.get('description') or ''
            mask_name = row.get('Ground Truth') or row.get('Mask') or row.get('mask') or image_name
            if image_name is None:
                continue
            pairs.append((image_name, mask_name, text))
        return pairs

    def _resolve_mask_path(self, mask_filename: str, image_filename: str):
        candidates = []
        if mask_filename:
            candidates.append(mask_filename)
        stem, ext = os.path.splitext(image_filename)
        candidates.extend([
            image_filename,
            f'{stem}_Segmentation{ext or ".png"}',
            f'{stem}_mask{ext or ".png"}',
            f'{stem}.png',
        ])

        seen = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            path = os.path.join(self.output_path, cand)
            if os.path.exists(path):
                return path, cand

        prefix = os.path.splitext(mask_filename or image_filename)[0]
        for fname in sorted(os.listdir(self.output_path)):
            if os.path.splitext(fname)[0] == prefix or fname.startswith(prefix):
                return os.path.join(self.output_path, fname), fname

        raise FileNotFoundError(os.path.join(self.output_path, mask_filename or image_filename))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_filename, mask_filename, text = self.data_pairs[idx]
        image_path = os.path.join(self.input_path, image_filename)
        mask_path, resolved_mask_name = self._resolve_mask_path(mask_filename, image_filename)

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(image_path)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)

        threshold = int(_cfg_get(self.cfg, 'DATASET.MASK_THRESHOLD', 127))
        mask = binarize_mask(mask, threshold=threshold)
        image, mask = correct_dims(image, mask)

        inputs = {
            'image': image,
            'ground_truth_mask': mask,
            'image_name': image_filename,
            'mask_name': resolved_mask_name,
            'text_prompt': text,
            'dataset_name': self.task_name,
        }
        inputs = self.joint_transform(inputs)

        if self.one_hot_mask:
            m = inputs['ground_truth_mask']
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            inputs['ground_truth_mask'] = torch.zeros(
                (self.one_hot_mask, m.shape[-2], m.shape[-1]),
                dtype=torch.float32,
            ).scatter_(0, m.long(), 1)

        return inputs
