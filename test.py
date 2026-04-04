import os
import cv2
import torch
import argparse
import logging
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text, normalize
import matplotlib.pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, type=str, help="Path to config file",
    )
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument('--prompt_design', type=str, default="original", help="Text prompt design.")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use.")
    parser.add_argument("--source_dataset", type=str, help="source dataset name for loading trained model.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def logger_config(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def _cfg_get(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _round_to_multiple(x, base=32):
    return max(base, int(round(float(x) / base) * base))


def _forward_predictive_mean(model, images, texts, num_samples):
    seg_samples = model(image=images, text=texts, num_samples=num_samples)
    seg_samples = torch.sigmoid(seg_samples)
    return seg_samples.mean(dim=0)


def predict_with_tta(model, images, texts, cfg):
    num_samples = int(_cfg_get(cfg, "TEST.NUM_SAMPLES", 8))
    use_hflip = bool(_cfg_get(cfg, "TEST.TTA.HFLIP", True))
    use_vflip = bool(_cfg_get(cfg, "TEST.TTA.VFLIP", False))
    rot90_list = list(_cfg_get(cfg, "TEST.TTA.ROT90", []))
    use_scales = list(_cfg_get(cfg, "TEST.TTA.SCALES", [0.875, 1.125]))
    allow_scaled_tta = bool(getattr(model, "supports_scaled_inference", False))

    if use_scales and not allow_scaled_tta:
        if not getattr(predict_with_tta, "_warned_scale_skip", False):
            print(
                f"Skip TEST.TTA.SCALES={use_scales} because this ViT image encoder uses fixed positional embeddings "
                f"and only supports the trained input size ({images.shape[-1]}x{images.shape[-2]}) without encoder changes."
            )
            predict_with_tta._warned_scale_skip = True
        use_scales = []

    preds = []
    base_pred = _forward_predictive_mean(model, images, texts, num_samples)
    preds.append(base_pred)

    if use_hflip:
        flip_imgs = torch.flip(images, dims=[-1])
        flip_pred = _forward_predictive_mean(model, flip_imgs, texts, num_samples)
        flip_pred = torch.flip(flip_pred, dims=[-1])
        preds.append(flip_pred)

    if use_vflip:
        flip_imgs = torch.flip(images, dims=[-2])
        flip_pred = _forward_predictive_mean(model, flip_imgs, texts, num_samples)
        flip_pred = torch.flip(flip_pred, dims=[-2])
        preds.append(flip_pred)

    for k in rot90_list:
        k = int(k) % 4
        if k == 0:
            continue
        rot_imgs = torch.rot90(images, k=k, dims=[-2, -1])
        rot_pred = _forward_predictive_mean(model, rot_imgs, texts, num_samples)
        rot_pred = torch.rot90(rot_pred, k=(4 - k) % 4, dims=[-2, -1])
        preds.append(rot_pred)

    h, w = images.shape[-2:]
    for scale in use_scales:
        nh = _round_to_multiple(h * float(scale), 32)
        nw = _round_to_multiple(w * float(scale), 32)
        if (nh, nw) == (h, w):
            continue
        scaled = F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False)
        scaled_pred = _forward_predictive_mean(model, scaled, texts, num_samples)
        if scaled_pred.shape[-2:] != (h, w):
            scaled_pred = F.interpolate(
                scaled_pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
        preds.append(scaled_pred)

    return torch.stack(preds, dim=0).mean(dim=0)


def _to_hw_tensor(x, default=None):
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _paste_back_crop(arr: np.ndarray, orig_hw, crop_xywh=None, crop_applied=False, is_binary=False):
    arr = np.asarray(arr)
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
    canvas = np.zeros((orig_h, orig_w), dtype=arr.dtype)
    if crop_xywh is None or not crop_applied:
        return arr
    x, y, w, h = [int(v) for v in crop_xywh]
    x = max(0, min(orig_w, x))
    y = max(0, min(orig_h, y))
    w = max(1, min(orig_w - x, w))
    h = max(1, min(orig_h - y, h))
    interp = cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR
    arr = cv2.resize(arr, (w, h), interpolation=interp)
    canvas[y:y+h, x:x+w] = arr
    if is_binary:
        canvas = (canvas > 0).astype(np.uint8)
    return canvas


def _extract_components(mask: np.ndarray):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        cx, cy = centroids[idx]
        comps.append({
            'idx': idx,
            'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
            'area': int(area), 'cx': float(cx), 'cy': float(cy),
        })
    return labels, comps


def _fill_holes_binary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    h, w = mask.shape[:2]
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.clip(mask + holes, 0, 1).astype(np.uint8)


def _smooth_probability_map(prob: np.ndarray, cfg) -> np.ndarray:
    k = int(_cfg_get(cfg, 'TEST.POSTPROCESS.PROB_GAUSSIAN_K', 0))
    if k <= 1:
        return prob
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(prob.astype(np.float32), (k, k), 0)


def _postprocess_binary_mask(mask: np.ndarray, cfg):
    if not bool(_cfg_get(cfg, 'TEST.POSTPROCESS.ENABLE', False)):
        return mask.astype(np.uint8)

    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask

    k = int(_cfg_get(cfg, 'TEST.POSTPROCESS.MORPH_CLOSE_K', 0))
    if k > 1:
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if bool(_cfg_get(cfg, 'TEST.POSTPROCESS.FILL_HOLES', False)):
        mask = _fill_holes_binary(mask)

    h, w = mask.shape[:2]
    image_area = float(h * w)
    labels, comps = _extract_components(mask)
    if not comps:
        return mask

    min_area_ratio = float(_cfg_get(cfg, 'TEST.POSTPROCESS.MIN_AREA_RATIO', 0.0))
    border_margin = int(_cfg_get(cfg, 'TEST.POSTPROCESS.BORDER_MARGIN', 0))
    remove_border = bool(_cfg_get(cfg, 'TEST.POSTPROCESS.REMOVE_BORDER_COMPONENTS', False))
    score_by_center = bool(_cfg_get(cfg, 'TEST.POSTPROCESS.SCORE_BY_CENTER', False))
    center_weight = float(_cfg_get(cfg, 'TEST.POSTPROCESS.CENTER_WEIGHT', 0.35))
    keep_largest = bool(_cfg_get(cfg, 'TEST.POSTPROCESS.KEEP_LARGEST', True))

    kept = []
    cx0, cy0 = 0.5 * w, 0.5 * h
    max_dist = max((cx0 ** 2 + cy0 ** 2) ** 0.5, 1e-6)
    for comp in comps:
        if comp['area'] / image_area < min_area_ratio:
            continue
        touches_border = (
            comp['x'] <= border_margin or comp['y'] <= border_margin or
            (comp['x'] + comp['w']) >= (w - border_margin) or
            (comp['y'] + comp['h']) >= (h - border_margin)
        )
        if remove_border and touches_border:
            continue
        score = float(comp['area'])
        if score_by_center:
            dist = ((comp['cx'] - cx0) ** 2 + (comp['cy'] - cy0) ** 2) ** 0.5 / max_dist
            score = score * (1.0 + center_weight * (1.0 - dist))
        comp['score'] = score
        kept.append(comp)

    if not kept:
        return mask

    out = np.zeros_like(mask, dtype=np.uint8)
    if keep_largest:
        best = max(kept, key=lambda c: c['score'])
        out[labels == best['idx']] = 1
    else:
        for comp in kept:
            out[labels == comp['idx']] = 1
    return out


def _restore_to_original_canvas(arr: np.ndarray, orig_hw, resized_hw=None, pad_tblr=None, keep_ar=False,
                                crop_xywh=None, crop_applied=False, is_binary=False):
    arr = np.asarray(arr)
    orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])

    target_h, target_w = orig_h, orig_w
    if crop_applied and crop_xywh is not None:
        target_w = max(1, int(crop_xywh[2]))
        target_h = max(1, int(crop_xywh[3]))

    if keep_ar and resized_hw is not None and pad_tblr is not None:
        new_h, new_w = int(resized_hw[0]), int(resized_hw[1])
        pad_top, pad_bottom, pad_left, pad_right = [int(x) for x in pad_tblr]
        h, w = arr.shape[:2]
        y0 = max(0, min(h, pad_top))
        y1 = max(y0, min(h, h - pad_bottom))
        x0 = max(0, min(w, pad_left))
        x1 = max(x0, min(w, w - pad_right))
        arr = arr[y0:y1, x0:x1]
        if arr.size == 0:
            arr = np.zeros((max(1, new_h), max(1, new_w)), dtype=np.float32 if not is_binary else np.uint8)

    interp = cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR
    arr = cv2.resize(arr, (target_w, target_h), interpolation=interp)
    if crop_applied and crop_xywh is not None:
        arr = _paste_back_crop(arr, orig_hw=orig_hw, crop_xywh=crop_xywh, crop_applied=crop_applied, is_binary=is_binary)
    elif is_binary:
        arr = (arr > 0).astype(np.uint8)
    return arr


def main():
    cfg = get_arguments()
    if cfg.seed >= 0:
        print(f"Setting fixed seed: {cfg.seed}")
        set_random_seed(cfg.seed)

    cfg.DATASET.NAME = cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
    results_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}")
    os.makedirs(results_root, exist_ok=True)

    logger = logger_config(os.path.join(results_root, "log.txt"))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"

    checkpoint_type = "latest" if bool(_cfg_get(cfg, "TEST.USE_LATEST", False)) else "best_dice"
    checkpoint_dataset = cfg.source_dataset if cfg.data_percentage == 100 else cfg.DATASET.NAME
    checkpoint_path = os.path.join(
        cfg.output_dir,
        checkpoint_dataset,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth"
    )
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        model = build_medclipseg_unimedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "biomedclip":
        model = build_medclipseg_biomedclip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "clip":
        model = build_medclipseg_clip(cfg)
    elif cfg.MODEL.CLIP_MODEL == "pubmedclip":
        model = build_medclipseg_pubmedclip(cfg)
    else:
        raise ValueError(f"Unknown CLIP model: {cfg.MODEL.CLIP_MODEL}")

    checkpoint = torch.load(checkpoint_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(cfg.MODEL.DEVICE)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE], cfg=cfg)
    test_text_file = f"Test_text_{cfg.prompt_design}.xlsx"
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + test_text_file)
    test_dataset = DatasetSegmentation(
        cfg.DATASET.TEST_PATH,
        cfg.DATASET.NAME,
        test_text,
        test_tf,
        image_size=cfg.DATASET.SIZE,
        cfg=cfg,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(_cfg_get(cfg, "TEST.BATCH_SIZE", 32)),
        shuffle=False,
    )

    threshold = float(_cfg_get(cfg, "TEST.THRESHOLD", 0.5))

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images = batch["image"].to(cfg.MODEL.DEVICE)
            texts = batch["text_prompt"]

            seg_probs = predict_with_tta(model, images, texts, cfg)
            seg_unc = -(
                seg_probs * torch.log(seg_probs + 1e-8) +
                (1 - seg_probs) * torch.log(1 - seg_probs + 1e-8)
            )

            dataset_names = batch["dataset_name"]
            mask_names = batch["mask_name"]
            orig_hw = _to_hw_tensor(batch.get("meta_orig_hw"), None)
            resized_hw = _to_hw_tensor(batch.get("meta_resized_hw"), None)
            pad_tblr = _to_hw_tensor(batch.get("meta_pad_tblr"), None)
            keep_ar = _to_hw_tensor(batch.get("meta_keep_ar"), None)
            crop_xywh = _to_hw_tensor(batch.get("meta_crop_xywh"), None)
            crop_applied = _to_hw_tensor(batch.get("meta_crop_applied"), None)

            for i in range(len(dataset_names)):
                prob_map = seg_probs[i].cpu().numpy().astype(np.float32)
                u_map = seg_unc[i].cpu().numpy()
                mask_name = mask_names[i]

                sample_orig_hw = orig_hw[i].tolist() if orig_hw is not None else [prob_map.shape[0], prob_map.shape[1]]
                sample_resized_hw = resized_hw[i].tolist() if resized_hw is not None else [prob_map.shape[0], prob_map.shape[1]]
                sample_pad_tblr = pad_tblr[i].tolist() if pad_tblr is not None else [0, 0, 0, 0]
                sample_keep_ar = bool(int(keep_ar[i].item())) if keep_ar is not None else False
                sample_crop_xywh = crop_xywh[i].tolist() if crop_xywh is not None else [0, 0, sample_orig_hw[1], sample_orig_hw[0]]
                sample_crop_applied = bool(int(crop_applied[i].item())) if crop_applied is not None else False

                prob_map = _restore_to_original_canvas(
                    prob_map,
                    orig_hw=sample_orig_hw,
                    resized_hw=sample_resized_hw,
                    pad_tblr=sample_pad_tblr,
                    keep_ar=sample_keep_ar,
                    crop_xywh=sample_crop_xywh,
                    crop_applied=sample_crop_applied,
                    is_binary=False,
                )
                prob_map = _smooth_probability_map(prob_map, cfg)
                pred_mask = (prob_map > threshold).astype(np.uint8)
                pred_mask = _postprocess_binary_mask(pred_mask, cfg)

                u_map = _restore_to_original_canvas(
                    u_map,
                    orig_hw=sample_orig_hw,
                    resized_hw=sample_resized_hw,
                    pad_tblr=sample_pad_tblr,
                    keep_ar=sample_keep_ar,
                    crop_xywh=sample_crop_xywh,
                    crop_applied=sample_crop_applied,
                    is_binary=False,
                )

                save_dir = os.path.join(
                    cfg.output_dir,
                    cfg.DATASET.NAME,
                    "seg_results",
                    f"seed{cfg.seed}",
                    results_name + f"_Prompt-{cfg.prompt_design}",
                )
                save_unc_dir = os.path.join(
                    cfg.output_dir,
                    cfg.DATASET.NAME,
                    "unc_results",
                    f"seed{cfg.seed}",
                    results_name + f"_Prompt-{cfg.prompt_design}",
                )
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(save_unc_dir, exist_ok=True)

                cv2.imwrite(os.path.join(save_dir, mask_name), pred_mask * 255)

                u_map = normalize(u_map)
                colormap = plt.get_cmap("nipy_spectral")
                u_map_color = (colormap(u_map)[:, :, :3] * 255).astype(np.uint8)
                u_map_colored = cv2.cvtColor(u_map_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_unc_dir, mask_name), u_map_colored)


if __name__ == "__main__":
    main()
