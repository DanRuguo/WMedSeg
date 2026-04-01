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

            seg_logits = predict_with_tta(model, images, texts, cfg)
            seg_unc = -(
                seg_logits * torch.log(seg_logits + 1e-8) +
                (1 - seg_logits) * torch.log(1 - seg_logits + 1e-8)
            )
            mask_preds = (seg_logits > threshold)

            dataset_names = batch["dataset_name"]
            mask_names = batch["mask_name"]

            for i in range(len(dataset_names)):
                pred_mask = mask_preds[i].cpu().numpy().astype(np.uint8)
                mask_name = mask_names[i]

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

                u_map = seg_unc[i].cpu().numpy()
                u_map = normalize(u_map)
                colormap = plt.get_cmap("nipy_spectral")
                u_map_color = (colormap(u_map)[:, :, :3] * 255).astype(np.uint8)
                u_map_colored = cv2.cvtColor(u_map_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_unc_dir, mask_name), u_map_colored)


if __name__ == "__main__":
    main()
