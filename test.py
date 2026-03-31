import argparse
import logging
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, normalize, read_text


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--prompt_design", type=str, default="original", help="Text prompt design.")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use.")
    parser.add_argument("--source_dataset", type=str, help="source dataset name for loading trained model.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def logger_config(log_path):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def build_model(cfg):
    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        return build_medclipseg_unimedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "biomedclip":
        return build_medclipseg_biomedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "clip":
        return build_medclipseg_clip(cfg)
    if cfg.MODEL.CLIP_MODEL == "pubmedclip":
        return build_medclipseg_pubmedclip(cfg)
    raise NotImplementedError(cfg.MODEL.CLIP_MODEL)


def _tta_predict(model, images, text, num_samples, hflip=False):
    pred = model(image=images, text=text, num_samples=num_samples)
    pred_mean = torch.sigmoid(pred).mean(dim=0)
    if not hflip:
        return pred_mean

    pred_flip = model(image=torch.flip(images, dims=[3]), text=text, num_samples=num_samples)
    pred_flip = torch.flip(torch.sigmoid(pred_flip).mean(dim=0), dims=[2])
    return 0.5 * (pred_mean + pred_flip)


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
    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    checkpoint_path = os.path.join(
        cfg.output_dir,
        cfg.source_dataset if cfg.data_percentage == 100 else cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth",
    )

    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(cfg.MODEL.DEVICE)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE], task_name=cfg.DATASET.NAME, cfg=cfg)
    test_text_file = f"Test_text_{cfg.prompt_design}.xlsx"
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + test_text_file)
    test_dataset = DatasetSegmentation(cfg.DATASET.TEST_PATH, cfg.DATASET.NAME, test_text, test_tf, image_size=cfg.DATASET.SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=int(getattr(cfg.TEST, "BATCH_SIZE", 32)), shuffle=False)

    use_hflip = bool(getattr(getattr(cfg.TEST, "TTA", None), "HFLIP", False))
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            pred_mean = _tta_predict(
                model=model,
                images=batch["image"].to(cfg.MODEL.DEVICE),
                text=batch["text_prompt"],
                num_samples=cfg.TEST.NUM_SAMPLES,
                hflip=use_hflip,
            )
            seg_unc = -(pred_mean * torch.log(pred_mean + 1e-8) + (1 - pred_mean) * torch.log(1 - pred_mean + 1e-8))
            mask_preds = pred_mean > 0.5

            dataset_names = batch["dataset_name"]
            mask_names = batch["mask_name"]
            save_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", results_name + f"_Prompt-{cfg.prompt_design}")
            save_unc_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "unc_results", f"seed{cfg.seed}", results_name + f"_Prompt-{cfg.prompt_design}")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(save_unc_dir, exist_ok=True)

            for i in range(len(dataset_names)):
                pred_mask = mask_preds[i].cpu().numpy().astype(np.uint8)
                binary_pred = np.uint8(pred_mask > 0)
                mask_name = mask_names[i]
                cv2.imwrite(os.path.join(save_dir, mask_name), binary_pred * 255)

                u_map = seg_unc[i].cpu().numpy()
                u_map = normalize(u_map)
                colormap = plt.get_cmap("nipy_spectral")
                u_map_color = (colormap(u_map)[:, :, :3] * 255).astype(np.uint8)
                u_map_colored = cv2.cvtColor(u_map_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_unc_dir, mask_name), u_map_colored)


if __name__ == "__main__":
    main()
