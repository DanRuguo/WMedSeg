import argparse
import logging
import os
import random
from statistics import mean

import monai
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of data to use.")
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


def _single_loss(logits, masks, ce_loss, dice_loss, cfg):
    loss_ce = ce_loss(logits, masks.float())
    loss_dice = dice_loss(logits, masks)
    return cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce


def calc_loss(logits, masks, ce_loss, dice_loss, cfg, aux_logits=None):
    loss = _single_loss(logits, masks, ce_loss, dice_loss, cfg)
    aux_weight = float(getattr(cfg.TRAIN, "AUX_WEIGHT", 0.0))
    if aux_weight > 0 and aux_logits:
        aux_weights = list(getattr(cfg.TRAIN, "AUX_WEIGHTS", [0.5, 0.3, 0.2]))
        aux_total = 0.0
        for i, aux_logit in enumerate(aux_logits):
            w = aux_weights[i] if i < len(aux_weights) else aux_weights[-1]
            resized_mask = F.interpolate(masks.unsqueeze(1).float(), size=aux_logit.shape[-2:], mode="nearest").squeeze(1)
            aux_total = aux_total + float(w) * _single_loss(aux_logit, resized_mask, ce_loss, dice_loss, cfg)
        loss = loss + aux_weight * aux_total
    return loss


def evaluate_validation_loss(model, val_dataloader, device, ce_loss, dice_loss, cfg):
    model.eval()
    val_losses = []
    dice_scores = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            text = batch["text_prompt"]
            logits = model(images, text=text, num_samples=1)[0]
            loss = calc_loss(logits, masks, ce_loss, dice_loss, cfg)
            val_losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())

    model.train()
    return mean(val_losses), mean(dice_scores)


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


def build_optimizer(model, cfg):
    base_lr = float(cfg.TRAIN.LEARNING_RATE)
    gate_lr = float(getattr(cfg.TRAIN, "GATE_LR", base_lr * 0.5))
    wd = float(getattr(cfg.TRAIN, "WEIGHT_DECAY", 1e-4))

    gate_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "hybrid_alpha" in name:
            gate_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": wd})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


cfg = get_arguments()
cfg.DATASET.NAME = cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}"), exist_ok=True)
logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))
logger.info("************")
logger.info("** Config **")
logger.info("************")
logger.info(cfg)

if cfg.seed >= 0:
    logger.info(f"Setting fixed seed: {cfg.seed}")
    set_random_seed(cfg.seed)


def worker_init_fn(worker_id):
    seed = cfg.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


ce_loss = BCEWithLogitsLoss()
dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")

train_tf = transforms.Compose([RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE], task_name=cfg.DATASET.NAME, cfg=cfg)])
val_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE], task_name=cfg.DATASET.NAME, cfg=cfg)

train_text_file = f"Train_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Train_text.xlsx"
val_text_file = f"Val_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Val_text.xlsx"
train_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + train_text_file)
val_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + val_text_file)

train_dataset = DatasetSegmentation(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text, train_tf, image_size=cfg.DATASET.SIZE)
val_dataset = DatasetSegmentation(cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, val_text, val_tf, image_size=cfg.DATASET.SIZE)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    num_workers=int(getattr(cfg.TRAIN, "NUM_WORKERS", 8)),
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    shuffle=False,
    worker_init_fn=worker_init_fn,
    num_workers=int(getattr(cfg.TRAIN, "NUM_WORKERS", 8)),
    pin_memory=True,
)

model = build_model(cfg)
enabled = {name for name, param in model.named_parameters() if param.requires_grad}
logger.info(f"Parameters to be updated: {enabled}")
logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = build_optimizer(model, cfg)
num_epochs = int(cfg.TRAIN.NUM_EPOCHS)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=float(getattr(cfg.TRAIN, "MIN_LR", 1e-5)))

backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
resume_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", f"{results_name}_latest.pth")

start_epoch = 0
best_dice = 0.0
if cfg.resume and os.path.exists(resume_path):
    checkpoint = torch.load(resume_path, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint.get("scheduler", {}))
    start_epoch = checkpoint["epoch"] + 1
    best_dice = checkpoint.get("best_dice", 0.0)
    logger.info(f"Loaded checkpoint from epoch {start_epoch}, best dice: {best_dice:.4f}")

model.train().to(cfg.MODEL.DEVICE)
grad_clip = float(getattr(cfg.TRAIN, "GRAD_CLIP", 0.0))

for epoch in range(start_epoch, num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        out = model(image=batch["image"].to(cfg.MODEL.DEVICE), text=batch["text_prompt"])
        if isinstance(out, tuple) and len(out) == 3:
            seg_logits, clip_loss, aux_logits = out
        else:
            seg_logits, clip_loss = out
            aux_logits = None

        loss = calc_loss(seg_logits, batch["ground_truth_mask"].to(cfg.MODEL.DEVICE), ce_loss, dice_loss, cfg, aux_logits=aux_logits)
        loss = loss + float(cfg.TRAIN.CLIP_WEIGHT) * clip_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_losses.append(loss.item())

    scheduler.step()
    mean_epoch_loss = mean(epoch_losses)
    mean_val_loss, mean_val_dice = evaluate_validation_loss(model, val_dataloader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cfg)
    logger.info(f"EPOCH: {epoch + 1} | Training Loss: {mean_epoch_loss:.4f} | Validation Loss: {mean_val_loss:.4f} | Validation Dice: {mean_val_dice:.4f}")

    if mean_val_dice > best_dice:
        logger.info(f"New best Dice: {best_dice:.4f} -> {mean_val_dice:.4f}")
        best_dice = mean_val_dice
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
            },
            os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", f"{results_name}_best_dice.pth"),
        )

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
        },
        os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", f"{results_name}_latest.pth"),
    )
