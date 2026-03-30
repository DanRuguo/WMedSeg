import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from open_clip_lib import create_model_and_transforms, HFTokenizer, get_mean_std

from .layers import PVL_Adapter
from .local_cnn_encoder import LocalCNNEncoder
from .global_pyramid_builder import GlobalPyramidBuilder
from .w_pvl_decoder import WPVLDecoder

try:
    from .w_pvl_sr_decoder import WPVLStyleRobustDecoder
except ImportError:
    WPVLStyleRobustDecoder = None


def download_checkpoint(filename: str):
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    local_path = os.path.join(ckpt_dir, filename)

    if os.path.isfile(local_path):
        print(f"Found checkpoint: {local_path}")
        return local_path

    print(f"Checkpoint not found. Downloading {filename} from Hugging Face...")

    hf_hub_download(
        repo_id="TahaKoleilat/MedCLIPSeg",
        repo_type="model",
        filename=f"checkpoints/{filename}",
        local_dir=".",
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded checkpoint to {local_path}")
    return local_path


def load_unimedclip_to_device(cfg):
    if cfg.MODEL.BACKBONE == "ViT-B/16":
        model_name = "ViT-B-16-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_b16.pt")

    elif cfg.MODEL.BACKBONE == "ViT-L/14":
        model_name = "ViT-L-14-336-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_l14_base_text_encoder.pt")

    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

    text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    mean, std = get_mean_std()
    device = cfg.MODEL.DEVICE

    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained_weights,
        precision="amp",
        device=device,
        force_quick_gelu=True,
        mean=mean,
        std=std,
        inmem=True,
        text_encoder_name=text_encoder_name,
    )

    return model.to(device).eval()


def _num_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def _make_binary_head(in_channels: int) -> nn.Sequential:
    hidden = max(in_channels // 2, 32)
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GroupNorm(_num_groups(hidden), hidden),
        nn.GELU(),
        nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0),
    )


def _pool_text_feature(text_feat: torch.Tensor) -> torch.Tensor:
    if isinstance(text_feat, (list, tuple)):
        text_feat = text_feat[-1]
    if text_feat.dim() == 3:
        return text_feat.mean(dim=1)
    return text_feat


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states=False):
        super(CustomCLIP, self).__init__()

        self.cfg = cfg
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder
        self.logit_scale = clip_model.logit_scale
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512

        elif cfg.MODEL.BACKBONE == "ViT-L/14":
            self.embed_dim = 1024
            self.patch_size = 14
            self.text_proj_dim = 768
            raise NotImplementedError("ViT-L/14 not implemented yet.")

        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE

        self.tokenizer = HFTokenizer(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            context_length=256,
            **{},
        )

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        decoder_cfg = getattr(cfg.MODEL, "DECODER", None)
        self.decoder_type = getattr(decoder_cfg, "TYPE", "baseline") if decoder_cfg is not None else "baseline"

        # Old single-stream decoder params
        self.skip_layer = getattr(decoder_cfg, "SKIP_LAYER", 8) if decoder_cfg is not None else 8
        self.use_rlab = getattr(decoder_cfg, "USE_RLAB", self.decoder_type != "baseline") if decoder_cfg is not None else (self.decoder_type != "baseline")
        self.use_text_gate = getattr(decoder_cfg, "USE_TEXT_GATE", self.decoder_type != "baseline") if decoder_cfg is not None else (self.decoder_type != "baseline")

        # New aggressive W-PVL params
        self.global_layers = getattr(decoder_cfg, "GLOBAL_LAYERS", [2, 5, 8, 11]) if decoder_cfg is not None else [2, 5, 8, 11]
        self.local_channels = getattr(decoder_cfg, "LOCAL_CHANNELS", [64, 128, 256, 512]) if decoder_cfg is not None else [64, 128, 256, 512]

        self.decoder = None
        self.local_encoder = None
        self.global_pyramid = None
        self.w_decoder = None

        if self.decoder_type == "w_pvl":
            self.local_encoder = LocalCNNEncoder(
                in_channels=3,
                channels=tuple(self.local_channels)
            )

            self.global_pyramid = GlobalPyramidBuilder(
                clip_dim=self.embed_dim,
                out_channels=tuple(self.local_channels),
                selected_layers=tuple(self.global_layers)
            )

            self.w_decoder = WPVLDecoder(
                channels=tuple(self.local_channels),
                text_dim=self.text_proj_dim,
                out_dim=self.text_proj_dim
            )

        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(
                in_channels_vis=self.embed_dim,
                in_channels_txt=self.embed_dim,
                adapter_channels=adapter_channels,
                beta=self.beta,
                gate_init=self.gate_init
            )
            for _ in range(len(self.fusion_stages))
        ])

    def encode_text_image(
        self,
        tokenized_prompts,
        text_prompts,
        image,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()

        x_txt = self.text_model.transformer.embeddings(
            inputs_embeds=text_prompts
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)
        x_img = x_img.permute(0, 2, 1)

        x_img = torch.cat(
            [
                self.vision_model.class_embedding.to(x_img.dtype)
                + torch.zeros(
                    x_img.shape[0], 1, x_img.shape[-1],
                    dtype=x_img.dtype, device=x_img.device
                ),
                x_img
            ],
            dim=1
        )

        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)  # NLD -> LND

        hidden_states = []

        for i, (block, layer) in enumerate(
            zip(self.vision_model.transformer.resblocks, self.text_model.transformer.encoder.layer)
        ):
            if i in self.fusion_stages:
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](
                    x_img.transpose(1, 0), x_txt
                )

                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1, 0)

            x_img = block(x_img)
            x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            hidden_states.append(x_img)   # keep mid-layer vision features
            x_txt = x_txt[0]

        x_img = x_img.permute(1, 0, 2)  # LND -> NLD
        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)
        x_txt = self.text_model.proj(x_txt)

        if self.output_hidden_states:
            return x_img, hidden_states, projected
        else:
            return x_img, projected

    def _hidden_state_to_map(self, hidden_state, B, H, W):
        """
        Convert hidden_state to [B, C, h_patch, w_patch].

        hidden_state is typically saved in LND format during the transformer loop.
        """
        h_patch = H // self.patch_size
        w_patch = W // self.patch_size
        expected_tokens = h_patch * w_patch + 1

        if hidden_state.dim() != 3:
            raise ValueError(f"hidden_state must be 3D, got shape={hidden_state.shape}")

        # Case 1: [L, B, C] -> [B, L, C]
        if hidden_state.shape[0] == expected_tokens and hidden_state.shape[1] == B:
            hidden_state = hidden_state.permute(1, 0, 2).contiguous()

        # Case 2: already [B, L, C]
        elif hidden_state.shape[0] == B and hidden_state.shape[1] == expected_tokens:
            hidden_state = hidden_state.contiguous()

        else:
            raise ValueError(
                f"Unexpected hidden_state shape={hidden_state.shape}, "
                f"expected either [L,B,C] or [B,L,C] with L={expected_tokens}, B={B}"
            )

        hidden_state = hidden_state[:, 1:, :]  # remove CLS
        hidden_state = hidden_state.reshape(B, h_patch, w_patch, self.embed_dim).permute(0, 3, 1, 2)
        return hidden_state

    def _select_global_feature_maps(self, hidden_states, B, H, W):
        if hidden_states is None:
            raise ValueError("hidden_states is required for w_pvl_sr decoder.")

        selected = []
        num_states = len(hidden_states)

        for layer_id in self.global_layers:
            idx = int(layer_id) - 1  # config uses natural layer numbers, e.g. [2,5,8,11]
            idx = max(0, min(idx, num_states - 1))
            feat_map = self._hidden_state_to_map(hidden_states[idx], B, H, W)
            selected.append(feat_map)

        return selected

    def compute_seg_logits(self, image, image_features, text_features, hidden_states, B, H, W, local_feats=None):
        """
        image_features: final projected tokens [B, L, 512]
        text_features: projected text [B, 512]
        hidden_states: list of visual hidden states
        local_feats: optional local CNN pyramid for w_pvl
        """
        decoder_text = text_features
        norm_text = text_features / text_features.norm(dim=-1, keepdim=True)

        cls_token = image_features[:, 0, :]
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

        if self.decoder_type == "w_pvl":
            if local_feats is None:
                local_feats = self.local_encoder(image)

            global_feats = self.global_pyramid(
                hidden_states=hidden_states,
                B=B,
                H=H,
                W=W,
                patch_size=self.patch_size,
            )

            seg_feats = self.w_decoder(
                local_feats=local_feats,
                global_feats=global_feats,
                text_feat=decoder_text,
            )  # [B, 512, 56, 56]

            seg_feats = F.normalize(seg_feats, dim=1)

        else:
            seg_feats = image_features[:, 1:, :]
            seg_feats = seg_feats / seg_feats.norm(dim=-1, keepdim=True)

            h_patch = H // self.patch_size
            w_patch = W // self.patch_size
            seg_feats = seg_feats.reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2)

            skip_feat = None
            if hidden_states is not None:
                skip_idx = max(0, min(self.skip_layer, len(hidden_states) - 1))
                skip_feat = self._hidden_state_to_map(hidden_states[skip_idx], B, H, W)

            seg_feats = self.decoder(
                deep_feat=seg_feats,
                skip_feat=skip_feat,
                text_feat=decoder_text,
            )
            seg_feats = F.normalize(seg_feats, dim=1)

        seg_logits = torch.einsum(
            "bqc, bchw -> bqhw",
            self.mask_head(norm_text).unsqueeze(1),
            seg_feats,
        )

        seg_logits = F.interpolate(
            seg_logits,
            self.im_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        return seg_logits, cls_token

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def forward(self, image, text, num_samples=30):
        B, C, H, W = image.shape

        tokenized_prompts = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)

        # local branch is deterministic, compute once
        local_feats = self.local_encoder(image) if self.decoder_type == "w_pvl" else None

        encoded = self.encode_text_image(tokenized_prompts, prompts, image)

        if self.output_hidden_states:
            image_features, hidden_states, text_features = encoded
        else:
            image_features, text_features = encoded
            hidden_states = None

        seg_logits, cls_token = self.compute_seg_logits(
            image=image,
            image_features=image_features,
            text_features=text_features,
            hidden_states=hidden_states,
            local_feats=local_feats,
            B=B,
            H=H,
            W=W,
        )

        if self.training:
            patch_logits = image_features[:, 1:, :]
            patch_logits = patch_logits / patch_logits.norm(dim=-1, keepdim=True)
            patch_mean = patch_logits.mean(dim=1)

            logits_per_image = (patch_mean @ text_features.T) / self.temperature
            logits_per_text = (text_features @ patch_mean.T) / self.temperature

            with torch.no_grad():
                text_sim = (text_features @ text_features.T) / self.temperature
                text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
                soft_targets = F.softmax(text_sim, dim=-1)

            loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
            loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)
            clip_loss = (loss_i2t + loss_t2i) / 2

            return seg_logits, clip_loss

        else:
            seg_samples = []
            for _ in range(num_samples):
                encoded = self.encode_text_image(tokenized_prompts, prompts, image)

                if self.output_hidden_states:
                    image_features, hidden_states, text_features = encoded
                else:
                    image_features, text_features = encoded
                    hidden_states = None

                seg_logits, _ = self.compute_seg_logits(
                    image=image,
                    image_features=image_features,
                    text_features=text_features,
                    hidden_states=hidden_states,
                    local_feats=local_feats,   # reuse deterministic local branch
                    B=B,
                    H=H,
                    W=W,
                )
                seg_samples.append(seg_logits)

            seg_samples = torch.stack(seg_samples, dim=0)
            return seg_samples


def build_medclipseg_unimedclip(cfg):
    print(f"Loading UniMedCLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_unimedclip_to_device(cfg)

    clip_model.float()

    print("Building custom UniMedCLIP")
    model = CustomCLIP(cfg, clip_model, output_hidden_states=True)

    print("Turning off gradients in both the image and the text encoder")

    for name, param in model.named_parameters():
        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)
        elif "decoder" in name:
            param.requires_grad_(True)
        elif "local_encoder" in name:
            param.requires_grad_(True)
        elif "global_pyramid" in name:
            param.requires_grad_(True)
        elif "w_decoder" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model