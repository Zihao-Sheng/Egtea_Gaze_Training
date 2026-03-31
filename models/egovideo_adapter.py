from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
EGOVIDEO_BACKBONE_ROOT = ROOT / "third_party" / "EgoVideo-main" / "backbone"
if str(EGOVIDEO_BACKBONE_ROOT) not in sys.path:
    sys.path.insert(0, str(EGOVIDEO_BACKBONE_ROOT))

from model.vision_encoder import PretrainVisionTransformer


class EgoVideoSingleClipClassifier(nn.Module):
    """Official EgoVideo visual backbone plus a lightweight EGTEA classification head."""

    def __init__(
        self,
        num_classes: int,
        checkpoint_path: Path,
        freeze_mode: str = "partial",
        trainable_blocks: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = PretrainVisionTransformer(
            img_size=224,
            num_frames=4,
            tubelet_size=1,
            patch_size=14,
            embed_dim=1408,
            clip_embed_dim=768,
            clip_teacher_embed_dim=3200,
            clip_teacher_final_dim=768,
            clip_norm_type="l2",
            clip_return_layer=6,
            clip_student_return_interval=1,
            use_checkpoint=False,
            checkpoint_num=40,
            use_flash_attn=False,
            use_fused_rmsnorm=False,
            use_fused_mlp=False,
            sep_image_video_pos_embed=False,
        )
        self.embedding_dim = 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

        load_meta = load_egovideo_visual_weights(self.backbone, checkpoint_path)
        self.load_notes = [
            f"Loaded EgoVideo visual checkpoint from {checkpoint_path.as_posix()}",
            f"Checkpoint load missing keys: {len(load_meta['missing_keys'])}",
            f"Checkpoint load unexpected keys: {len(load_meta['unexpected_keys'])}",
        ]
        self.freeze_backbone(freeze_mode=freeze_mode, trainable_blocks=trainable_blocks)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the official pooled clip embedding before the EGTEA classifier."""

        _, pooled, _, _ = self.backbone(x)
        return pooled

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both EGTEA logits and the pooled EgoVideo clip embedding."""

        embeddings = self.extract_features(x)
        logits = self.classifier(self.dropout(embeddings))
        return logits, embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits

    def freeze_backbone(self, freeze_mode: str, trainable_blocks: int) -> None:
        """Configure frozen / partial / full fine-tuning without changing the head."""

        mode = freeze_mode.lower()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if mode == "frozen":
            pass
        elif mode == "partial":
            trainable_blocks = max(0, min(trainable_blocks, len(self.backbone.blocks)))
            if trainable_blocks > 0:
                for block in self.backbone.blocks[-trainable_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
            for module in [
                self.backbone.clip_projector,
                self.backbone.final_clip_decoder,
                self.backbone.patch_embed,
            ]:
                for parameter in module.parameters():
                    parameter.requires_grad = True
            for name in ["cls_token", "pos_embed", "clip_pos_embed"]:
                tensor = getattr(self.backbone, name, None)
                if tensor is not None:
                    tensor.requires_grad = True
        elif mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(f"Unsupported EgoVideo freeze mode: {freeze_mode}")

        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        for parameter in self.dropout.parameters():
            parameter.requires_grad = True


def load_egovideo_visual_weights(model: PretrainVisionTransformer, checkpoint_path: Path) -> dict[str, list[str]]:
    """Load only the official visual weights from the multimodal EgoVideo checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    visual_state_dict = {
        key.replace("module.visual.", ""): value
        for key, value in checkpoint.items()
        if key.startswith("module.visual.")
    }
    result = model.load_state_dict(visual_state_dict, strict=False)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def build_egovideo_model(config: dict, num_classes: int) -> tuple[nn.Module, dict]:
    """Build an EGTEA classifier backed by the official EgoVideo visual encoder."""

    checkpoint_path = Path(config["model"]["pretrained_checkpoint"])
    freeze_mode = str(config["model"].get("freeze_mode", "partial")).lower()
    trainable_blocks = int(config["model"].get("trainable_blocks", 4))
    dropout = float(config["model"].get("classifier_dropout", 0.3))

    model = EgoVideoSingleClipClassifier(
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        freeze_mode=freeze_mode,
        trainable_blocks=trainable_blocks,
        dropout=dropout,
    )

    notes = list(model.load_notes)
    notes.append("Using official EgoVideo visual backbone with 4 RGB frames and 224x224 crops.")
    notes.append(f"Freeze mode: {freeze_mode}")
    if freeze_mode == "partial":
        notes.append(f"Unfroze the last {trainable_blocks} EgoVideo transformer blocks plus projection layers.")
    return model, {"notes": notes}


def prepare_egovideo_inputs(videos: torch.Tensor, _: dict) -> torch.Tensor:
    """EgoVideo expects `[B, T, C, H, W]` from the dataset and converts to `[B, C, T, H, W]`."""

    return videos.permute(0, 2, 1, 3, 4).contiguous()
