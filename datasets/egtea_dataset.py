from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClipRecord:
    clip_stem: str
    video_session: str
    video_path: Path
    label_id: int


def parse_clip_session(clip_stem: str) -> str:
    parts = clip_stem.split("-")
    if len(parts) < 5:
        return clip_stem
    return "-".join(parts[:-4])


def parse_clip_order_key(clip_stem: str) -> int:
    parts = clip_stem.split("-")
    for token in reversed(parts):
        if token.startswith("F") and token[1:].isdigit():
            return int(token[1:])
        if token.isdigit():
            return int(token)
    return 0


def read_split_entries(split_file: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for raw in split_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Official EGTEA split files are 1-based while our classifier uses 0-based labels.
        rows.append((parts[0].removesuffix(".mp4"), int(parts[1]) - 1))
    return rows


def locate_clip_path(data_root: Path, clip_stem: str) -> Path:
    direct = data_root / "cropped_clips" / f"{clip_stem}.mp4"
    if direct.exists():
        return direct
    session_dir = data_root / "cropped_clips" / parse_clip_session(clip_stem)
    candidate = session_dir / f"{clip_stem}.mp4"
    if candidate.exists():
        return candidate
    matches = list((data_root / "cropped_clips").rglob(f"{clip_stem}.mp4"))
    if not matches:
        raise FileNotFoundError(f"Could not locate clip {clip_stem}.mp4 under {data_root / 'cropped_clips'}")
    return matches[0]


def build_split_records(
    data_root: Path,
    split_id: int,
    split_name: str,
    max_samples: int | None = None,
    seed: int | None = None,
    split_file: Path | None = None,
) -> list[ClipRecord]:
    del seed
    split_path = split_file
    if split_path is None:
        split_path = data_root / f"{split_name}_split{split_id}.txt"
    rows = read_split_entries(split_path)
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    return [
        ClipRecord(
            clip_stem=clip_stem,
            video_session=parse_clip_session(clip_stem),
            video_path=locate_clip_path(data_root, clip_stem),
            label_id=label_id,
        )
        for clip_stem, label_id in rows
    ]


def export_split_manifest(records: list[ClipRecord], output_path: Path) -> None:
    payload = [
        {
            "clip_stem": record.clip_stem,
            "video_session": record.video_session,
            "video_path": str(record.video_path),
            "label_id": record.label_id,
        }
        for record in records
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def decode_video_rgb_frames(video_path: Path) -> list[torch.Tensor]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[torch.Tensor] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb.copy()))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def sample_uniform_indices(num_frames_total: int, num_frames_target: int) -> list[int]:
    if num_frames_total <= 0:
        raise ValueError("num_frames_total must be positive")
    if num_frames_total >= num_frames_target:
        values = torch.linspace(0, num_frames_total - 1, steps=num_frames_target)
        return [int(round(v.item())) for v in values]
    values = torch.linspace(0, num_frames_total - 1, steps=num_frames_target)
    indices = [int(round(v.item())) for v in values]
    return [max(0, min(num_frames_total - 1, idx)) for idx in indices]


def _resize_frames(frames: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)


def _center_crop(frames: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, _, height, width = frames.shape
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return frames[:, :, top : top + crop_size, left : left + crop_size]


def _random_resized_crop(frames: torch.Tensor, crop_size: int, scale: tuple[float, float], ratio: tuple[float, float]) -> torch.Tensor:
    _, _, height, width = frames.shape
    area = float(height * width)
    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(scale[0], scale[1]).item())
        aspect = float(torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item())
        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / aspect) ** 0.5))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
            left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
            cropped = frames[:, :, top : top + crop_h, left : left + crop_w]
            return F.interpolate(cropped, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    resized = _resize_frames(frames, crop_size)
    return _center_crop(resized, crop_size)


def _apply_color_jitter(frames: torch.Tensor, brightness: float, contrast: float, saturation: float) -> torch.Tensor:
    out = frames
    if brightness > 0:
        factor = float(torch.empty(1).uniform_(1 - brightness, 1 + brightness).item())
        out = torch.clamp(out * factor, 0.0, 1.0)
    if contrast > 0:
        factor = float(torch.empty(1).uniform_(1 - contrast, 1 + contrast).item())
        mean = out.mean(dim=(2, 3), keepdim=True)
        out = torch.clamp((out - mean) * factor + mean, 0.0, 1.0)
    if saturation > 0:
        factor = float(torch.empty(1).uniform_(1 - saturation, 1 + saturation).item())
        gray = out.mean(dim=1, keepdim=True)
        out = torch.clamp((out - gray) * factor + gray, 0.0, 1.0)
    return out


def apply_spatial_transform(
    frames: torch.Tensor,
    resize_size: int,
    crop_size: int,
    is_train: bool,
    augmentation: dict | None,
) -> torch.Tensor:
    frames = _resize_frames(frames, resize_size)
    train_aug = (augmentation or {}).get("train", {}) if is_train else {}
    rrc = train_aug.get("random_resized_crop", {})
    if is_train and bool(rrc.get("enabled", False)):
        scale = tuple(rrc.get("scale", [0.8, 1.0]))
        ratio = tuple(rrc.get("ratio", [0.9, 1.1]))
        frames = _random_resized_crop(frames, crop_size=crop_size, scale=scale, ratio=ratio)
    else:
        frames = _center_crop(frames, crop_size)
    if is_train and float(train_aug.get("horizontal_flip_prob", 0.0)) > 0:
        if torch.rand(1).item() < float(train_aug.get("horizontal_flip_prob", 0.0)):
            frames = torch.flip(frames, dims=[3])
    jitter = train_aug.get("color_jitter", {})
    if is_train and bool(jitter.get("enabled", False)):
        frames = _apply_color_jitter(
            frames,
            brightness=float(jitter.get("brightness", 0.2)),
            contrast=float(jitter.get("contrast", 0.2)),
            saturation=float(jitter.get("saturation", 0.2)),
        )
    return frames


def normalize_video(frames: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=frames.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean_tensor) / std_tensor


class EgteaVideoDataset(Dataset):
    def __init__(
        self,
        records: list[ClipRecord],
        num_frames: int,
        resize_size: int,
        crop_size: int,
        mean: list[float],
        std: list[float],
        is_train: bool,
        augmentation: dict | None = None,
    ) -> None:
        self.records = records
        self.num_frames = int(num_frames)
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.mean = list(mean)
        self.std = list(std)
        self.is_train = bool(is_train)
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        record = self.records[index]
        decoded_frames = decode_video_rgb_frames(record.video_path)
        frame_indices = sample_uniform_indices(len(decoded_frames), self.num_frames)
        sampled = [decoded_frames[i] for i in frame_indices]
        video = torch.stack(sampled, dim=0).permute(0, 3, 1, 2).float() / 255.0
        video = apply_spatial_transform(
            frames=video,
            resize_size=self.resize_size,
            crop_size=self.crop_size,
            is_train=self.is_train,
            augmentation=self.augmentation,
        )
        video = normalize_video(video, self.mean, self.std)
        return {
            "video": video,
            "label": torch.tensor(record.label_id, dtype=torch.long),
            "clip_stem": record.clip_stem,
            "session_id": record.video_session,
        }
