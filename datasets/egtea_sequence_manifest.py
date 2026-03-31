from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .egtea_dataset import build_split_records, parse_clip_order_key


@dataclass(frozen=True)
class SequenceRecord:
    clip_stem: str
    session_id: str
    video_path: Path
    label: int
    clip_idx: int


def build_sequence_records(
    data_root: Path,
    split_name: str,
    split_id: int = 1,
    manifest_split_name: str | None = None,
) -> list[SequenceRecord]:
    del manifest_split_name
    if split_name in {"train_internal", "val_internal"}:
        split_file = data_root / f"{split_name}_split{split_id}.txt"
    else:
        split_file = data_root / f"{split_name}_split{split_id}.txt"
    records = build_split_records(
        data_root=data_root,
        split_id=split_id,
        split_name=split_name,
        split_file=split_file,
    )
    return [
        SequenceRecord(
            clip_stem=record.clip_stem,
            session_id=record.video_session,
            video_path=record.video_path,
            label=record.label_id,
            clip_idx=parse_clip_order_key(record.clip_stem),
        )
        for record in records
    ]
