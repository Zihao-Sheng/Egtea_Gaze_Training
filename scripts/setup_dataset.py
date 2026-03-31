#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_retrieval import DEFAULT_ROOT, default_manifest, ensure_files


DATA_ROOT = DEFAULT_ROOT


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def extract_zip_to_temp(archive_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{archive_path.stem}_", dir=str(DATA_ROOT)))
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(temp_dir)
    return temp_dir


def extract_tar_to_temp(archive_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{archive_path.stem}_", dir=str(DATA_ROOT)))
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(temp_dir)
    return temp_dir


def find_first_path(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def normalize_action_annotations(temp_root: Path) -> None:
    target_dir = DATA_ROOT / "raw_annotations"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["action_labels.csv", "cls_label_index.csv", "noun_idx.txt", "verb_idx.txt", "action_idx.txt"]:
        source = find_first_path(temp_root, filename)
        if source is not None:
            shutil.copy2(source, target_dir / filename)


def normalize_gaze_data(temp_root: Path) -> None:
    target_dir = DATA_ROOT / "gaze_data"
    source_dir = find_first_path(temp_root, "gaze_data")
    if source_dir is not None and source_dir.is_dir():
        copytree_replace(source_dir, target_dir)
        return
    copytree_replace(temp_root, target_dir)


def normalize_hand_data(temp_root: Path) -> None:
    images_dir = find_first_path(temp_root, "Images")
    masks_dir = find_first_path(temp_root, "Masks")
    if images_dir is None or masks_dir is None:
        raise RuntimeError("hand14k.zip did not contain expected Images/ and Masks/ folders.")
    copytree_replace(images_dir, DATA_ROOT / "Images")
    copytree_replace(masks_dir, DATA_ROOT / "Masks")


def normalize_video_clips(temp_root: Path) -> None:
    source_dir = find_first_path(temp_root, "cropped_clips")
    if source_dir is not None and source_dir.is_dir():
        copytree_replace(source_dir, DATA_ROOT / "cropped_clips")
        return

    mp4_files = list(temp_root.rglob("*.mp4"))
    if not mp4_files:
        raise RuntimeError("video_clips.tar did not contain any .mp4 files.")

    target_dir = DATA_ROOT / "cropped_clips"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for session_dir in [path for path in temp_root.iterdir() if path.is_dir()]:
        if list(session_dir.glob("*.mp4")):
            shutil.copytree(session_dir, target_dir / session_dir.name, dirs_exist_ok=True)


def ensure_extracted_layout() -> None:
    action_zip = DATA_ROOT / "action_annotation.zip"
    gaze_zip = DATA_ROOT / "gaze_data.zip"
    hand_zip = DATA_ROOT / "hand14k.zip"
    clips_tar = DATA_ROOT / "video_clips.tar"

    if action_zip.exists() and not (DATA_ROOT / "raw_annotations" / "action_labels.csv").exists():
        temp_root = extract_zip_to_temp(action_zip)
        try:
            normalize_action_annotations(temp_root)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    if gaze_zip.exists() and not (DATA_ROOT / "gaze_data").exists():
        temp_root = extract_zip_to_temp(gaze_zip)
        try:
            normalize_gaze_data(temp_root)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    if hand_zip.exists() and not ((DATA_ROOT / "Images").exists() and (DATA_ROOT / "Masks").exists()):
        temp_root = extract_zip_to_temp(hand_zip)
        try:
            normalize_hand_data(temp_root)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    if clips_tar.exists() and not (DATA_ROOT / "cropped_clips").exists():
        temp_root = extract_tar_to_temp(clips_tar)
        try:
            normalize_video_clips(temp_root)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


def ensure_internal_split() -> None:
    train_internal = DATA_ROOT / "train_internal_split1.txt"
    val_internal = DATA_ROOT / "val_internal_split1.txt"
    if train_internal.exists() and val_internal.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_internal_val_split.py"), "--split-id", "1"],
        check=True,
        cwd=ROOT,
    )


def write_structure_note() -> None:
    structure_path = DATA_ROOT / "LOCAL_STRUCTURE.md"
    structure_path.write_text(
        "\n".join(
            [
                "EGTEA Gaze+ local structure",
                "",
                "Main directories",
                "- `cropped_clips/`: action clips in `.mp4` format, grouped by recording/session.",
                "- `gaze_data/`: gaze tracking files from the dataset release.",
                "- `raw_annotations/`: annotation CSV files such as `action_labels.csv` and `cls_label_index.csv`.",
                "- `Images/`: hand image frames from the hand annotation package.",
                "- `Masks/`: hand mask images paired with `Images/`.",
                "",
                "Generated split files",
                "- `train_internal_split1.txt`",
                "- `val_internal_split1.txt`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    downloaded, reused, planned = ensure_files(
        data_root=DATA_ROOT,
        manifest=default_manifest(),
        force_download=False,
        timeout_s=60,
        dry_run=False,
    )
    ensure_extracted_layout()
    ensure_internal_split()
    write_structure_note()

    summary = {
        "data_root": str(DATA_ROOT),
        "downloaded": downloaded,
        "reused": reused,
        "planned": planned,
        "cropped_clips_ready": (DATA_ROOT / "cropped_clips").exists(),
        "raw_annotations_ready": (DATA_ROOT / "raw_annotations" / "action_labels.csv").exists(),
        "train_internal_ready": (DATA_ROOT / "train_internal_split1.txt").exists(),
        "val_internal_ready": (DATA_ROOT / "val_internal_split1.txt").exists(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
