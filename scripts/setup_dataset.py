#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "data" / "egtea_gaze_plus"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class FileSpec:
    path: Path
    url: str


def default_manifest() -> list[FileSpec]:
    return [
        FileSpec(path=Path("readme.md"), url="https://www.dropbox.com/s/i0qdxz484ufai5m/readme.md?dl=1"),
        FileSpec(path=Path("Recipes.pdf"), url="https://www.dropbox.com/s/w260trfnhdfcooh/Recipes.pdf?dl=1"),
        FileSpec(path=Path("action_annotation.zip"), url="https://www.dropbox.com/s/ksro6eqa6v59859/action_annotation.zip?dl=1"),
        FileSpec(path=Path("gaze_data.zip"), url="https://www.dropbox.com/s/2aryvztw044w9ih/gaze_data.zip?dl=1"),
        FileSpec(path=Path("hand14k.zip"), url="https://www.dropbox.com/s/ysi2jv8qr9xvzli/hand14k.zip?dl=1"),
        FileSpec(path=Path("video_clips.tar"), url="https://www.dropbox.com/s/udynz2u62wpdva6/video_clips.tar?dl=1"),
    ]


def normalize_download_url(url: str) -> str:
    parsed = urlparse(url)
    if "dropbox.com" not in parsed.netloc.lower():
        return url
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def file_is_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def get_remote_file_size(url: str) -> int | None:
    request = Request(normalize_download_url(url), headers={"User-Agent": "egtea-demo-ready/1.0"}, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            total = response.headers.get("Content-Length")
            if total and total.isdigit():
                return int(total)
    except Exception:
        pass

    # Some providers do not honor HEAD; fall back to a streamed GET and only read headers.
    request = Request(normalize_download_url(url), headers={"User-Agent": "egtea-demo-ready/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            total = response.headers.get("Content-Length")
            if total and total.isdigit():
                return int(total)
    except Exception:
        pass
    return None


def print_progress(prefix: str, downloaded: int, total: int | None) -> None:
    if total and total > 0:
        ratio = min(max(downloaded / total, 0.0), 1.0)
        width = 28
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        message = f"\r{prefix} [{bar}] {ratio * 100:5.1f}% ({downloaded / (1024**2):.1f}/{total / (1024**2):.1f} MB)"
    else:
        message = f"\r{prefix} {downloaded / (1024**2):.1f} MB"
    print(message, end="", flush=True)


def iterate_with_progress(items: list, prefix: str):
    total = len(items)
    if total <= 0:
        print(f"{prefix}: nothing to process")
        return
    for index, item in enumerate(items, start=1):
        ratio = index / total
        width = 28
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{prefix} [{bar}] {ratio * 100:5.1f}% ({index}/{total})", end="", flush=True)
        yield item
    print()


def validate_archive(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    try:
        if suffixes.endswith(".zip"):
            with zipfile.ZipFile(path, "r") as zf:
                return len(zf.infolist()) > 0
        if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
            with tarfile.open(path, "r:*") as tf:
                member = next(iter(tf), None)
                return member is not None
    except Exception:
        return False
    return file_is_ready(path)


def spec_is_ready(spec: FileSpec, destination: Path, expected_size: int | None) -> bool:
    if not file_is_ready(destination):
        return False
    if expected_size is not None and destination.stat().st_size != expected_size:
        return False
    if destination.suffix.lower() in {".zip", ".tar", ".gz"} or "".join(destination.suffixes).lower().endswith(".tar.gz"):
        return validate_archive(destination)
    return True


def download_file(spec: FileSpec, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(normalize_download_url(spec.url), headers={"User-Agent": "egtea-demo-ready/1.0"})
    with urlopen(request, timeout=60) as response, temp_path.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            print_progress(f"Downloading {spec.path.name}", downloaded, total_bytes)
    print()
    temp_path.replace(destination)


def ensure_files() -> tuple[int, int, int, dict[str, int | None]]:
    downloaded = 0
    reused = 0
    repaired = 0
    remote_sizes: dict[str, int | None] = {}
    for spec in default_manifest():
        target = DATA_ROOT / spec.path
        expected_size = get_remote_file_size(spec.url)
        remote_sizes[str(spec.path)] = expected_size
        if spec_is_ready(spec, target, expected_size):
            reused += 1
            continue
        if target.exists():
            target.unlink()
            repaired += 1
        download_file(spec, target)
        downloaded += 1
    return downloaded, reused, repaired, remote_sizes


def build_internal_split() -> None:
    source_path = DATA_ROOT / "train_split1.txt"
    entries = [line.strip() for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    session_to_entries: dict[str, list[str]] = {}
    for entry in entries:
        clip_stem = entry.split()[0]
        session = "-".join(clip_stem.split("-")[:-4])
        session_to_entries.setdefault(session, []).append(entry)
    sessions = sorted(session_to_entries)
    rng = random.Random(42)
    rng.shuffle(sessions)
    val_count = max(1, round(len(sessions) * 0.15))
    val_sessions = set(sessions[:val_count])
    train_entries: list[str] = []
    val_entries: list[str] = []
    for session in sorted(session_to_entries):
        if session in val_sessions:
            val_entries.extend(session_to_entries[session])
        else:
            train_entries.extend(session_to_entries[session])
    (DATA_ROOT / "train_internal_split1.txt").write_text("\n".join(train_entries) + "\n", encoding="utf-8")
    (DATA_ROOT / "val_internal_split1.txt").write_text("\n".join(val_entries) + "\n", encoding="utf-8")


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def extract_zip_to_temp(archive_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{archive_path.stem}_", dir=str(DATA_ROOT)))
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        for member in iterate_with_progress(members, f"Extracting {archive_path.name}"):
            zf.extract(member, temp_dir)
    return temp_dir


def extract_tar_to_temp(archive_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{archive_path.stem}_", dir=str(DATA_ROOT)))
    with tarfile.open(archive_path, "r:*") as tf:
        members = [member for member in tf.getmembers()]
        for member in iterate_with_progress(members, f"Extracting {archive_path.name}"):
            tf.extract(member, temp_dir, filter="data")
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
    for split_name in ["train_split1.txt", "train_split2.txt", "train_split3.txt", "test_split1.txt", "test_split2.txt", "test_split3.txt"]:
        source = find_first_path(temp_root, split_name)
        if source is not None:
            shutil.copy2(source, DATA_ROOT / split_name)


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

    split_targets = [DATA_ROOT / f"train_split{i}.txt" for i in range(1, 4)] + [DATA_ROOT / f"test_split{i}.txt" for i in range(1, 4)]
    need_action_extract = not (DATA_ROOT / "raw_annotations" / "action_labels.csv").exists() or any(not path.exists() for path in split_targets)
    if action_zip.exists() and need_action_extract:
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
    build_internal_split()


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
    downloaded, reused, repaired, remote_sizes = ensure_files()
    ensure_extracted_layout()
    ensure_internal_split()
    write_structure_note()

    summary = {
        "data_root": str(DATA_ROOT),
        "downloaded": downloaded,
        "reused": reused,
        "repaired": repaired,
        "remote_sizes": remote_sizes,
        "cropped_clips_ready": (DATA_ROOT / "cropped_clips").exists(),
        "raw_annotations_ready": (DATA_ROOT / "raw_annotations" / "action_labels.csv").exists(),
        "train_internal_ready": (DATA_ROOT / "train_internal_split1.txt").exists(),
        "val_internal_ready": (DATA_ROOT / "val_internal_split1.txt").exists(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
