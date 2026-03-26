#!/usr/bin/env python3
"""
EGTEA Gaze+ data retrieval utility.

Behavior:
1) Check whether EGTEA files already exist locally and reuse them.
2) Download only missing files with a progress bar.
3) Support custom manifest override for alternative mirrors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

CHUNK_SIZE = 1024 * 1024
DATASET_NAME = "egtea_gaze_plus"
DEFAULT_ROOT = Path("data") / DATASET_NAME
DEFAULT_MANIFEST_CANDIDATES = [
    Path("manifests/egtea_gaze_plus_manifest.private.json"),
    Path("manifests/egtea_gaze_plus_manifest.json"),
]
ANNOTATION_FILE_BASENAMES = {
    "readme.md",
    "recipes.pdf",
    "action_annotation.zip",
}


@dataclass(frozen=True)
class FileSpec:
    path: Path
    url: str
    sha256: str | None = None
    size_bytes: int | None = None


def default_manifest() -> list[FileSpec]:
    # URLs are published on the official Georgia Tech FPV dataset page.
    return [
        FileSpec(path=Path("readme.md"), url="https://www.dropbox.com/s/i0qdxz484ufai5m/readme.md?dl=1"),
        FileSpec(path=Path("Recipes.pdf"), url="https://www.dropbox.com/s/w260trfnhdfcooh/Recipes.pdf?dl=1"),
        FileSpec(
            path=Path("action_annotation.zip"),
            url="https://www.dropbox.com/s/ksro6eqa6v59859/action_annotation.zip?dl=1",
        ),
        FileSpec(path=Path("gaze_data.zip"), url="https://www.dropbox.com/s/2aryvztw044w9ih/gaze_data.zip?dl=1"),
        FileSpec(path=Path("hand14k.zip"), url="https://www.dropbox.com/s/ysi2jv8qr9xvzli/hand14k.zip?dl=1"),
        FileSpec(path=Path("video_clips.tar"), url="https://www.dropbox.com/s/udynz2u62wpdva6/video_clips.tar?dl=1"),
    ]


def human_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{value:.2f}B"


def render_progress(label: str, downloaded: int, total: int, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    speed = downloaded / elapsed
    if total > 0:
        ratio = min(downloaded / total, 1.0)
        width = 30
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r{label} [{bar}] {ratio * 100:6.2f}% "
            f"{human_bytes(downloaded)}/{human_bytes(total)} "
            f"{human_bytes(speed)}/s"
        )
    else:
        message = f"\r{label} {human_bytes(downloaded)} {human_bytes(speed)}/s"
    print(message, end="", flush=True)


def is_safe_relative_path(path: Path) -> bool:
    if path.is_absolute():
        return False
    return ".." not in path.parts


def normalize_download_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if "dropbox.com" not in netloc:
        return url

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def file_is_ready(target: Path, spec: FileSpec) -> bool:
    if not target.exists() or not target.is_file():
        return False
    if target.stat().st_size <= 0:
        return False
    if spec.size_bytes is not None and target.stat().st_size != spec.size_bytes:
        return False
    if spec.sha256 is not None:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest().lower() != spec.sha256.lower():
            return False
    return True


def download_file(spec: FileSpec, destination: Path, timeout_s: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = destination.with_suffix(destination.suffix + ".part")
    if temp_file.exists():
        temp_file.unlink()

    request_url = normalize_download_url(spec.url)
    request = Request(request_url, headers={"User-Agent": "egtea-gaze-plus-data-retrieval/1.0"})
    hasher = hashlib.sha256() if spec.sha256 else None
    start_time = time.time()
    downloaded = 0

    try:
        with urlopen(request, timeout=timeout_s) as response, temp_file.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if hasher is not None:
                    hasher.update(chunk)
                render_progress(destination.name, downloaded, total, start_time)
        print()
    except (HTTPError, URLError, TimeoutError) as exc:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to download {request_url}: {exc}") from exc
    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise

    if downloaded <= 0:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Downloaded file is empty: {destination}")
    if spec.size_bytes is not None and downloaded != spec.size_bytes:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(
            f"File size mismatch for {destination}. "
            f"Expected {spec.size_bytes}, got {downloaded}."
        )
    if hasher is not None and hasher.hexdigest().lower() != spec.sha256.lower():
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"SHA256 mismatch for {destination}")

    temp_file.replace(destination)


def load_manifest(path: Path) -> list[FileSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a list of file objects.")

    manifest: list[FileSpec] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item #{idx} must be an object.")
        if "path" not in item or "url" not in item:
            raise ValueError(f"Manifest item #{idx} must include 'path' and 'url'.")

        rel_path = Path(str(item["path"]))
        if not is_safe_relative_path(rel_path):
            raise ValueError(
                f"Manifest item #{idx} has unsafe path '{rel_path}'. "
                "Use a relative path under dataset root."
            )

        manifest.append(
            FileSpec(
                path=rel_path,
                url=str(item["url"]),
                sha256=str(item["sha256"]) if item.get("sha256") else None,
                size_bytes=int(item["size_bytes"]) if item.get("size_bytes") is not None else None,
            )
        )
    return manifest


def resolve_manifest_path(user_value: Path | None) -> Path | None:
    if user_value:
        return user_value
    for candidate in DEFAULT_MANIFEST_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def ensure_files(
    data_root: Path,
    manifest: Iterable[FileSpec],
    force_download: bool = False,
    timeout_s: int = 60,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    downloaded_count = 0
    reused_count = 0
    planned_download_count = 0
    data_root.mkdir(parents=True, exist_ok=True)

    for spec in manifest:
        target = data_root / spec.path
        if not force_download and file_is_ready(target, spec):
            print(f"[reuse] {target}")
            reused_count += 1
            continue

        if dry_run:
            print(f"[plan-download] {target} <- {normalize_download_url(spec.url)}")
            planned_download_count += 1
            continue

        print(f"[download] {target}")
        download_file(spec, target, timeout_s=timeout_s)
        downloaded_count += 1

    return downloaded_count, reused_count, planned_download_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve EGTEA Gaze+ with cache detection and progress bars."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON manifest override. "
            "If omitted, script uses default EGTEA links or local manifests/egtea_gaze_plus_manifest*.json."
        ),
    )
    parser.add_argument(
        "--core-manifest",
        dest="manifest",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--subset",
        choices=["all", "annotations"],
        default="all",
        help="Download subset. 'annotations' = readme + Recipes + action annotations.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local files already look valid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print reuse/download decisions without downloading files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout (seconds) per HTTP request. Default: 60",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    subset = "annotations" if args.annotations_only else args.subset

    manifest_path = resolve_manifest_path(args.manifest)
    if manifest_path is not None:
        print(f"Using custom manifest: {manifest_path.resolve()}")
        manifest = load_manifest(manifest_path)
    else:
        print("Using built-in EGTEA Gaze+ official links.")
        manifest = default_manifest()

    if subset == "annotations":
        manifest = [
            spec
            for spec in manifest
            if spec.path.name.lower() in ANNOTATION_FILE_BASENAMES
        ]

    if not manifest:
        raise RuntimeError("Selected manifest is empty; nothing to download.")

    print(f"Dataset: {DATASET_NAME}")
    print(f"Data root: {data_root.resolve()}")
    print(f"Subset: {subset}")
    if args.dry_run:
        print("Mode: dry-run")

    downloaded, reused, planned = ensure_files(
        data_root=data_root,
        manifest=manifest,
        force_download=args.force,
        timeout_s=args.timeout,
        dry_run=args.dry_run,
    )

    print()
    print("Summary:")
    print(f"- Downloaded: {downloaded}")
    print(f"- Reused local files: {reused}")
    print(f"- Planned downloads: {planned}")
    print(f"- Final path: {data_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
