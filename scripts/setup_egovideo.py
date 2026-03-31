#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

THIRD_PARTY_ROOT = ROOT / "third_party"
REPO_ROOT = THIRD_PARTY_ROOT / "EgoVideo-main"
BACKBONE_ROOT = ROOT / "third_party" / "EgoVideo-main" / "backbone"
CHECKPOINT_PATH = BACKBONE_ROOT / "ckpt_4frames.pth"
REPO_ZIP_URL = "https://github.com/OpenGVLab/EgoVideo/archive/refs/heads/main.zip"
CHECKPOINT_URL = "https://drive.google.com/file/d/1k6f1eRdcL17IvXtdX_J8WxNbju2Ms3AW/view?usp=sharing"
CHUNK_SIZE = 1024 * 1024


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


def backbone_code_ready() -> bool:
    return (BACKBONE_ROOT / "model" / "vision_encoder.py").exists()


def get_remote_file_size(url: str) -> int | None:
    request = Request(url, headers={"User-Agent": "egtea-demo-ready/1.0"}, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            total = response.headers.get("Content-Length")
            if total and total.isdigit():
                return int(total)
    except Exception:
        pass
    request = Request(url, headers={"User-Agent": "egtea-demo-ready/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            total = response.headers.get("Content-Length")
            if total and total.isdigit():
                return int(total)
    except Exception:
        pass
    return None


def download_file(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "egtea-demo-ready/1.0"})
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
            print_progress(label, downloaded, total_bytes)
    print()
    temp_path.replace(destination)


def ensure_repo_code() -> None:
    if backbone_code_ready():
        return
    THIRD_PARTY_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = THIRD_PARTY_ROOT / "egovideo_main.zip"
    expected_size = get_remote_file_size(REPO_ZIP_URL)
    if (
        (not zip_path.exists())
        or zip_path.stat().st_size <= 0
        or (expected_size is not None and zip_path.stat().st_size != expected_size)
    ):
        download_file(REPO_ZIP_URL, zip_path, "Downloading EgoVideo repo")
    temp_dir = Path(tempfile.mkdtemp(prefix="egovideo_extract_", dir=str(THIRD_PARTY_ROOT)))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            for member in iterate_with_progress(members, "Extracting EgoVideo repo"):
                zf.extract(member, temp_dir)
        top_level_dirs = sorted(
            {
                Path(member.filename).parts[0]
                for member in members
                if member.filename and not member.filename.startswith("__MACOSX")
            }
        )
        if not top_level_dirs:
            raise RuntimeError("EgoVideo zip was empty.")
        extracted_root = temp_dir / top_level_dirs[0]
        if not extracted_root.exists():
            raise RuntimeError("Unexpected EgoVideo zip layout.")
        if REPO_ROOT.exists():
            shutil.rmtree(REPO_ROOT)
        shutil.move(str(extracted_root), str(REPO_ROOT))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def ensure_gdown() -> None:
    if importlib.util.find_spec("gdown") is not None:
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True, cwd=ROOT)


def ensure_runtime_dependencies() -> None:
    return


def patch_xbert_for_new_transformers() -> None:
    xbert_path = BACKBONE_ROOT / "model" / "bert" / "xbert.py"
    if not xbert_path.exists():
        return
    text = xbert_path.read_text(encoding="utf-8")
    marker = "EGTEA_PATCH_TRANSFORMERS_COMPAT"
    if marker in text:
        return

    old_import = """from transformers.modeling_utils import (PreTrainedModel,\n                                         apply_chunking_to_forward,\n                                         find_pruneable_heads_and_indices,\n                                         prune_linear_layer)\n"""
    new_import = """from transformers.modeling_utils import PreTrainedModel\ntry:\n    from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer\nexcept ImportError:\n    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer\n\n    # EGTEA_PATCH_TRANSFORMERS_COMPAT\n    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):\n        mask = torch.ones(n_heads, head_size)\n        heads = set(heads) - already_pruned_heads\n        for head in heads:\n            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)\n            mask[head] = 0\n        mask = mask.view(-1).contiguous().eq(1)\n        index = torch.arange(len(mask))[mask].long()\n        return heads, index\n"""
    if old_import not in text:
        raise RuntimeError(f"Could not find expected import block in {xbert_path}")
    xbert_path.write_text(text.replace(old_import, new_import), encoding="utf-8")


def patch_flash_attention_fallback() -> None:
    flash_path = BACKBONE_ROOT / "model" / "flash_attention_class.py"
    if not flash_path.exists():
        return
    text = flash_path.read_text(encoding="utf-8")
    marker = "_FLASH_ATTN_AVAILABLE = True"
    if marker in text:
        return

    old_import = """from flash_attn import flash_attn_varlen_qkvpacked_func\nfrom flash_attn.bert_padding import unpad_input, pad_input\n"""
    new_import = """try:\n    from flash_attn import flash_attn_varlen_qkvpacked_func\n    from flash_attn.bert_padding import unpad_input, pad_input\n    _FLASH_ATTN_AVAILABLE = True\nexcept Exception:\n    flash_attn_varlen_qkvpacked_func = None\n    unpad_input = None\n    pad_input = None\n    _FLASH_ATTN_AVAILABLE = False\n"""
    text = text.replace(old_import, new_import)

    old_assert_block = """        assert not need_weights\n        assert qkv.dtype in [torch.float16, torch.bfloat16]\n        assert qkv.is_cuda\n\n        if cu_seqlens is None:\n"""
    new_assert_block = """        assert not need_weights\n\n        if not _FLASH_ATTN_AVAILABLE:\n            q, k, v = qkv.unbind(dim=2)\n            q = q.permute(0, 2, 1, 3)\n            k = k.permute(0, 2, 1, 3)\n            v = v.permute(0, 2, 1, 3)\n            scale = self.softmax_scale if self.softmax_scale is not None else (q.shape[-1] ** -0.5)\n            scores = torch.matmul(q, k.transpose(-2, -1)) * scale\n\n            if key_padding_mask is not None:\n                mask = ~key_padding_mask[:, None, None, :]\n                scores = scores.masked_fill(mask, float(\"-inf\"))\n\n            if causal:\n                seq_len = scores.shape[-1]\n                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)\n                scores = scores.masked_fill(causal_mask, float(\"-inf\"))\n\n            attn = torch.softmax(scores, dim=-1)\n            if self.training and self.dropout_p > 0:\n                attn = torch.nn.functional.dropout(attn, p=self.dropout_p)\n            output = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()\n            return output, None\n\n        assert qkv.dtype in [torch.float16, torch.bfloat16]\n        assert qkv.is_cuda\n\n        if cu_seqlens is None:\n"""
    if old_assert_block not in text:
        raise RuntimeError(f"Could not find expected attention block in {flash_path}")
    text = text.replace(old_assert_block, new_assert_block)
    flash_path.write_text(text, encoding="utf-8")


def patch_vision_encoder_fallbacks() -> None:
    vision_path = BACKBONE_ROOT / "model" / "vision_encoder.py"
    if not vision_path.exists():
        return
    text = vision_path.read_text(encoding="utf-8")
    if "FusedMLP = None" not in text:
        text = text.replace(
            "except:\n    logger.warn(f'FusedMLP of flash_attn is not installed!!!')\n    raise NotImplementedError\n",
            "except:\n    logger.warn(f'FusedMLP of flash_attn is not installed!!!')\n    FusedMLP = None\n",
        )
    if "DropoutAddRMSNorm = None" not in text:
        text = text.replace(
            "except:\n    logger.warn(f'DropoutAddRMSNorm of flash_attn is not installed!!!')\n",
            "except:\n    logger.warn(f'DropoutAddRMSNorm of flash_attn is not installed!!!')\n    DropoutAddRMSNorm = None\n",
        )
    vision_path.write_text(text, encoding="utf-8")


def download_checkpoint() -> None:
    ensure_gdown()
    BACKBONE_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--fuzzy",
            CHECKPOINT_URL,
            "-O",
            str(CHECKPOINT_PATH),
        ],
        check=True,
        cwd=ROOT,
    )


def checkpoint_ready() -> bool:
    return CHECKPOINT_PATH.exists() and CHECKPOINT_PATH.stat().st_size > 0


def main() -> int:
    ensure_runtime_dependencies()
    ensure_repo_code()
    patch_xbert_for_new_transformers()
    patch_flash_attention_fallback()
    patch_vision_encoder_fallbacks()

    if not checkpoint_ready():
        download_checkpoint()

    summary = {
        "repo_root": str(REPO_ROOT),
        "backbone_root": str(BACKBONE_ROOT),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "backbone_code_ready": backbone_code_ready(),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "repo_zip_expected_size": get_remote_file_size(REPO_ZIP_URL),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
