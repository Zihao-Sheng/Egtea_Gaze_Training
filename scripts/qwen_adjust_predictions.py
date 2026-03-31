#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT_JSON = ROOT / "outputs" / "demo_ready" / "predictions" / "session_top5.json"
DEFAULT_OUTPUT_JSON = ROOT / "outputs" / "demo_ready" / "predictions" / "session_top5_qwen_adjusted.json"
DEFAULT_RECIPE_JSON = ROOT / "data" / "egtea_gaze_plus" / "metadata" / "recipes.json"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


SYSTEM_PROMPT = """You refine egocentric cooking workflow predictions in streaming mode.

You must obey these rules:
- strict causal processing: only use clips up to current_idx
- fixed-lag refinement: only clips in the current adjustable range may be changed
- clips before locked_until are frozen and cannot be changed
- choose only from each clip's provided top-5 candidates
- prefer minimal corrections
- use recipe order and short-range temporal consistency
- return valid JSON only

Required output schema:
{
  "current_idx": 12,
  "adjustments": [
    {
      "clip_idx": 10,
      "selected_action_id": 12,
      "selected_action_label": "...",
      "selected_rank_from_top5": 2,
      "confidence_note": "keep_top1 | switched_to_rank2 | switched_to_rank3 | uncertain",
      "reason": "short explanation"
    }
  ],
  "step_summary": "short summary"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use local Qwen2.5-3B via Ollama to refine predictions with streaming fixed-lag logic.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--recipe-json", type=Path, default=DEFAULT_RECIPE_JSON)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--recipe-name", type=str, default=None, help="Optional explicit recipe name hint.")
    parser.add_argument("--lag", type=int, default=3, help="Only the most recent lag clips may be revised.")
    parser.add_argument("--context-clips", type=int, default=16, help="Maximum number of past/current clips shown to Qwen.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional limit on how many streaming steps to process.")
    parser.add_argument("--max-gap-to-adjust", type=float, default=0.20, help="Only allow adjustments when top1-top2 gap is below this threshold.")
    parser.add_argument("--max-rank-to-adjust", type=int, default=2, help="Only allow switching to candidates up to this top-k rank.")
    parser.add_argument("--only-adjust-on-ambiguity", action="store_true", help="Only invoke Qwen when the current clip is ambiguous under the gap threshold.")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer_recipe_name(session_folder: str, recipe_payload: list[dict]) -> str | None:
    session_lower = session_folder.lower()
    for recipe in recipe_payload:
        title = str(recipe.get("title", "")).strip()
        if not title:
            continue
        normalized = (
            title.lower()
            .replace("(", "")
            .replace(")", "")
            .replace("/", " ")
            .replace("-", " ")
        )
        tokens = [token for token in normalized.split() if token]
        if tokens and any(token in session_lower for token in tokens):
            return title
    return None


def recipe_to_prompt_block(recipe_name: str | None, recipe_payload: list[dict]) -> dict:
    if recipe_name is None:
        return {"recipe_name": None, "steps": []}
    for recipe in recipe_payload:
        if str(recipe.get("title", "")).strip() == recipe_name:
            sections = recipe.get("sections", [])
            flattened_steps: list[str] = []
            for section in sections:
                for item in section.get("items", []) or []:
                    flattened_steps.append(str(item))
                for subsection in section.get("subsections", []) or []:
                    for item in subsection.get("items", []) or []:
                        flattened_steps.append(str(item))
            return {"recipe_name": recipe_name, "steps": flattened_steps}
    return {"recipe_name": recipe_name, "steps": []}


def normalize_row(row: dict) -> dict:
    clip_name = row.get("clip", row.get("clip_name"))
    pred_label = row.get("pred_action_label", row.get("predicted_action"))
    pred_id = row.get("pred_action_id", row.get("predicted_action_id"))
    top5 = row.get("top5", [])
    top1_top2_gap = None
    if len(top5) >= 2:
        top1_top2_gap = float(top5[0]["probability"]) - float(top5[1]["probability"])
    return {
        "clip_idx": row.get("clip_idx"),
        "clip": clip_name,
        "clip_duration_sec": row.get("clip_duration_sec"),
        "pred_action_id": pred_id,
        "pred_action_label": pred_label,
        "pred_state_name": row.get("pred_state_name", row.get("predicted_state")),
        "top5": top5,
        "top1_top2_gap": top1_top2_gap,
    }


def build_streaming_prompt(
    session_folder: str,
    recipe_block: dict,
    normalized_rows: list[dict],
    selected_history: dict[int, dict],
    current_idx: int,
    lag: int,
    context_clips: int,
) -> str:
    window_start = max(0, current_idx - context_clips + 1)
    adjustable_start = max(0, current_idx - lag + 1)
    locked_until = adjustable_start - 1
    prompt_rows: list[dict] = []
    for row in normalized_rows[window_start : current_idx + 1]:
        idx = int(row["clip_idx"])
        selected = selected_history[idx]
        base = {
            "clip_idx": idx,
            "clip": row["clip"],
            "clip_duration_sec": row["clip_duration_sec"],
            "current_selected_action_id": selected["selected_action_id"],
            "current_selected_action_label": selected["selected_action_label"],
            "current_selected_rank_from_top5": selected["selected_rank_from_top5"],
            "pred_state_name": row["pred_state_name"],
            "status": "locked" if idx <= locked_until else "adjustable",
        }
        if idx <= locked_until:
            base["top5"] = []
        else:
            base["top5"] = row["top5"]
        prompt_rows.append(base)
    payload = {
        "session_folder": session_folder,
        "recipe_hint": recipe_block["recipe_name"],
        "recipe_steps": recipe_block["steps"],
        "current_idx": current_idx,
        "locked_until": locked_until,
        "adjustable_range": [adjustable_start, current_idx],
        "clips": prompt_rows,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def call_ollama(model: str, ollama_url: str, system_prompt: str, user_prompt: str) -> dict:
    request_payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nUser payload:\n{user_prompt}",
        "stream": True,
        "format": "json",
    }
    req = urllib.request.Request(
        url=ollama_url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            response_parts: list[str] = []
            with tqdm(desc="Qwen decoding", unit="chunk", leave=False) as pbar:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        response_parts.append(str(chunk["response"]))
                    pbar.update(1)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Failed to reach local Ollama. Start Ollama first and run: ollama pull qwen2.5:3b-instruct"
        ) from exc
    text = "".join(response_parts).strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return json.loads(text)


def build_default_selection(row: dict) -> dict:
    top5 = row.get("top5", [])
    if top5:
        top1 = top5[0]
        return {
            "selected_action_id": int(top1["action_id"]),
            "selected_action_label": top1["action_label"],
            "selected_rank_from_top5": 1,
            "confidence_note": "keep_top1",
            "reason": "default_top1",
        }
    return {
        "selected_action_id": int(row["pred_action_id"]),
        "selected_action_label": row["pred_action_label"],
        "selected_rank_from_top5": 1,
        "confidence_note": "keep_top1",
        "reason": "default_prediction",
    }


def apply_adjustments(
    selected_history: dict[int, dict],
    normalized_rows: list[dict],
    current_idx: int,
    lag: int,
    llm_payload: dict,
    max_gap_to_adjust: float,
    max_rank_to_adjust: int,
) -> list[dict]:
    adjustable_start = max(0, current_idx - lag + 1)
    refined_clip_indices: list[dict] = []
    rows_by_idx = {int(row["clip_idx"]): row for row in normalized_rows}
    for item in llm_payload.get("adjustments", []):
        idx = int(item.get("clip_idx"))
        if idx < adjustable_start or idx > current_idx:
            continue
        row = rows_by_idx[idx]
        gap = row.get("top1_top2_gap")
        if gap is not None and float(gap) > float(max_gap_to_adjust):
            continue
        valid_by_rank = {int(candidate["rank"]): candidate for candidate in row.get("top5", [])}
        valid_by_id = {int(candidate["action_id"]): candidate for candidate in row.get("top5", [])}
        candidate = None
        rank = item.get("selected_rank_from_top5")
        if rank is not None and int(rank) in valid_by_rank:
            candidate = valid_by_rank[int(rank)]
        elif item.get("selected_action_id") is not None and int(item["selected_action_id"]) in valid_by_id:
            candidate = valid_by_id[int(item["selected_action_id"])]
            rank = int(candidate["rank"])
        if candidate is None:
            continue
        if int(candidate["rank"]) > int(max_rank_to_adjust):
            continue
        selected_history[idx] = {
            "selected_action_id": int(candidate["action_id"]),
            "selected_action_label": candidate["action_label"],
            "selected_rank_from_top5": int(candidate["rank"]),
            "confidence_note": str(item.get("confidence_note", "uncertain")),
            "reason": str(item.get("reason", "")),
        }
        refined_clip_indices.append(
            {
                "clip_idx": idx,
                "selected_action_id": int(candidate["action_id"]),
                "selected_action_label": candidate["action_label"],
                "selected_rank_from_top5": int(candidate["rank"]),
            }
        )
    return refined_clip_indices


def main() -> int:
    args = parse_args()
    if not args.input_json.exists():
        raise FileNotFoundError(f"Prediction json not found: {args.input_json}")
    if not args.recipe_json.exists():
        raise FileNotFoundError(f"Recipe json not found: {args.recipe_json}")

    prediction_payload = load_json(args.input_json)
    recipe_payload = load_json(args.recipe_json)
    recipe_name = args.recipe_name or infer_recipe_name(str(prediction_payload.get("session_folder", "")), recipe_payload)
    recipe_block = recipe_to_prompt_block(recipe_name, recipe_payload)

    normalized_rows = [normalize_row(row) for row in prediction_payload.get("results", [])]
    if args.max_steps is not None:
        normalized_rows = normalized_rows[: int(args.max_steps)]

    selected_history = {int(row["clip_idx"]): build_default_selection(row) for row in normalized_rows}
    online_trace: list[dict] = []

    with tqdm(total=len(normalized_rows), desc="Streaming Qwen refine", unit="clip") as progress:
        for current_idx in range(len(normalized_rows)):
            current_row = normalized_rows[current_idx]
            current_gap = current_row.get("top1_top2_gap")
            should_query_qwen = True
            if args.only_adjust_on_ambiguity:
                should_query_qwen = current_gap is not None and float(current_gap) <= float(args.max_gap_to_adjust)

            if should_query_qwen:
                user_prompt = build_streaming_prompt(
                    session_folder=str(prediction_payload.get("session_folder")),
                    recipe_block=recipe_block,
                    normalized_rows=normalized_rows,
                    selected_history=selected_history,
                    current_idx=current_idx,
                    lag=int(args.lag),
                    context_clips=int(args.context_clips),
                )
                llm_payload = call_ollama(
                    model=args.model,
                    ollama_url=args.ollama_url,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                refined_recent = apply_adjustments(
                    selected_history=selected_history,
                    normalized_rows=normalized_rows,
                    current_idx=current_idx,
                    lag=int(args.lag),
                    llm_payload=llm_payload,
                    max_gap_to_adjust=float(args.max_gap_to_adjust),
                    max_rank_to_adjust=int(args.max_rank_to_adjust),
                )
                step_summary = llm_payload.get("step_summary", "")
            else:
                refined_recent = []
                step_summary = "skipped_qwen_due_to_confidence_gate"
            locked_until = max(-1, current_idx - int(args.lag))
            current_selection = selected_history[current_idx]
            online_trace.append(
                {
                    "current_idx": current_idx,
                    "clip": normalized_rows[current_idx]["clip"],
                    "recipe_name": recipe_name,
                    "locked_until": locked_until,
                    "adjustable_range": [max(0, current_idx - int(args.lag) + 1), current_idx],
                    "current_selected_action_id": current_selection["selected_action_id"],
                    "current_selected_action_label": current_selection["selected_action_label"],
                    "current_selected_rank_from_top5": current_selection["selected_rank_from_top5"],
                    "current_confidence_note": current_selection["confidence_note"],
                    "current_reason": current_selection["reason"],
                    "current_top1_top2_gap": current_gap,
                    "qwen_invoked": should_query_qwen,
                    "refined_recent_clips": refined_recent,
                    "step_summary": step_summary,
                }
            )
            progress.set_postfix_str(
                f"idx={current_idx} locked<= {locked_until} current={current_selection['selected_action_label'][:24]}"
            )
            progress.update(1)

    final_results = []
    for row in normalized_rows:
        idx = int(row["clip_idx"])
        final_results.append(
            {
                **row,
                **selected_history[idx],
            }
        )

    final_payload = {
        "backend": "ollama",
        "mode": "streaming_fixed_lag",
        "model": args.model,
        "lag": int(args.lag),
        "context_clips": int(args.context_clips),
        "max_gap_to_adjust": float(args.max_gap_to_adjust),
        "max_rank_to_adjust": int(args.max_rank_to_adjust),
        "only_adjust_on_ambiguity": bool(args.only_adjust_on_ambiguity),
        "input_json": str(args.input_json),
        "recipe_json": str(args.recipe_json),
        "recipe_name": recipe_name,
        "raw_predictions": prediction_payload,
        "final_results": final_results,
        "online_trace": online_trace,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "recipe_name": recipe_name,
                "num_steps": len(online_trace),
                "backend": "ollama",
                "model": args.model,
                "mode": "streaming_fixed_lag",
                "lag": int(args.lag),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
