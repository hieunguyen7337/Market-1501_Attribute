#!/usr/bin/env python3
"""Run cross-model prompt tuning sweeps for the Market-1501 evaluator."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from eval_qwen3_vl_openrouter import (
    GalleryItem,
    load_market_attribute,
    load_prompt,
    list_gallery_items,
    slugify_name,
    write_sample_manifest,
)
from rank_reid_useful_models import compute_reid_useful_score


DEFAULT_GATE_MODELS = [
    "google/gemma-4-31b-it",
    "qwen/qwen3.5-9b",
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3.5-35b-a3b",
]
DEFAULT_FULL_MANIFEST = "sample_outputs/market1501_age_balanced_seed1501_manifest.json"
DEFAULT_PILOT_MANIFEST = "sample_outputs/market1501_age_balanced30_seed1501_manifest.json"
DEFAULT_PROMPT_ORDER = [
    "baseline.txt",
    "reid_focus.txt",
    "color_disambiguation.txt",
    "reid_focus_plus_color.txt",
]
PRIMARY_MODEL = "google/gemma-4-31b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a prompt tuning sweep across the main ReID models.")
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--prompt-files", nargs="*", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=DEFAULT_GATE_MODELS)
    parser.add_argument("--full-manifest", type=Path, default=Path(DEFAULT_FULL_MANIFEST))
    parser.add_argument("--pilot-manifest", type=Path, default=Path(DEFAULT_PILOT_MANIFEST))
    parser.add_argument("--pilot-limit", type=int, default=30)
    parser.add_argument("--pilot-seed", type=int, default=1501)
    parser.add_argument("--stage", choices=["pilot", "final", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--average-regression-tolerance", type=float, default=0.005)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def resolve_prompt_files(repo_root: Path, prompt_files: Sequence[Path] | None) -> List[Path]:
    if prompt_files:
        resolved = [path.resolve() if path.is_absolute() else (repo_root / path).resolve() for path in prompt_files]
    else:
        prompts_dir = repo_root / "prompts"
        resolved = [(prompts_dir / name).resolve() for name in DEFAULT_PROMPT_ORDER]
    baseline_path = (repo_root / "prompts" / "baseline.txt").resolve()
    if baseline_path not in resolved:
        resolved = [baseline_path, *resolved]
    deduped: List[Path] = []
    seen = set()
    for path in resolved:
        if path in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        deduped.append(path)
        seen.add(path)
    return deduped


def build_age_balanced_pilot_items(
    gallery_items: Sequence[GalleryItem],
    *,
    test_split: Any,
    limit: int,
    seed: int,
) -> List[GalleryItem]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    attr_values = getattr(test_split, "age", None)
    if attr_values is None:
        raise ValueError("test_split.age is required for age-balanced pilot selection")

    by_class: Dict[int, List[GalleryItem]] = {}
    for item in gallery_items:
        val = int(attr_values[item.class_index])
        by_class.setdefault(val, []).append(item)

    if not by_class:
        raise ValueError("No classes found for age-balanced pilot selection")

    rng = random.Random(seed)
    remaining_by_class: Dict[int, List[GalleryItem]] = {}
    for value, items in sorted(by_class.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        remaining_by_class[value] = shuffled

    target = min(limit, sum(len(items) for items in remaining_by_class.values()))
    class_values = sorted(remaining_by_class)
    base_quota = target // len(class_values)
    selected: List[GalleryItem] = []

    for value in class_values:
        take = min(base_quota, len(remaining_by_class[value]))
        selected.extend(remaining_by_class[value][:take])
        remaining_by_class[value] = remaining_by_class[value][take:]

    while len(selected) < target:
        progressed = False
        for value in class_values:
            if len(selected) >= target:
                break
            leftovers = remaining_by_class[value]
            if not leftovers:
                continue
            selected.append(leftovers.pop(0))
            progressed = True
        if not progressed:
            break

    rng.shuffle(selected)
    return selected


def ensure_pilot_manifest(dataset_root: Path, images_dir: Path, manifest_path: Path, limit: int, seed: int) -> Path:
    if manifest_path.exists():
        return manifest_path.resolve()
    market = load_market_attribute((dataset_root / "market_attribute.mat").resolve())
    all_gallery_items = list_gallery_items(images_dir.resolve())
    pilot_items = build_age_balanced_pilot_items(
        all_gallery_items,
        test_split=market.test,
        limit=limit,
        seed=seed,
    )
    write_sample_manifest(manifest_path.resolve(), pilot_items)
    return manifest_path.resolve()


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def prompt_variant_info(prompt_path: Path) -> Dict[str, Any]:
    prompt_text, prompt_meta = load_prompt(prompt_path)
    return {
        "path": str(prompt_path.resolve()),
        "name": prompt_meta["name"],
        "slug": prompt_meta["slug"],
        "sha256": prompt_meta["sha256"],
        "text": prompt_text,
    }


def run_eval(
    repo_root: Path,
    dataset_root: Path,
    images_dir: Path,
    stage_dir: Path,
    manifest_path: Path,
    limit: int,
    model: str,
    prompt_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt_info = prompt_variant_info(prompt_path)
    prompt_slug = prompt_info["slug"]
    outputs_dir = stage_dir / "outputs" / prompt_slug
    outputs_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_dir = stage_dir / "raw_runs"
    raw_runs_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in raw_runs_dir.iterdir() if path.is_dir()}
    command = [
        sys.executable,
        str((repo_root / "eval_qwen3_vl_openrouter.py").resolve()),
        "--dataset-root",
        str(dataset_root.resolve()),
        "--images-dir",
        str(images_dir.resolve()),
        "--limit",
        str(limit),
        "--model",
        model,
        "--prompt-file",
        str(prompt_path.resolve()),
        "--max-tokens",
        str(args.max_tokens),
        "--sleep-seconds",
        str(args.sleep_seconds),
        "--output-mat",
        str((outputs_dir / f"{model_slug(model)}_gallery.mat").resolve()),
        "--metrics-json",
        str((outputs_dir / f"{model_slug(model)}_metrics.json").resolve()),
        "--sample-manifest",
        str(manifest_path.resolve()),
        "--run-log-dir",
        str(raw_runs_dir.resolve()),
        "--timeout",
        str(args.timeout),
        "--request-retries",
        str(args.request_retries),
        "--temperature",
        str(args.temperature),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=repo_root)
    after_dirs = [path for path in raw_runs_dir.iterdir() if path.is_dir() and path.name not in before]
    if not after_dirs:
        raise RuntimeError(
            f"No run artifact directory created for model={model}, prompt={prompt_path.name}.\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    run_dir = max(after_dirs, key=lambda path: path.stat().st_mtime)
    run_logs = sorted(run_dir.glob("run_log_*.json"))
    if not run_logs:
        raise RuntimeError(f"No run log found in {run_dir}")
    run_log_path = max(run_logs, key=lambda path: path.stat().st_mtime)
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    metrics = run_log.get("metrics", {})
    prediction_failures = run_log.get("prediction_failures", {})
    total_failures = int(prediction_failures.get("invalid_format_count", 0)) + int(
        prediction_failures.get("request_failure_count", 0)
    )
    return {
        "model": model,
        "prompt": {k: v for k, v in prompt_info.items() if k != "text"},
        "status": run_log.get("status", "unknown"),
        "metrics": metrics,
        "reid_useful_score": compute_reid_useful_score(metrics) if metrics else None,
        "average": metrics.get("average") if metrics else None,
        "invalid_format_count": int(prediction_failures.get("invalid_format_count", 0)),
        "request_failure_count": int(prediction_failures.get("request_failure_count", 0)),
        "total_failure_count": total_failures,
        "run_log_path": str(run_log_path.resolve()),
        "artifact_dir": str(run_dir.resolve()),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def build_stage_results(
    repo_root: Path,
    dataset_root: Path,
    images_dir: Path,
    stage_dir: Path,
    manifest_path: Path,
    limit: int,
    prompt_files: Sequence[Path],
    models: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt_infos = [prompt_variant_info(path) for path in prompt_files]
    results: Dict[str, Dict[str, Any]] = {}
    for prompt_info in prompt_infos:
        prompt_path = Path(prompt_info["path"])
        per_model: Dict[str, Any] = {}
        print(f"[stage] {stage_dir.name}: {prompt_info['slug']}")
        for model in models:
            print(f"  [run] {model}")
            per_model[model] = run_eval(
                repo_root=repo_root,
                dataset_root=dataset_root,
                images_dir=images_dir,
                stage_dir=stage_dir,
                manifest_path=manifest_path,
                limit=limit,
                model=model,
                prompt_path=prompt_path,
                args=args,
            )
        results[prompt_info["slug"]] = {
            "prompt": {k: v for k, v in prompt_info.items() if k != "text"},
            "per_model": per_model,
        }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "limit": limit,
        "results": results,
    }


def compare_to_baseline(stage_results: Dict[str, Any], average_tolerance: float, final_gate: bool) -> Dict[str, Any]:
    results = stage_results["results"]
    baseline = results["baseline"]
    baseline_models = baseline["per_model"]
    comparisons: Dict[str, Any] = {}
    for prompt_slug, prompt_result in results.items():
        if prompt_slug == "baseline":
            comparisons[prompt_slug] = {
                "decision": "baseline_reference",
                "per_model": {},
            }
            continue
        per_model_deltas: Dict[str, Any] = {}
        for model, result in prompt_result["per_model"].items():
            baseline_result = baseline_models[model]
            per_model_deltas[model] = {
                "reid_useful_score": result["reid_useful_score"],
                "baseline_reid_useful_score": baseline_result["reid_useful_score"],
                "reid_useful_delta": result["reid_useful_score"] - baseline_result["reid_useful_score"],
                "average": result["average"],
                "baseline_average": baseline_result["average"],
                "average_delta": float(result["average"]) - float(baseline_result["average"]),
                "total_failure_count": result["total_failure_count"],
                "baseline_total_failure_count": baseline_result["total_failure_count"],
                "failure_delta": result["total_failure_count"] - baseline_result["total_failure_count"],
            }

        decision = decide_prompt_variant(per_model_deltas, average_tolerance, final_gate)
        comparisons[prompt_slug] = {
            "decision": decision,
            "per_model": per_model_deltas,
        }
    return comparisons


def decide_prompt_variant(
    per_model_deltas: Dict[str, Dict[str, Any]],
    average_tolerance: float,
    final_gate: bool,
) -> str:
    gemma_delta = float(per_model_deltas[PRIMARY_MODEL]["reid_useful_delta"])
    if gemma_delta <= 0:
        return "rejected_gemma_no_gain"

    anchor_models = [model for model in per_model_deltas if model != PRIMARY_MODEL]
    negative_anchor_count = sum(float(per_model_deltas[model]["reid_useful_delta"]) <= 0 for model in anchor_models)
    if final_gate:
        if negative_anchor_count > 0:
            return "rejected_anchor_no_gain"
        if any(float(item["average_delta"]) < -average_tolerance for item in per_model_deltas.values()):
            return "rejected_overall_regression"
        if any(int(item["failure_delta"]) > 0 for item in per_model_deltas.values()):
            return "rejected_format_regression"
        return "accepted"

    if negative_anchor_count >= 2:
        return "rejected_anchor_no_gain"
    if int(per_model_deltas[PRIMARY_MODEL]["failure_delta"]) > 0:
        return "rejected_format_regression"
    return "screen_passed"


def build_markdown(
    stage_name: str,
    stage_results: Dict[str, Any],
    comparisons: Dict[str, Any],
) -> str:
    model_order = list(next(iter(stage_results["results"].values()))["per_model"].keys())
    delta_models = [model for model in model_order if model != PRIMARY_MODEL]
    headers = ["Prompt", "Decision", f"{PRIMARY_MODEL} delta", *[f"{model} delta" for model in delta_models]]
    lines = [
        f"## {stage_name.title()}",
        "",
        f"Manifest: `{stage_results['manifest_path']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for prompt_slug, comparison in comparisons.items():
        if prompt_slug == "baseline":
            lines.append("| " + " | ".join(["baseline", "baseline_reference", *["-"] * (len(headers) - 2)]) + " |")
            continue
        per_model = comparison["per_model"]
        row = [
            prompt_slug,
            comparison["decision"],
            f"{per_model[PRIMARY_MODEL]['reid_useful_delta']:.6f}",
        ]
        row.extend(f"{per_model[model]['reid_useful_delta']:.6f}" for model in delta_models)
        lines.append(
            "| " + " | ".join(row) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    dataset_root = (repo_root / args.dataset_root).resolve() if not args.dataset_root.is_absolute() else args.dataset_root.resolve()
    images_dir = (
        args.images_dir.resolve()
        if args.images_dir is not None
        else (dataset_root / "data" / "Market-1501-v15.09.15" / "bounding_box_test").resolve()
    )
    prompt_files = resolve_prompt_files(repo_root, args.prompt_files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / "sample_outputs" / f"prompt_tuning_sweep_{timestamp}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "settings": {
            "models": list(args.models),
            "average_regression_tolerance": args.average_regression_tolerance,
            "stage": args.stage,
            "prompt_files": [str(path) for path in prompt_files],
            "dataset_root": str(dataset_root),
            "images_dir": str(images_dir),
        }
    }
    markdown_sections = ["# Prompt Tuning Sweep", ""]
    surviving_prompt_files = list(prompt_files)

    if args.stage in {"pilot", "both"}:
        pilot_manifest = ensure_pilot_manifest(
            dataset_root=dataset_root,
            images_dir=images_dir,
            manifest_path=(repo_root / args.pilot_manifest).resolve()
            if not args.pilot_manifest.is_absolute()
            else args.pilot_manifest.resolve(),
            limit=args.pilot_limit,
            seed=args.pilot_seed,
        )
        pilot_stage_dir = output_dir / "pilot"
        pilot_results = build_stage_results(
            repo_root=repo_root,
            dataset_root=dataset_root,
            images_dir=images_dir,
            stage_dir=pilot_stage_dir,
            manifest_path=pilot_manifest,
            limit=args.pilot_limit,
            prompt_files=prompt_files,
            models=args.models,
            args=args,
        )
        pilot_comparisons = compare_to_baseline(
            pilot_results,
            average_tolerance=args.average_regression_tolerance,
            final_gate=False,
        )
        payload["pilot"] = {"results": pilot_results, "comparisons": pilot_comparisons}
        markdown_sections.append(build_markdown("pilot", pilot_results, pilot_comparisons))
        surviving_prompt_files = []
        for prompt_path in prompt_files:
            prompt_slug = prompt_variant_info(prompt_path)["slug"]
            decision = pilot_comparisons[prompt_slug]["decision"]
            if prompt_slug == "baseline" or decision == "screen_passed":
                surviving_prompt_files.append(prompt_path)

    if args.stage in {"final", "both"}:
        full_manifest = (
            (repo_root / args.full_manifest).resolve()
            if not args.full_manifest.is_absolute()
            else args.full_manifest.resolve()
        )
        final_prompt_files = surviving_prompt_files if args.stage == "both" else prompt_files
        final_stage_dir = output_dir / "final"
        final_results = build_stage_results(
            repo_root=repo_root,
            dataset_root=dataset_root,
            images_dir=images_dir,
            stage_dir=final_stage_dir,
            manifest_path=full_manifest,
            limit=120,
            prompt_files=final_prompt_files,
            models=args.models,
            args=args,
        )
        final_comparisons = compare_to_baseline(
            final_results,
            average_tolerance=args.average_regression_tolerance,
            final_gate=True,
        )
        payload["final"] = {"results": final_results, "comparisons": final_comparisons}
        markdown_sections.append(build_markdown("final", final_results, final_comparisons))

    json_path = output_dir / "prompt_tuning_summary.json"
    markdown_path = output_dir / "prompt_tuning_summary.md"
    write_json(json_path, payload)
    markdown_path.write_text("\n".join(markdown_sections).strip() + "\n", encoding="utf-8")
    print(f"Saved tuning summary JSON to: {json_path}")
    print(f"Saved tuning summary Markdown to: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
