#!/usr/bin/env python3
"""Rank evaluated models by ReID-useful attribute quality.

This script scans successful eval run logs, picks one representative run per
model, computes a weighted ReID-oriented score, and saves the results as both
JSON and Markdown tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


REID_USEFUL_WEIGHTS: Dict[str, float] = {
    "backpack": 1.5,
    "bag": 1.3,
    "handbag": 1.3,
    "clothes": 1.2,
    "up": 1.1,
    "down": 1.0,
    "hat": 1.0,
    "up_color": 0.8,
    "down_color": 0.8,
    "gender": 0.7,
    "hair": 0.6,
    "age": 0.0,
    "age_balanced": 0.0,
}

DISPLAY_FIELDS = [
    "backpack",
    "bag",
    "handbag",
    "clothes",
    "up",
    "down",
    "hat",
    "up_color",
    "down_color",
    "gender",
    "hair",
]

BALANCED_MANIFEST_NAME = "market1501_age_balanced_seed1501_manifest.json"


def load_run_log(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_field(run_log: Dict[str, Any], key: str) -> Any:
    return run_log.get("gallery", {}).get("selection", {}).get(key)


def is_comparable_balanced_run(run_log: Dict[str, Any]) -> bool:
    sample_manifest = _selection_field(run_log, "sample_manifest")
    write_sample_manifest = _selection_field(run_log, "write_sample_manifest")
    stratify_by = _selection_field(run_log, "stratify_by")
    return (
        (isinstance(sample_manifest, str) and sample_manifest.endswith(BALANCED_MANIFEST_NAME))
        or (isinstance(write_sample_manifest, str) and write_sample_manifest.endswith(BALANCED_MANIFEST_NAME))
        or stratify_by == "age"
    )


def representative_run_sort_key(run_log: Dict[str, Any]) -> tuple:
    gallery_size = int(run_log.get("gallery", {}).get("num_items", 0))
    finished_at = str(run_log.get("finished_at_utc", ""))
    preferred_comparable = is_comparable_balanced_run(run_log) and gallery_size >= 100
    return (
        1 if preferred_comparable else 0,
        gallery_size,
        1 if is_comparable_balanced_run(run_log) else 0,
        finished_at,
    )


def choose_representative_runs(eval_runs_dir: Path) -> List[Dict[str, Any]]:
    best_by_model: Dict[str, Dict[str, Any]] = {}
    for path in sorted(eval_runs_dir.glob("*/run_log_success.json")):
        run_log = load_run_log(path)
        model = run_log.get("model")
        if not isinstance(model, str):
            continue
        run_log["_run_log_path"] = str(path.resolve())
        run_log["_artifact_dir"] = str(path.parent.resolve())
        existing = best_by_model.get(model)
        if existing is None or representative_run_sort_key(run_log) > representative_run_sort_key(existing):
            best_by_model[model] = run_log
    return list(best_by_model.values())


def compute_reid_useful_score(metrics: Dict[str, Any]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for field, weight in REID_USEFUL_WEIGHTS.items():
        if weight <= 0 or field not in metrics:
            continue
        weighted_sum += float(metrics[field]) * weight
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("No positive ReID-useful weights were applied.")
    return weighted_sum / total_weight


def build_rows(run_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_log in run_logs:
        metrics = run_log.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        row = {
            "model": run_log["model"],
            "score_reid_useful": compute_reid_useful_score(metrics),
            "run_log_path": run_log["_run_log_path"],
            "artifact_dir": run_log["_artifact_dir"],
            "comparable_balanced_run": is_comparable_balanced_run(run_log),
            "gallery_size": int(run_log.get("gallery", {}).get("num_items", 0)),
            "metrics": {field: metrics.get(field) for field in DISPLAY_FIELDS},
        }
        rows.append(row)
    rows.sort(key=lambda item: (-item["score_reid_useful"], item["model"]))
    for index, row in enumerate(rows, 1):
        row["rank_reid_useful"] = index
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Rank",
        "Model",
        "Score",
        "backpack",
        "bag",
        "handbag",
        "clothes",
        "up",
        "down",
        "hat",
        "up_color",
        "down_color",
        "gender",
        "hair",
    ]
    lines = [
        "# ReID-Useful Model Ranking",
        "",
        "Representative run per model, preferring the balanced-manifest runs.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        metrics = row["metrics"]
        values = [
            str(row["rank_reid_useful"]),
            f"`{row['model']}`",
            f"{row['score_reid_useful']:.6f}",
        ]
        values.extend(f"{float(metrics[field]):.6f}" for field in DISPLAY_FIELDS)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Run Sources",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['model']}` -> `{row['run_log_path']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    eval_runs_dir = repo_root / "eval_runs"
    output_dir = repo_root / "sample_outputs"

    rows = build_rows(choose_representative_runs(eval_runs_dir))
    payload = {
        "weights": REID_USEFUL_WEIGHTS,
        "rows": rows,
    }
    json_path = output_dir / "reid_useful_ranking.json"
    markdown_path = output_dir / "reid_useful_ranking.md"
    write_json(json_path, payload)
    markdown_path.write_text(build_markdown(rows), encoding="utf-8")

    print(f"Saved JSON ranking to: {json_path}")
    print(f"Saved Markdown ranking to: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
