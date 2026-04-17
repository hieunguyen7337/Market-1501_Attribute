#!/usr/bin/env python3
"""Evaluate Market-1501 attributes with Qwen3-VL through OpenRouter.

This script mirrors the official MATLAB evaluation flow:
1. Load identity-level annotations from ``market_attribute.mat``.
2. Expand test annotations to image-level labels using the sorted gallery set.
3. Query an OpenRouter vision-language model for each gallery image.
4. Save predictions to ``gallery_market.mat``-compatible format.
5. Compute the same 12 attribute accuracies as ``evaluate_market_attribute.m``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests
import scipy.io as sio


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-vl-32b-instruct"
PREDICTION_COLUMNS = [
    "gender",
    "hair",
    "age",
    "up",
    "down",
    "clothes",
    "hat",
    "backpack",
    "bag",
    "handbag",
    "up_color",
    "down_color",
]


PROMPT = """Return only a compact JSON object with integer fields:
{
  "gender": 1|2,
  "hair": 1|2,
  "age": 1|2|3|4,
  "up": 1|2,
  "down": 1|2,
  "clothes": 1|2,
  "hat": 1|2,
  "backpack": 1|2,
  "bag": 1|2,
  "handbag": 1|2,
  "up_color": 1|2|3|4|5|6|7|8,
  "down_color": 1|2|3|4|5|6|7|8|9
}

You are labeling one pedestrian image for the Market-1501 attribute benchmark.
Use these exact dataset labels:
- gender: male=1, female=2
- hair: short hair=1, long hair=2
- up: long sleeve=1, short sleeve=2
- down: long lower-body clothing=1, short=2
- clothes: dress=1, pants=2
- hat: no=1, yes=2
- backpack: no=1, yes=2
- bag: no=1, yes=2
- handbag: no=1, yes=2
- age: young=1, teenager=2, adult=3, old=4
- up_color:
  1=black, 2=white, 3=red, 4=purple, 5=gray, 6=blue, 7=green, 8=yellow
- down_color:
  1=black, 2=white, 3=pink, 4=gray, 5=blue, 6=green, 7=brown, 8=yellow, 9=purple

Rules:
- Pick exactly one value for every field.
- If uncertain, still choose the single best label.
- Do not include markdown fences or any explanation.
"""


class EvalError(RuntimeError):
    """Raised when evaluation data or model output is invalid."""


@dataclass(frozen=True)
class GalleryItem:
    image_path: Path
    pid: int
    class_index: int


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Market-1501 attribute evaluation with Qwen3-VL via OpenRouter."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Directory containing market_attribute.mat and gallery images.",
    )
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=None,
        help="Path to market_attribute.mat. Defaults to <dataset-root>/market_attribute.mat.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Path to bounding_box_test. Defaults to <dataset-root>/bounding_box_test.",
    )
    parser.add_argument(
        "--output-mat",
        type=Path,
        default=Path("gallery_market.mat"),
        help="Output MAT file path for the 12-column gallery prediction matrix.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("market_attribute_metrics.json"),
        help="Optional JSON file to save per-attribute metrics.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".openrouter_cache"),
        help="Directory for per-image JSON predictions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--http-referer",
        default=os.environ.get("OPENROUTER_HTTP_REFERER"),
        help="Optional HTTP-Referer header for OpenRouter ranking/attribution.",
    )
    parser.add_argument(
        "--app-title",
        default=os.environ.get("OPENROUTER_APP_TITLE", "Market1501 Attribute Eval"),
        help="Optional X-Title header for OpenRouter attribution.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max output tokens per request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per image on transient API failures.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between successful requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N valid gallery images.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Ignore cached predictions and request everything again.",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Do not call OpenRouter. Only use cached predictions.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    dataset_root = args.dataset_root.resolve()
    mat_path = (args.mat_path or (dataset_root / "market_attribute.mat")).resolve()
    images_dir = (args.images_dir or (dataset_root / "bounding_box_test")).resolve()
    output_mat = args.output_mat.resolve()
    metrics_json = args.metrics_json.resolve()
    cache_dir = args.cache_dir.resolve()
    return mat_path, images_dir, output_mat, metrics_json, cache_dir


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise EvalError(f"{description} not found: {path}")


def load_market_attribute(mat_path: Path) -> Any:
    data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "market_attribute" not in data:
        raise EvalError(f"'market_attribute' key not found in {mat_path}")
    return data["market_attribute"]


def list_gallery_items(images_dir: Path, limit: int | None = None) -> List[GalleryItem]:
    files = sorted(images_dir.glob("*.jpg"))
    if not files:
        raise EvalError(f"No JPG files found in {images_dir}")

    items: List[GalleryItem] = []
    current_pid: int | None = None
    class_index = -1

    for image_path in files:
        stem = image_path.stem
        pid_text = stem.split("_", 1)[0]
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise EvalError(f"Unexpected Market-1501 filename: {image_path.name}") from exc

        if pid != current_pid:
            if pid > 0:
                class_index += 1
            current_pid = pid

        if pid > 0:
            items.append(GalleryItem(image_path=image_path, pid=pid, class_index=class_index))
            if limit is not None and len(items) >= limit:
                break

    if not items:
        raise EvalError(f"No valid positive-id gallery images found in {images_dir}")
    return items


def attr_array(split: Any, name: str) -> np.ndarray:
    values = np.asarray(getattr(split, name)).reshape(-1).astype(np.int32)
    return values


def build_ground_truth(test_split: Any, gallery_items: Sequence[GalleryItem]) -> Dict[str, np.ndarray]:
    field_names = [
        "gender",
        "hair",
        "age",
        "up",
        "down",
        "clothes",
        "hat",
        "backpack",
        "bag",
        "handbag",
        "upblack",
        "upwhite",
        "upred",
        "uppurple",
        "upgray",
        "upblue",
        "upgreen",
        "upyellow",
        "downblack",
        "downwhite",
        "downpink",
        "downgray",
        "downblue",
        "downgreen",
        "downbrown",
        "downyellow",
        "downpurple",
    ]

    identity_level = {name: attr_array(test_split, name) for name in field_names}
    max_class_index = max(item.class_index for item in gallery_items)
    num_identities = len(identity_level["gender"])
    if max_class_index >= num_identities:
        raise EvalError(
            f"Gallery expansion expected at most {num_identities} identities, "
            f"but saw class index {max_class_index}"
        )

    image_level: Dict[str, np.ndarray] = {}
    for name, values in identity_level.items():
        image_level[name] = np.asarray(
            [values[item.class_index] for item in gallery_items], dtype=np.int32
        )
    return image_level


def encode_image_to_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise EvalError("Model returned an empty response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise EvalError(f"Could not find JSON object in model response: {text!r}")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise EvalError(f"Invalid JSON in model response: {text!r}") from exc

    if not isinstance(parsed, dict):
        raise EvalError(f"Expected a JSON object, got: {type(parsed).__name__}")
    return parsed


def validate_prediction(raw: Dict[str, Any]) -> Dict[str, int]:
    ranges = {
        "gender": (1, 2),
        "hair": (1, 2),
        "age": (1, 4),
        "up": (1, 2),
        "down": (1, 2),
        "clothes": (1, 2),
        "hat": (1, 2),
        "backpack": (1, 2),
        "bag": (1, 2),
        "handbag": (1, 2),
        "up_color": (1, 8),
        "down_color": (1, 9),
    }

    pred: Dict[str, int] = {}
    for key, (low, high) in ranges.items():
        if key not in raw:
            raise EvalError(f"Missing key '{key}' in model response")
        try:
            value = int(raw[key])
        except (TypeError, ValueError) as exc:
            raise EvalError(f"Non-integer value for '{key}': {raw[key]!r}") from exc
        if value < low or value > high:
            raise EvalError(f"Value for '{key}' out of range [{low}, {high}]: {value}")
        pred[key] = value
    return pred


def request_prediction(
    image_path: Path,
    model: str,
    api_key: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    http_referer: str | None,
    app_title: str | None,
) -> Dict[str, int]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if app_title:
        headers["X-Title"] = app_title

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_to_data_url(image_path)},
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return validate_prediction(parse_json_object(content))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            backoff = min(30.0, 2 ** (attempt - 1))
            print(
                f"[warn] request failed for {image_path.name} (attempt {attempt}/{max_retries}): {exc}",
                file=sys.stderr,
            )
            time.sleep(backoff)

    raise EvalError(f"OpenRouter request failed for {image_path}: {last_error}")


def cache_path(cache_dir: Path, image_path: Path) -> Path:
    return cache_dir / f"{image_path.stem}.json"


def load_cached_prediction(path: Path) -> Dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvalError(f"Cache file is not a JSON object: {path}")
    return validate_prediction(data)


def save_cached_prediction(path: Path, prediction: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prediction, indent=2, sort_keys=True), encoding="utf-8")


def collect_predictions(
    gallery_items: Sequence[GalleryItem],
    args: argparse.Namespace,
    cache_dir: Path,
) -> List[Dict[str, int]]:
    predictions: List[Dict[str, int]] = []

    if not args.api_key and not args.skip_api:
        raise EvalError("OpenRouter API key missing. Pass --api-key or set OPENROUTER_API_KEY.")

    for index, item in enumerate(gallery_items, start=1):
        one_cache_path = cache_path(cache_dir, item.image_path)
        prediction: Dict[str, int] | None = None

        if one_cache_path.exists() and not args.overwrite_cache:
            prediction = load_cached_prediction(one_cache_path)
        elif not args.skip_api:
            prediction = request_prediction(
                image_path=item.image_path,
                model=args.model,
                api_key=args.api_key,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                http_referer=args.http_referer,
                app_title=args.app_title,
            )
            save_cached_prediction(one_cache_path, prediction)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        else:
            raise EvalError(
                f"Missing cache for {item.image_path.name} and --skip-api was set."
            )

        predictions.append(prediction)
        print(
            f"[{index}/{len(gallery_items)}] {item.image_path.name} -> "
            f"{json.dumps(prediction, sort_keys=True)}"
        )

    return predictions


def predictions_to_matrix(predictions: Sequence[Dict[str, int]]) -> np.ndarray:
    gallery = np.asarray(
        [[pred[col] for col in PREDICTION_COLUMNS] for pred in predictions],
        dtype=np.int32,
    )
    return gallery


def compute_metrics(gallery: np.ndarray, gt: Dict[str, np.ndarray]) -> Dict[str, float]:
    metrics = {
        "gender": float(np.mean(gallery[:, 0] == gt["gender"])),
        "age": float(np.mean(gallery[:, 2] == gt["age"])),
        "hair": float(np.mean(gallery[:, 1] == gt["hair"])),
        "up": float(np.mean(gallery[:, 3] == gt["up"])),
        "down": float(np.mean(gallery[:, 4] == gt["down"])),
        "clothes": float(np.mean(gallery[:, 5] == gt["clothes"])),
        "backpack": float(np.mean(gallery[:, 7] == gt["backpack"])),
        "handbag": float(np.mean(gallery[:, 9] == gt["handbag"])),
        "bag": float(np.mean(gallery[:, 8] == gt["bag"])),
        "hat": float(np.mean(gallery[:, 6] == gt["hat"])),
    }

    up_color_hits = np.zeros(len(gallery), dtype=bool)
    up_color_hits |= (gallery[:, 10] == 1) & (gt["upblack"] == 2)
    up_color_hits |= (gallery[:, 10] == 2) & (gt["upwhite"] == 2)
    up_color_hits |= (gallery[:, 10] == 3) & (gt["upred"] == 2)
    up_color_hits |= (gallery[:, 10] == 4) & (gt["uppurple"] == 2)
    up_color_hits |= (gallery[:, 10] == 5) & (gt["upgray"] == 2)
    up_color_hits |= (gallery[:, 10] == 6) & (gt["upblue"] == 2)
    up_color_hits |= (gallery[:, 10] == 7) & (gt["upgreen"] == 2)
    up_color_hits |= (gallery[:, 10] == 8) & (gt["upyellow"] == 2)

    down_color_hits = np.zeros(len(gallery), dtype=bool)
    down_color_hits |= (gallery[:, 11] == 1) & (gt["downblack"] == 2)
    down_color_hits |= (gallery[:, 11] == 2) & (gt["downwhite"] == 2)
    down_color_hits |= (gallery[:, 11] == 3) & (gt["downpink"] == 2)
    down_color_hits |= (gallery[:, 11] == 4) & (gt["downgray"] == 2)
    down_color_hits |= (gallery[:, 11] == 5) & (gt["downblue"] == 2)
    down_color_hits |= (gallery[:, 11] == 6) & (gt["downgreen"] == 2)
    down_color_hits |= (gallery[:, 11] == 7) & (gt["downbrown"] == 2)
    down_color_hits |= (gallery[:, 11] == 8) & (gt["downyellow"] == 2)
    down_color_hits |= (gallery[:, 11] == 9) & (gt["downpurple"] == 2)

    metrics["up_color"] = float(np.mean(up_color_hits))
    metrics["down_color"] = float(np.mean(down_color_hits))
    metrics["average"] = float(np.mean(list(metrics.values())))
    return metrics


def write_metrics(path: Path, metrics: Dict[str, float], gallery_size: int, model: str) -> None:
    payload = {
        "model": model,
        "gallery_size": gallery_size,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nMetrics")
    print("-------")
    ordered = [
        "gender",
        "age",
        "hair",
        "up",
        "down",
        "clothes",
        "backpack",
        "handbag",
        "bag",
        "hat",
        "up_color",
        "down_color",
        "average",
    ]
    for key in ordered:
        print(f"{key:>10}: {metrics[key]:.6f}")


def main() -> int:
    args = parse_args()

    try:
        dataset_root = args.dataset_root.resolve()
        load_dotenv(dataset_root / ".env")
        if args.api_key is None:
            args.api_key = os.environ.get("OPENROUTER_API_KEY")
        if args.http_referer is None:
            args.http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        if args.app_title == "Market1501 Attribute Eval":
            args.app_title = os.environ.get("OPENROUTER_APP_TITLE", args.app_title)

        mat_path, images_dir, output_mat, metrics_json, cache_dir = resolve_paths(args)
        require_path(mat_path, "Attribute MAT file")
        require_path(images_dir, "Gallery image directory")

        market = load_market_attribute(mat_path)
        gallery_items = list_gallery_items(images_dir, limit=args.limit)
        gt = build_ground_truth(market.test, gallery_items)

        predictions = collect_predictions(gallery_items, args, cache_dir)
        gallery = predictions_to_matrix(predictions)

        output_mat.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(str(output_mat), {"gallery": gallery})

        metrics = compute_metrics(gallery, gt)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        write_metrics(metrics_json, metrics, len(gallery_items), args.model)

        print(f"\nSaved gallery predictions to: {output_mat}")
        print(f"Saved metrics to: {metrics_json}")
        print_metrics(metrics)
        return 0
    except EvalError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
