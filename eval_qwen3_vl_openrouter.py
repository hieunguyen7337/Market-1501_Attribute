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
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests
import scipy.io as sio


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.5-9b"
INVALID_PREDICTION_VALUE = -1
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

AGE_LABELS = {
    1: "child",
    2: "teenager",
    3: "adult",
    4: "old",
}

UP_COLOR_LABELS = {
    1: "black",
    2: "white",
    3: "red",
    4: "purple",
    5: "gray",
    6: "blue",
    7: "green",
    8: "yellow",
}

DOWN_COLOR_LABELS = {
    1: "black",
    2: "white",
    3: "pink",
    4: "gray",
    5: "blue",
    6: "green",
    7: "brown",
    8: "yellow",
    9: "purple",
}

UP_COLOR_GT_FIELDS = [
    "upblack",
    "upwhite",
    "upred",
    "uppurple",
    "upgray",
    "upblue",
    "upgreen",
    "upyellow",
]

DOWN_COLOR_GT_FIELDS = [
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

# Heuristic remaps applied before strict label validation.
# Allows the model to output natural color/category names that aren't in the
# dataset's exact label set; a nearest-neighbor remap keeps the prediction
# instead of zeroing the whole row.
LABEL_REMAP: Dict[str, Dict[str, str]] = {
    "up_color":   {"pink": "red", "orange": "red", "brown": "gray"},
    "down_color": {"red": "pink", "orange": "brown"},
    "clothes":    {"shorts": "pants", "short": "pants", "skirt": "dress"},
}

EXPECTED_RESPONSE_KEYS = {
    "gender",
    "hair",
    "age",
    "clothing_type",
    "upper_body_clothes",
    "lower_body_clothes",
    "hat",
    "backpack",
    "bag",
    "handbag",
    "upper_body_clothes_color",
    "lower_body_clothes_color",
}


PROMPT = """Label every attribute of the main pedestrian in this image.
If multiple people are visible, focus on the most prominent (largest/most central) person.
If a body region is occluded or cropped, infer from whatever is visible.

Return only a raw JSON object with these exact keys and text values:

{
  "gender":                    "male" | "female",
  "hair":                      "short" | "long",
  "age":                       "child" | "teenager" | "adult" | "old",
  "clothing_type":             "dress" | "pants" | "shorts" | "skirt",
  "upper_body_clothes":        "long sleeve" | "short sleeve",
  "lower_body_clothes":        "long" | "short",
  "hat":                       "no" | "yes",
  "backpack":                  "no" | "yes",
  "bag":                       "no" | "yes",
  "handbag":                   "no" | "yes",
  "upper_body_clothes_color":  "black" | "white" | "red" | "purple" | "gray" | "blue" | "green" | "yellow" | "pink" | "orange" | "brown",
  "lower_body_clothes_color":  "black" | "white" | "pink" | "gray" | "blue" | "green" | "brown" | "yellow" | "purple" | "red"
}

For upper_body_clothes_color and lower_body_clothes_color, choose the dominant color.
No markdown, no explanation - raw JSON only.
"""

DEFAULT_PROMPT_NAME = "baseline"


class EvalError(RuntimeError):
    """Raised when evaluation data or model output is invalid."""


class InvalidFormatError(EvalError):
    """Raised when the model does not return a parseable prediction format."""

    def __init__(
        self,
        message: str,
        *,
        responses: List[Dict[str, Any]] | None = None,
        request_payload: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.responses = responses or []
        self.request_payload = request_payload or {}


class RequestFailureError(EvalError):
    """Raised when the API request fails after retries."""

    def __init__(
        self,
        message: str,
        *,
        responses: List[Dict[str, Any]] | None = None,
        request_payload: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.responses = responses or []
        self.request_payload = request_payload or {}


@dataclass(frozen=True)
class GalleryItem:
    image_path: Path
    pid: int
    class_index: int
    source_index: int


@dataclass(frozen=True)
class RequestArtifacts:
    prediction: Dict[str, int]
    responses: List[Dict[str, Any]]
    request_payload: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "prompt"


def load_prompt(prompt_file: Path | None) -> Tuple[str, Dict[str, Any]]:
    if prompt_file is None:
        prompt_text = PROMPT
        source = "builtin"
        resolved_path = None
        prompt_name = DEFAULT_PROMPT_NAME
    else:
        resolved = prompt_file.resolve()
        if not resolved.exists():
            raise EvalError(f"Prompt file not found: {resolved}")
        prompt_text = resolved.read_text(encoding="utf-8")
        source = "file"
        resolved_path = str(resolved)
        prompt_name = resolved.stem

    if not prompt_text.strip():
        raise EvalError("Prompt text is empty.")

    return prompt_text, {
        "source": source,
        "file_path": resolved_path,
        "name": prompt_name,
        "slug": slugify_name(prompt_name),
        "sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
    }


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
        help=(
            "Path to bounding_box_test. Defaults to MARKET1501_IMAGES_DIR or "
            "<dataset-root>/data/Market-1501-v15.09.15/bounding_box_test."
        ),
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
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file containing the exact prompt to send instead of the built-in prompt.",
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
        "--request-retries",
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
        help="Only evaluate the first N images after subset selection.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many gallery images before applying --limit.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1501,
        help="Seed used for reproducible random gallery sampling.",
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help="JSON file containing a fixed list of gallery image filenames to evaluate.",
    )
    parser.add_argument(
        "--write-sample-manifest",
        type=Path,
        default=None,
        help="Optional path to write the selected gallery subset as a JSON manifest.",
    )
    parser.add_argument(
        "--stratify-by",
        default=None,
        choices=["age"],
        help=(
            "Stratify gallery selection by this attribute so each class is equally represented. "
            "Per-class count = min(class_size, limit // num_classes). "
            "Cannot be combined with --sample-manifest or --sample-size."
        ),
    )
    parser.add_argument(
        "--run-log-dir",
        type=Path,
        default=Path("eval_runs"),
        help="Directory for per-run JSON logs with config, timing, outputs, and metrics.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, Path]:
    dataset_root = args.dataset_root.resolve()
    mat_path = (args.mat_path or (dataset_root / "market_attribute.mat")).resolve()
    images_dir_input = args.images_dir or os.environ.get("MARKET1501_IMAGES_DIR")
    if images_dir_input is None:
        images_dir_input = dataset_root / "data" / "Market-1501-v15.09.15" / "bounding_box_test"
    images_dir = Path(images_dir_input).resolve()
    output_mat = args.output_mat.resolve()
    metrics_json = args.metrics_json.resolve()
    run_log_dir = args.run_log_dir.resolve()
    return mat_path, images_dir, output_mat, metrics_json, run_log_dir


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise EvalError(f"{description} not found: {path}")


def load_market_attribute(mat_path: Path) -> Any:
    data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "market_attribute" not in data:
        raise EvalError(f"'market_attribute' key not found in {mat_path}")
    return data["market_attribute"]


def list_gallery_items(images_dir: Path) -> List[GalleryItem]:
    files = sorted(images_dir.glob("*.jpg"))
    if not files:
        raise EvalError(f"No JPG files found in {images_dir}")

    items: List[GalleryItem] = []
    current_pid: int | None = None
    class_index = -1

    for source_index, image_path in enumerate(files):
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
            items.append(
                GalleryItem(
                    image_path=image_path,
                    pid=pid,
                    class_index=class_index,
                    source_index=source_index,
                )
            )

    if not items:
        raise EvalError(f"No valid positive-id gallery images found in {images_dir}")
    return items


def write_sample_manifest(path: Path, gallery_items: Sequence[GalleryItem]) -> None:
    payload = {
        "count": len(gallery_items),
        "images": [item.image_path.name for item in gallery_items],
    }
    write_json(path.resolve(), payload)


def select_gallery_items(
    gallery_items: Sequence[GalleryItem],
    *,
    sample_manifest: Path | None,
    sample_size: int | None,
    sample_seed: int,
    limit: int | None,
    stratify_by: str | None = None,
    test_split: Any = None,
) -> List[GalleryItem]:
    selected = list(gallery_items)

    if stratify_by is not None:
        if sample_manifest is not None or sample_size is not None:
            raise EvalError("--stratify-by cannot be combined with --sample-manifest or --sample-size")
        if test_split is None:
            raise EvalError("--stratify-by requires dataset ground truth (test_split)")

        attr_values = attr_array(test_split, stratify_by)
        by_class: Dict[int, List[GalleryItem]] = {}
        for item in selected:
            val = int(attr_values[item.class_index])
            by_class.setdefault(val, []).append(item)

        num_classes = len(by_class)
        if num_classes == 0:
            raise EvalError(f"No classes found for --stratify-by {stratify_by!r}")

        per_class = min(len(items) for items in by_class.values())
        if limit is not None and limit > 0:
            per_class = min(per_class, limit // num_classes)
        if per_class <= 0:
            raise EvalError(f"--stratify-by {stratify_by!r}: computed per-class count is 0")

        rng = random.Random(sample_seed)
        selected = []
        for _val, items in sorted(by_class.items()):
            n = min(per_class, len(items))
            selected.extend(rng.sample(items, n))
        rng.shuffle(selected)
    elif sample_manifest is not None:
        manifest_path = sample_manifest.resolve()
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest_data, dict):
            image_names = manifest_data.get("images")
        else:
            image_names = manifest_data
        if not isinstance(image_names, list) or not image_names:
            raise EvalError(f"Sample manifest must contain a non-empty 'images' list: {manifest_path}")

        by_name = {item.image_path.name: item for item in selected}
        missing = [name for name in image_names if name not in by_name]
        if missing:
            raise EvalError(
                f"Sample manifest references {len(missing)} missing images. First missing entry: {missing[0]}"
            )
        selected = [by_name[name] for name in image_names]
    elif sample_size is not None:
        if sample_size <= 0:
            raise EvalError(f"--sample-size must be positive, got {sample_size}")
        if sample_size > len(selected):
            raise EvalError(
                f"--sample-size {sample_size} exceeds available gallery size {len(selected)}"
            )
        rng = random.Random(sample_seed)
        sampled_indices = sorted(rng.sample(range(len(selected)), sample_size))
        selected = [selected[index] for index in sampled_indices]

    if limit is not None:
        if limit <= 0:
            raise EvalError(f"--limit must be positive, got {limit}")
        selected = selected[:limit]

    if not selected:
        raise EvalError("Subset selection produced no gallery images to evaluate.")
    return selected


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
        raise InvalidFormatError("Model returned an empty response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # OpenRouter thinking models sometimes drop leading chars at the reasoning/content
        # boundary. Try prefix recovery first, then fall back to regex key-value extraction.
        stripped = text.strip()
        if not stripped.startswith("{"):
            tail = stripped.lstrip('{"')
            if re.match(r"\s*:", tail):
                try:
                    parsed = json.loads('{"gender"' + tail)
                    if isinstance(parsed, dict) and next(iter(parsed), None) == "gender":
                        return parsed
                except json.JSONDecodeError:
                    pass
            if re.match(r"gender\"\s*:", tail):
                for prefix in ("{", '{"'):
                    try:
                        parsed = json.loads(prefix + tail)
                        if isinstance(parsed, dict) and next(iter(parsed), None) == "gender":
                            return parsed
                    except json.JSONDecodeError:
                        pass
        # Regex fallback: extract all "key": "value" pairs — handles any truncation depth.
        # Safe here because our schema has only string values. Require more than one
        # recovered pair so we do not silently accept a trivially partial object.
        pairs = [
            (key, value)
            for key, value in re.findall(r'"([\w_]+)"\s*:\s*"([^"]*)"', text)
            if key in EXPECTED_RESPONSE_KEYS
        ]
        if len(pairs) >= 2:
            return {k: v for k, v in pairs}
        raise InvalidFormatError(f"Could not find JSON object in model response: {text!r}")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InvalidFormatError(f"Invalid JSON in model response: {text!r}") from exc

    if not isinstance(parsed, dict):
        raise InvalidFormatError(f"Expected a JSON object, got: {type(parsed).__name__}")
    return parsed


def compact_json(data: Any, limit: int = 800) -> str:
    try:
        text = json.dumps(data, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(data)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_request_record(
    payload: Dict[str, Any],
    http_referer: str | None,
    app_title: str | None,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": "Bearer <redacted>"}
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if app_title:
        headers["X-Title"] = app_title
    return {"url": OPENROUTER_URL, "headers": headers, "json": payload}


def extract_message_text(body: Dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InvalidFormatError(f"OpenRouter response missing choices: {compact_json(body)}")

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise InvalidFormatError(f"Unexpected choice payload: {compact_json(choice0)}")

    message = choice0.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    if part.strip():
                        text_parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                for key in ("text", "content", "output_text"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value)
                if part.get("type") == "text":
                    value = part.get("text")
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value)
            merged = "\n".join(text_parts).strip()
            if merged:
                return merged

        for key in ("reasoning", "refusal"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value

    for key in ("text",):
        value = choice0.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise InvalidFormatError(
        "OpenRouter response did not contain parseable text content: "
        f"{compact_json(choice0)}"
    )


def validate_prediction(raw: Dict[str, Any]) -> Dict[str, int]:
    key_remap = {
        "upper_body_clothes": "up",
        "lower_body_clothes": "down",
        "clothing_type": "clothes",
        "upper_body_clothes_color": "up_color",
        "lower_body_clothes_color": "down_color",
    }
    label_maps = {
        "gender": {"male": 1, "female": 2},
        "hair": {"short": 1, "long": 2},
        "age": {"child": 1, "teenager": 2, "adult": 3, "old": 4},
        "up": {"long sleeve": 1, "short sleeve": 2},
        "down": {"long": 1, "short": 2},
        "clothes": {"dress": 1, "pants": 2},
        "hat": {"no": 1, "yes": 2},
        "backpack": {"no": 1, "yes": 2},
        "bag": {"no": 1, "yes": 2},
        "handbag": {"no": 1, "yes": 2},
        "up_color": {
            "black": 1,
            "white": 2,
            "red": 3,
            "purple": 4,
            "gray": 5,
            "blue": 6,
            "green": 7,
            "yellow": 8,
        },
        "down_color": {
            "black": 1,
            "white": 2,
            "pink": 3,
            "gray": 4,
            "blue": 5,
            "green": 6,
            "brown": 7,
            "yellow": 8,
            "purple": 9,
        },
    }

    normalized_raw: Dict[str, Any] = {}
    for key, value in raw.items():
        mapped_key = key_remap.get(key, key)
        normalized_raw[mapped_key] = value

    clothes_value = normalized_raw.get("clothes")
    if isinstance(clothes_value, str) and " ".join(clothes_value.strip().lower().split()) == "dress":
        normalized_raw["down"] = "long"

    pred: Dict[str, int] = {}
    for key, mapping in label_maps.items():
        if key not in normalized_raw:
            raise InvalidFormatError(f"Missing key '{key}' in model response")

        value = normalized_raw[key]
        if isinstance(value, str):
            normalized = " ".join(value.strip().lower().split())
            remap = LABEL_REMAP.get(key)
            if remap and normalized in remap:
                normalized = remap[normalized]
            if normalized not in mapping:
                raise InvalidFormatError(
                    f"Unexpected label for '{key}': {value!r}. Expected one of {sorted(mapping)}"
                )
            pred[key] = mapping[normalized]
            continue

        try:
            numeric_value = int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidFormatError(
                f"Value for '{key}' must be a string label or integer: {value!r}"
            ) from exc

        if numeric_value not in mapping.values():
            raise InvalidFormatError(
                f"Integer value for '{key}' out of range: {numeric_value}"
            )
        pred[key] = numeric_value

    return pred


def invalid_prediction() -> Dict[str, int]:
    return {key: INVALID_PREDICTION_VALUE for key in PREDICTION_COLUMNS}


def request_prediction(
    image_path: Path,
    model: str,
    prompt_text: str,
    api_key: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    request_retries: int,
    http_referer: str | None,
    app_title: str | None,
) -> RequestArtifacts:
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
                    {"type": "text", "text": prompt_text},
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
    response_log: List[Dict[str, Any]] = []
    for attempt in range(1, request_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            response_log.append(
                {
                    "stage": "request_attempt",
                    "request_attempt": attempt,
                    "format_attempt": 1,
                    "response": body,
                }
            )
            format_error: Exception | None = None
            for format_attempt in range(1, 3):
                try:
                    content = extract_message_text(body)
                    return RequestArtifacts(
                        prediction=validate_prediction(parse_json_object(content)),
                        responses=response_log,
                        request_payload=payload,
                    )
                except InvalidFormatError as exc:
                    format_error = exc
                    if format_attempt == 1:
                        print(
                            f"[warn] invalid format for {image_path.name} "
                            f"(format attempt 1/2), retrying once: {exc}",
                            file=sys.stderr,
                        )
                        response = requests.post(
                            OPENROUTER_URL,
                            headers=headers,
                            json=payload,
                            timeout=timeout,
                        )
                        response.raise_for_status()
                        body = response.json()
                        response_log.append(
                            {
                                "stage": "format_retry",
                                "request_attempt": attempt,
                                "format_attempt": 2,
                                "response": body,
                            }
                        )
                    else:
                        raise InvalidFormatError(
                            f"Invalid format after 2 attempts: {format_error}",
                            responses=response_log,
                            request_payload=payload,
                        ) from format_error
        except InvalidFormatError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == request_retries:
                break
            backoff = min(30.0, 2 ** (attempt - 1))
            print(
                f"[warn] request failed for {image_path.name} "
                f"(attempt {attempt}/{request_retries}): {exc}",
                file=sys.stderr,
            )
            time.sleep(backoff)

    raise RequestFailureError(
        f"OpenRouter request failed for {image_path}: {last_error}",
        responses=response_log,
        request_payload=payload,
    )


def collect_predictions(
    gallery_items: Sequence[GalleryItem],
    args: argparse.Namespace,
    run_log: Dict[str, Any],
    run_artifact_dir: Path,
    prompt_text: str,
) -> List[Dict[str, int]]:
    predictions: List[Dict[str, int]] = []
    format_failures: List[Dict[str, Any]] = []
    request_failures: List[Dict[str, Any]] = []
    api_response_dir = run_artifact_dir / "api_responses"
    request_payload_written = False

    if not args.api_key:
        raise EvalError("OpenRouter API key missing. Pass --api-key or set OPENROUTER_API_KEY.")

    for index, item in enumerate(gallery_items, start=1):
        try:
            artifacts = request_prediction(
                image_path=item.image_path,
                model=args.model,
                prompt_text=prompt_text,
                api_key=args.api_key,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_retries=args.request_retries,
                http_referer=args.http_referer,
                app_title=args.app_title,
            )
            prediction = artifacts.prediction
            if not request_payload_written:
                write_json(
                    run_artifact_dir / "api_request_template.json",
                    build_request_record(artifacts.request_payload, args.http_referer, args.app_title),
                )
                request_payload_written = True
            write_json(
                api_response_dir / f"{index:05d}_{item.image_path.stem}.json",
                {
                    "image_path": str(item.image_path),
                    "request_index": index,
                    "model": args.model,
                    "responses": artifacts.responses,
                    "final_prediction": prediction,
                },
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        except InvalidFormatError as exc:
            prediction = invalid_prediction()
            message = str(exc)
            format_failures.append({"image_path": str(item.image_path), "error": message})
            if not request_payload_written and exc.request_payload:
                write_json(
                    run_artifact_dir / "api_request_template.json",
                    build_request_record(exc.request_payload, args.http_referer, args.app_title),
                )
                request_payload_written = True
            write_json(
                api_response_dir / f"{index:05d}_{item.image_path.stem}.json",
                {
                    "image_path": str(item.image_path),
                    "request_index": index,
                    "model": args.model,
                    "error_type": "InvalidFormatError",
                    "error": message,
                    "responses": exc.responses,
                    "request_payload": exc.request_payload,
                    "final_prediction": prediction,
                },
            )
            print(
                f"[warn] counting {item.image_path.name} as all incorrect due to invalid format: {message}",
                file=sys.stderr,
            )
        except RequestFailureError as exc:
            prediction = invalid_prediction()
            message = str(exc)
            request_failures.append({"image_path": str(item.image_path), "error": message})
            if not request_payload_written and exc.request_payload:
                write_json(
                    run_artifact_dir / "api_request_template.json",
                    build_request_record(exc.request_payload, args.http_referer, args.app_title),
                )
                request_payload_written = True
            write_json(
                api_response_dir / f"{index:05d}_{item.image_path.stem}.json",
                {
                    "image_path": str(item.image_path),
                    "request_index": index,
                    "model": args.model,
                    "error_type": type(exc).__name__,
                    "error": message,
                    "responses": exc.responses,
                    "request_payload": exc.request_payload,
                    "final_prediction": prediction,
                },
            )
            print(
                f"[warn] counting {item.image_path.name} as all incorrect due to request failure: {message}",
                file=sys.stderr,
            )
        except EvalError as exc:
            prediction = invalid_prediction()
            message = str(exc)
            request_failures.append({"image_path": str(item.image_path), "error": message})
            write_json(
                api_response_dir / f"{index:05d}_{item.image_path.stem}.json",
                {
                    "image_path": str(item.image_path),
                    "request_index": index,
                    "model": args.model,
                    "error_type": type(exc).__name__,
                    "error": message,
                    "final_prediction": prediction,
                },
            )
            print(
                f"[warn] counting {item.image_path.name} as all incorrect due to request failure: {message}",
                file=sys.stderr,
            )

        predictions.append(prediction)
        print(
            f"[{index}/{len(gallery_items)}] {item.image_path.name} -> "
            f"{json.dumps(prediction, sort_keys=True)}"
        )

    run_log["prediction_failures"] = {
        "invalid_format_count": len(format_failures),
        "request_failure_count": len(request_failures),
        "invalid_format_samples": format_failures,
        "request_failure_samples": request_failures,
    }
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

    age_classes = np.unique(gt["age"])
    per_class_acc = [
        float(np.mean(gallery[:, 2][gt["age"] == cls] == cls))
        for cls in age_classes
        if (gt["age"] == cls).sum() > 0
    ]
    metrics["age_balanced"] = float(np.mean(per_class_acc)) if per_class_acc else 0.0

    metrics["average"] = float(np.mean(list(metrics.values())))
    return metrics


def compute_error_breakdown(gallery: np.ndarray, gt: Dict[str, np.ndarray]) -> Dict[str, Any]:
    format_invalid_mask = np.any(gallery == INVALID_PREDICTION_VALUE, axis=1)

    up_color_correct = (
        ((gallery[:, 10] == 1) & (gt["upblack"] == 2))
        | ((gallery[:, 10] == 2) & (gt["upwhite"] == 2))
        | ((gallery[:, 10] == 3) & (gt["upred"] == 2))
        | ((gallery[:, 10] == 4) & (gt["uppurple"] == 2))
        | ((gallery[:, 10] == 5) & (gt["upgray"] == 2))
        | ((gallery[:, 10] == 6) & (gt["upblue"] == 2))
        | ((gallery[:, 10] == 7) & (gt["upgreen"] == 2))
        | ((gallery[:, 10] == 8) & (gt["upyellow"] == 2))
    )
    down_color_correct = (
        ((gallery[:, 11] == 1) & (gt["downblack"] == 2))
        | ((gallery[:, 11] == 2) & (gt["downwhite"] == 2))
        | ((gallery[:, 11] == 3) & (gt["downpink"] == 2))
        | ((gallery[:, 11] == 4) & (gt["downgray"] == 2))
        | ((gallery[:, 11] == 5) & (gt["downblue"] == 2))
        | ((gallery[:, 11] == 6) & (gt["downgreen"] == 2))
        | ((gallery[:, 11] == 7) & (gt["downbrown"] == 2))
        | ((gallery[:, 11] == 8) & (gt["downyellow"] == 2))
        | ((gallery[:, 11] == 9) & (gt["downpurple"] == 2))
    )

    attr_correct = np.column_stack(
        [
            gallery[:, 0] == gt["gender"],
            gallery[:, 2] == gt["age"],
            gallery[:, 1] == gt["hair"],
            gallery[:, 3] == gt["up"],
            gallery[:, 4] == gt["down"],
            gallery[:, 5] == gt["clothes"],
            gallery[:, 7] == gt["backpack"],
            gallery[:, 9] == gt["handbag"],
            gallery[:, 8] == gt["bag"],
            gallery[:, 6] == gt["hat"],
            up_color_correct,
            down_color_correct,
        ]
    )

    exact_match_mask = np.all(attr_correct, axis=1)
    valid_but_inaccurate_mask = (~format_invalid_mask) & (~exact_match_mask)

    return {
        "total_samples": int(len(gallery)),
        "invalid_format_or_failure_count": int(np.sum(format_invalid_mask)),
        "invalid_format_or_failure_rate": float(np.mean(format_invalid_mask)),
        "valid_but_inaccurate_count": int(np.sum(valid_but_inaccurate_mask)),
        "valid_but_inaccurate_rate": float(np.mean(valid_but_inaccurate_mask)),
        "exact_match_count": int(np.sum(exact_match_mask)),
        "exact_match_rate": float(np.mean(exact_match_mask)),
    }


def decode_one_hot_labels(gt: Dict[str, np.ndarray], field_names: Sequence[str]) -> np.ndarray:
    stacked = np.column_stack([gt[name] == 2 for name in field_names])
    if stacked.size == 0:
        return np.zeros(0, dtype=np.int32)
    any_match = np.any(stacked, axis=1)
    decoded = np.argmax(stacked, axis=1) + 1
    decoded = decoded.astype(np.int32)
    decoded[~any_match] = 0
    return decoded


def build_confusion_report(
    name: str,
    truth: np.ndarray,
    pred: np.ndarray,
    label_names: Dict[int, str],
) -> Dict[str, Any]:
    valid_truth_ids = sorted(label_names)
    valid_mask = (pred != INVALID_PREDICTION_VALUE) & np.isin(truth, valid_truth_ids)
    truth_valid = truth[valid_mask]
    pred_valid = pred[valid_mask]

    matrix: List[List[int]] = []
    rows: List[Dict[str, Any]] = []
    top_confusions: List[Dict[str, Any]] = []
    valid_pred_mask = np.isin(pred_valid, valid_truth_ids)

    for true_id in valid_truth_ids:
        row_counts: List[int] = []
        true_mask = truth_valid == true_id
        support = int(np.sum(true_mask))
        for pred_id in valid_truth_ids:
            row_counts.append(int(np.sum(true_mask & (pred_valid == pred_id))))
        matrix.append(row_counts)
        rows.append(
            {
                "label": label_names[true_id],
                "support": support,
                "accuracy": (row_counts[true_id - 1] / support) if support else 0.0,
                "predicted_counts": {
                    label_names[pred_id]: row_counts[pred_id - 1] for pred_id in valid_truth_ids
                },
            }
        )

    predicted_distribution = {
        label_names[pred_id]: int(np.sum(pred_valid == pred_id)) for pred_id in valid_truth_ids
    }
    invalid_prediction_count = int(np.sum(~valid_pred_mask))

    for true_id in valid_truth_ids:
        for pred_id in valid_truth_ids:
            if true_id == pred_id:
                continue
            count = matrix[true_id - 1][pred_id - 1]
            if count <= 0:
                continue
            support = rows[true_id - 1]["support"]
            top_confusions.append(
                {
                    "true_label": label_names[true_id],
                    "predicted_label": label_names[pred_id],
                    "count": count,
                    "rate_within_true_label": (count / support) if support else 0.0,
                }
            )

    top_confusions.sort(
        key=lambda item: (-item["count"], -item["rate_within_true_label"], item["true_label"], item["predicted_label"])
    )

    return {
        "attribute": name,
        "evaluated_samples": int(len(truth_valid)),
        "invalid_prediction_count": invalid_prediction_count,
        "labels": [label_names[label_id] for label_id in valid_truth_ids],
        "matrix_true_rows_pred_cols": matrix,
        "rows": rows,
        "predicted_distribution": predicted_distribution,
        "top_confusions": top_confusions[:5],
    }


def compute_analysis_report(gallery: np.ndarray, gt: Dict[str, np.ndarray]) -> Dict[str, Any]:
    up_color_truth = decode_one_hot_labels(gt, UP_COLOR_GT_FIELDS)
    down_color_truth = decode_one_hot_labels(gt, DOWN_COLOR_GT_FIELDS)
    return {
        "age": build_confusion_report("age", gt["age"], gallery[:, 2], AGE_LABELS),
        "up_color": build_confusion_report("up_color", up_color_truth, gallery[:, 10], UP_COLOR_LABELS),
        "down_color": build_confusion_report(
            "down_color", down_color_truth, gallery[:, 11], DOWN_COLOR_LABELS
        ),
    }


def write_metrics(
    path: Path,
    metrics: Dict[str, float],
    gallery_size: int,
    model: str,
    analysis_report: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "model": model,
        "gallery_size": gallery_size,
        "metrics": metrics,
    }
    if analysis_report is not None:
        payload["analysis_report"] = analysis_report
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nMetrics")
    print("-------")
    ordered = [
        "gender",
        "age",
        "age_balanced",
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


def print_analysis_report(analysis_report: Dict[str, Any]) -> None:
    print("\nAnalysis")
    print("--------")
    for attr_name in ("age", "up_color", "down_color"):
        report = analysis_report[attr_name]
        print(
            f"{attr_name}: evaluated={report['evaluated_samples']}, "
            f"invalid_pred={report['invalid_prediction_count']}"
        )
        if not report["top_confusions"]:
            print("  top_confusions: none")
            continue
        summary = ", ".join(
            (
                f"{item['true_label']}->{item['predicted_label']} "
                f"({item['count']}, {item['rate_within_true_label']:.3f})"
            )
            for item in report["top_confusions"]
        )
        print(f"  top_confusions: {summary}")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def build_run_log_base(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "started_at_utc": utc_now_iso(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "arguments": {
            key: json_safe(value) for key, value in vars(args).items()
        },
        "environment": {
            "OPENROUTER_API_KEY_present": bool(os.environ.get("OPENROUTER_API_KEY")),
            "OPENROUTER_HTTP_REFERER": os.environ.get("OPENROUTER_HTTP_REFERER"),
            "OPENROUTER_APP_TITLE": os.environ.get("OPENROUTER_APP_TITLE"),
            "MARKET1501_IMAGES_DIR": os.environ.get("MARKET1501_IMAGES_DIR"),
        },
    }


def write_run_log(run_artifact_dir: Path, run_log: Dict[str, Any]) -> Path:
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    status = str(run_log.get("status", "unknown"))
    log_path = run_artifact_dir / f"run_log_{status}.json"
    log_path.write_text(json.dumps(run_log, indent=2, sort_keys=True), encoding="utf-8")
    return log_path


def make_run_artifact_dir(run_log_dir: Path, model: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "__")
    artifact_dir = run_log_dir / f"{timestamp}_{model_slug}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def main() -> int:
    args = parse_args()
    run_started = time.time()
    run_log = build_run_log_base(args)
    exit_code = 1
    run_log_path: Path | None = None
    run_artifact_dir: Path | None = None

    try:
        dataset_root = args.dataset_root.resolve()
        load_dotenv(dataset_root / ".env")
        if args.api_key is None:
            args.api_key = os.environ.get("OPENROUTER_API_KEY")
        if args.http_referer is None:
            args.http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        if args.app_title == "Market1501 Attribute Eval":
            args.app_title = os.environ.get("OPENROUTER_APP_TITLE", args.app_title)

        mat_path, images_dir, output_mat, metrics_json, run_log_dir = resolve_paths(args)
        run_artifact_dir = make_run_artifact_dir(run_log_dir, args.model)
        prompt_text, prompt_info = load_prompt(args.prompt_file)
        prompt_artifact_path = run_artifact_dir / "prompt.txt"
        prompt_metadata_path = run_artifact_dir / "prompt_metadata.json"
        prompt_artifact_path.write_text(prompt_text, encoding="utf-8")
        write_json(prompt_metadata_path, prompt_info)
        run_log["resolved_paths"] = {
            "dataset_root": str(dataset_root),
            "mat_path": str(mat_path),
            "images_dir": str(images_dir),
            "output_mat": str(output_mat),
            "metrics_json": str(metrics_json),
            "run_log_dir": str(run_log_dir),
            "run_artifact_dir": str(run_artifact_dir),
        }
        run_log["model"] = args.model
        run_log["prompt"] = prompt_info
        require_path(mat_path, "Attribute MAT file")
        require_path(images_dir, "Gallery image directory")

        market = load_market_attribute(mat_path)
        all_gallery_items = list_gallery_items(images_dir)
        gallery_items = select_gallery_items(
            all_gallery_items,
            sample_manifest=args.sample_manifest,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            limit=args.limit,
            stratify_by=args.stratify_by,
            test_split=market.test if args.stratify_by else None,
        )
        if args.write_sample_manifest is not None:
            write_sample_manifest(args.write_sample_manifest, gallery_items)
        run_log["gallery"] = {
            "num_items": len(gallery_items),
            "first_image": str(gallery_items[0].image_path),
            "last_image": str(gallery_items[-1].image_path),
            "selection": {
                "available_items": len(all_gallery_items),
                "limit": args.limit,
                "sample_manifest": (
                    str(args.sample_manifest.resolve()) if args.sample_manifest is not None else None
                ),
                "sample_seed": args.sample_seed,
                "sample_size": args.sample_size,
                "stratify_by": args.stratify_by,
                "write_sample_manifest": (
                    str(args.write_sample_manifest.resolve())
                    if args.write_sample_manifest is not None
                    else None
                ),
            },
        }
        gt = build_ground_truth(market.test, gallery_items)

        predictions = collect_predictions(gallery_items, args, run_log, run_artifact_dir, prompt_text)
        gallery = predictions_to_matrix(predictions)

        output_mat.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(str(output_mat), {"gallery": gallery})

        metrics = compute_metrics(gallery, gt)
        error_breakdown = compute_error_breakdown(gallery, gt)
        analysis_report = compute_analysis_report(gallery, gt)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        write_metrics(metrics_json, metrics, len(gallery_items), args.model, analysis_report)
        write_json(run_artifact_dir / "analysis_report.json", analysis_report)
        run_log["outputs"] = {
            "output_mat": str(output_mat),
            "metrics_json": str(metrics_json),
            "analysis_report": str(run_artifact_dir / "analysis_report.json"),
            "prompt_text": str(prompt_artifact_path),
            "prompt_metadata": str(prompt_metadata_path),
            "api_request_template": str(run_artifact_dir / "api_request_template.json"),
            "api_response_dir": str(run_artifact_dir / "api_responses"),
        }
        run_log["metrics"] = metrics
        run_log["error_breakdown"] = error_breakdown
        run_log["analysis_report"] = analysis_report
        run_log["status"] = "success"

        print(f"\nSaved gallery predictions to: {output_mat}")
        print(f"Saved metrics to: {metrics_json}")
        print_metrics(metrics)
        print("\nError Breakdown")
        print("---------------")
        for key, value in error_breakdown.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        print_analysis_report(analysis_report)
        exit_code = 0
    except KeyboardInterrupt:
        run_log["status"] = "aborted"
        run_log["error_type"] = "KeyboardInterrupt"
        run_log["error"] = "Run interrupted by user."
        print("[warn] run interrupted by user.", file=sys.stderr)
    except EvalError as exc:
        run_log["status"] = "error"
        run_log["error_type"] = type(exc).__name__
        run_log["error"] = str(exc)
        print(f"[error] {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        run_log["status"] = "error"
        run_log["error_type"] = type(exc).__name__
        run_log["error"] = str(exc)
        raise
    finally:
        run_log["finished_at_utc"] = utc_now_iso()
        run_log["duration_seconds"] = round(time.time() - run_started, 3)
        if "status" not in run_log:
            run_log["status"] = "unknown"
        try:
            if run_artifact_dir is not None:
                run_log["run_artifact_dir"] = str(run_artifact_dir)
                run_log_path = write_run_log(run_artifact_dir, run_log)
            else:
                _, _, _, _, run_log_dir = resolve_paths(args)
                run_log_path = write_run_log(run_log_dir, run_log)
        except Exception as log_exc:  # noqa: BLE001
            print(f"[warn] failed to write run log: {log_exc}", file=sys.stderr)
        else:
            print(f"Saved run log to: {run_log_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
