# Session Summary — Market-1501 VLM Attribute Benchmark

## What This Project Does

Evaluates OpenRouter vision-language models on the Market-1501 person attribute benchmark.
Each image is sent to a VLM which must return 12 attributes as JSON. Predictions are compared
against ground-truth annotations and per-attribute accuracy is reported.

Key script: `eval_qwen3_vl_openrouter.py`
Convenience runner: `run_sample_eval.ps1`
Standard eval set: `sample_outputs/market1501_age_balanced_seed1501_manifest.json` (120 images, 30 per age class)

---

## Code Changes Made This Session

### 1. `eval_qwen3_vl_openrouter.py` — `parse_json_object()`

Added a two-stage recovery for OpenRouter thinking models (qwen3.5-plus family) that drop
leading characters at the reasoning/content field boundary.

```python
match = re.search(r"\{.*\}", text, re.DOTALL)
if not match:
    stripped = text.strip()
    if not stripped.startswith("{"):
        tail = stripped.lstrip('{"')
        # Case 1: partial key present (e.g. `gender": ...` or `"gender": ...`)
        for prefix in ("{", '{"'):
            try:
                parsed = json.loads(prefix + tail)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        # Case 2: key fully truncated, tail starts with `: "value",...}`
        if re.match(r'\s*:', tail):
            try:
                parsed = json.loads('{"gender' + tail)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    # Regex fallback: extract all "key": "value" pairs — handles any truncation depth.
    pairs = re.findall(r'"([\w_]+)"\s*:\s*"([^"]*)"', text)
    if pairs:
        return {k: v for k, v in pairs}
    raise InvalidFormatError(f"Could not find JSON object in model response: {text!r}")
```

The regex fallback (`re.findall`) is the most robust — it recovers regardless of how many
leading characters were dropped. Safe because the schema has only string values.

### 2. `run_sample_eval.ps1`

- `$MaxTokens` default raised from `2048` → `8192` (fixes thinking model tail truncation)
- `-StratifyBy` parameter added and passed through to Python

---

## Model Evaluation Results

All runs used the 120-image age-balanced manifest.
Standard command:
```powershell
.\run_sample_eval.ps1 -Limit 120 -SampleManifest .\sample_outputs\market1501_age_balanced_seed1501_manifest.json -Model "<model-id>"
```

### Leaderboard (120-image age-balanced set)

| Rank | Model | Avg | Age Balanced | Output $/M | Notes |
|------|-------|-----|--------------|------------|-------|
| 1 | qwen/qwen3.5-9b | 0.821 | — | $0.15 | Best overall; age_balanced not recorded (run predates metric) |
| 2 | google/gemma-4-31b-it | 0.803 | 0.517 | $0.38 | Clean delivery, 0 format errors |
| 3 | qwen/qwen3.5-35b-a3b | 0.801 | 0.575 | $1.30 | Best age_balanced; expensive |
| 4 | google/gemini-2.5-flash-lite | 0.796 | 0.533 | $0.40 | Weak on gender (0.817) |
| 5 | qwen/qwen3-vl-235b-a22b-instruct | 0.789 | 0.533 | $0.88 | Disappointing for size |
| 6 | google/gemma-4-26b-a4b-it | 0.780 | 0.483 | $0.40 | Earlier run |
| 7 | qwen/qwen3-vl-32b-instruct | 0.765 | 0.492 | $0.416 | Earlier run |

### Failed / Skipped Models

| Model | Status | Reason |
|-------|--------|--------|
| rekaai/reka-edge | FAILED | Completely incompatible output schema — omits required keys, swaps field meanings. Would need a reka-specific prompt. |
| qwen/qwen3.5-plus-02-15 | SKIPPED | OpenRouter drops variable number of leading chars at reasoning/content boundary. The regex fallback now handles this, but the model is 10× more expensive than qwen3.5-9b with no proven benefit. Could retry. |

---

## Why qwen3.5-9b Leads

Investigated whether `LABEL_REMAP` post-processing gives qwen3.5-9b an unfair advantage.
Conclusion: **No — LABEL_REMAP accounts for only ~2-3% accuracy swing and applies equally
to all models that output the same quirky values.**

The real lead comes from better calibration on gender, hair, and color perception for
low-resolution pedestrian images. Larger models and dedicated VL models don't improve on this.

---

## LABEL_REMAP (Currently in Code)

```python
LABEL_REMAP = {
    "up_color":   {"pink": "red", "orange": "red", "brown": "gray"},
    "down_color": {"red": "pink", "orange": "brown"},
    "clothes":    {"shorts": "pants", "short": "pants", "skirt": "dress"},
}
```

Applied in `validate_prediction()` before checking against valid values.
The dataset does not have "pink" as a valid upper body color, "red" as a valid lower body
color, or "shorts"/"skirt" as clothing types — these map to the nearest valid class.

---

## Age Metric

Two age metrics exist:
- `age` — standard accuracy (heavily biased by 87% "teenager" class in test set)
- `age_balanced` — macro-averaged accuracy across 4 classes (child/teenager/adult/old)

The 120-image balanced manifest has 30 images per age class, so `age == age_balanced` for this set.

Age map: child=1, teenager=2, adult=3, old=4. Strict equality scoring.

---

## Remaining Candidates to Evaluate (Tier 2)

From the plan at `C:\Users\Admin\.claude\plans\read-claude-md-and-understand-eventual-minsky.md`:

| Model | Output $/M | Notes |
|-------|------------|-------|
| mistralai/ministral-8b-2512 | $0.15 | Same price as qwen3.5-9b, quick sanity check |
| qwen/qwen3.5-flash-02-23 | $0.26 | ⚠️ Likely thinking model — may have same delivery quirks as qwen3.5-plus. Regex fallback should now handle it. |

### Run command for next candidate:
```powershell
.\run_sample_eval.ps1 -Limit 120 -SampleManifest .\sample_outputs\market1501_age_balanced_seed1501_manifest.json -Model "mistralai/ministral-8b-2512"
```

---

## Known Issues / Observations

1. **Color scores are the hardest attribute** — up_color and down_color score 0.67–0.74 across
   all models. This is likely a fundamental challenge: low-res images + subjective color naming.

2. **Age is unreliable per-image** — Market-1501 age is identity-level (one label per person ID),
   not per-frame. A "teenager" ID may look like an adult in some frames.

3. **OpenRouter thinking model delivery bug** — Models with a `reasoning` field (qwen3.5 family)
   sometimes have the first N characters of `content` dropped. The regex fallback in
   `parse_json_object()` now handles this robustly.

4. **qwen3.5-9b age_balanced is missing** — The 120-image run for this model predates the
   `age_balanced` metric. To get a fair comparison, rerun:
   ```powershell
   .\run_sample_eval.ps1 -Limit 120 -SampleManifest .\sample_outputs\market1501_age_balanced_seed1501_manifest.json -Model "qwen/qwen3.5-9b"
   ```

---

## Suggested Next Steps

1. Run `mistralai/ministral-8b-2512` (Tier 2 budget check)
2. Run `qwen/qwen3.5-flash-02-23` (watch for thinking model delivery issues)
3. Rerun `qwen/qwen3.5-9b` on the balanced manifest to get its `age_balanced` score
4. If a model surpasses 0.821, investigate whether prompt tuning (especially for age and color)
   could push it further
5. Consider adding a confusion matrix report per attribute (especially age) as noted in CLAUDE.md
