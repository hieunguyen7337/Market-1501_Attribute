# CLAUDE.md

## What This Repo Is

This repository evaluates vision-language models on the Market-1501 person attribute benchmark.

The main workflow is:

1. Load identity-level annotations from `market_attribute.mat`
2. Expand those identity labels to image-level labels for the gallery images
3. Send each image to an OpenRouter vision model
4. Parse the model response into the 12-column attribute format expected by the original MATLAB evaluation
5. Save predictions and per-attribute metrics

Important dataset detail:

- Market-1501 attributes are identity-level annotations, not per-image annotations
- That matters especially for `age`, because the visual appearance in one frame may look more like `adult`, while the dataset label is still `teenager`


## Main Files

- `eval_qwen3_vl_openrouter.py`
  The main evaluator. This is the script to run for real experiments.

- `run_sample_eval.ps1`
  Convenience wrapper for small/sample runs using the local `.venv`.

- `evaluate_market_attribute.m`
  Original MATLAB-side evaluation reference.

- `market_attribute.mat`
  Ground-truth attribute annotations.

- `sample_outputs/`
  Default location for sample-run outputs.

- `eval_runs/`
  Per-run logs, API responses, and run metadata.

- `sample_outputs/market1501_random500_seed1501_manifest.json`
  Stable random subset manifest added to avoid biased “first N sorted images” small evals.


## Repo Goal

The project is trying to measure how well OpenRouter VLMs can predict person attributes on Market-1501, especially using a reproducible Python evaluation flow instead of a manual process.

The current practical goals are:

- run small, cheap sanity-check evals on reproducible subsets
- run larger evals when needed
- inspect failure cases by looking at saved API responses in `eval_runs/`
- improve prompts and sampling so attribute scores reflect model behavior more fairly


## Environment

Use the repo-local virtual environment:

```powershell
.\.venv\Scripts\python.exe --version
```

Dependencies are in `requirements.txt`.


## Expected Data Layout

Default gallery image directory:

`data\Market-1501-v15.09.15\bounding_box_test`

If needed, the evaluator can also take a custom image directory through `--images-dir` or the `MARKET1501_IMAGES_DIR` env var.


## Core Evaluator Command

Run the evaluator directly:

```powershell
.\.venv\Scripts\python.exe .\eval_qwen3_vl_openrouter.py `
  --dataset-root . `
  --images-dir .\data\Market-1501-v15.09.15\bounding_box_test `
  --model qwen/qwen3.5-9b `
  --max-tokens 2048 `
  --sleep-seconds 0.25 `
  --output-mat .\sample_outputs\gallery_market_sample.mat `
  --metrics-json .\sample_outputs\market_attribute_metrics_sample.json
```

This requires `OPENROUTER_API_KEY` to be available, either from `.env` or the shell environment.


## Small Eval Commands

### Recommended stable small eval

Use the prebuilt random 500-image manifest, then cap to 100 images for a cheap run:

```powershell
.\run_sample_eval.ps1 `
  -Limit 100 `
  -SampleManifest .\sample_outputs\market1501_random500_seed1501_manifest.json
```

This is the preferred quick check because it avoids the biased “first 100 sorted images” slice.

### Generate a new reproducible random subset manifest

```powershell
.\run_sample_eval.ps1 `
  -Limit 500 `
  -SampleSize 500 `
  -SampleSeed 2026 `
  -WriteSampleManifest .\sample_outputs\market1501_random500_seed2026_manifest.json
```

Note:

- `-SampleSize` randomly samples before `-Limit`
- `-SampleSeed` makes the subset reproducible
- `-SampleManifest` reuses an exact saved subset


## Why The Sampling Change Was Added

The previous small eval used `--limit 100` on the first sorted gallery images.

That produced a pathological slice for age:

- first 100 images all had ground-truth age label `teenager`
- the model often predicted `adult`
- the age score collapsed even when the rest of the attributes were reasonable

The random subset manifest was added to make small evals less misleading.


## Outputs To Check

### Metrics

- `sample_outputs\market_attribute_metrics_sample.json`

### Prediction matrix

- `sample_outputs\gallery_market_sample.mat`

### Run artifacts

Each run creates a folder in `eval_runs\...` containing:

- `run_log_success.json` or `run_log_aborted.json`
- `api_request_template.json`
- `api_responses\*.json`

The per-image response files are the best place to debug bad attributes.


## How Age Is Scored

Age mapping in the evaluator is:

- `young` -> `1`
- `teenager` -> `2`
- `adult` -> `3`
- `old` -> `4`

Scoring is strict equality against the expanded ground-truth labels.

There is no fuzzy matching between adjacent age groups.


## If A Score Looks Wrong

Check in this order:

1. `eval_runs/<latest_run>/run_log_success.json`
   Confirm the run actually completed and inspect per-attribute metrics.

2. `eval_runs/<latest_run>/api_responses/*.json`
   Inspect raw model outputs and parsed predictions.

3. The subset selection
   Confirm whether the run used:
   - the first sorted images
   - a random subset
   - a manifest-based subset

4. Ground-truth distribution for the evaluated slice
   A skewed slice can make one attribute look much worse than it really is.


## Good Defaults

For quick sanity checks:

- model: `qwen/qwen3.5-9b`
- `--max-tokens 2048`
- `--sleep-seconds 0.25`
- use the saved 500-image manifest
- evaluate only 100 images unless you need stronger signal


## Things Claude Should Avoid Assuming

- Do not assume the first `N` sorted images are representative
- Do not assume age labels are visually obvious per image
- Do not assume a low age score is a parser bug
- Do not overwrite user outputs in `sample_outputs/` unless intentionally rerunning


## Suggested Next Improvements

If continuing work in this repo, the highest-value next steps are:

1. Add a confusion-matrix report per attribute, especially for age
2. Add stratified subset generation so age classes are more balanced
3. Tune the age wording in the prompt to better match dataset taxonomy
4. Compare multiple OpenRouter models on the same saved manifest for fairer benchmarking
