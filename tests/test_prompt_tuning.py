import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval_qwen3_vl_openrouter import DEFAULT_PROMPT_NAME, PROMPT, GalleryItem, load_prompt
from run_prompt_tuning_sweep import (
    build_age_balanced_pilot_items,
    decide_prompt_variant,
    resolve_prompt_files,
)


class PromptLoadingTests(unittest.TestCase):
    def test_load_prompt_builtin_matches_default(self) -> None:
        prompt_text, prompt_info = load_prompt(None)

        self.assertEqual(prompt_text, PROMPT)
        self.assertEqual(prompt_info["source"], "builtin")
        self.assertEqual(prompt_info["name"], DEFAULT_PROMPT_NAME)
        self.assertIsNone(prompt_info["file_path"])
        self.assertTrue(prompt_info["sha256"])

    def test_load_prompt_file_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "custom_prompt.txt"
            prompt_path.write_text("hello prompt\n", encoding="utf-8")

            prompt_text, prompt_info = load_prompt(prompt_path)

            self.assertEqual(prompt_text, "hello prompt\n")
            self.assertEqual(prompt_info["source"], "file")
            self.assertEqual(prompt_info["name"], "custom_prompt")
            self.assertEqual(prompt_info["slug"], "custom_prompt")
            self.assertEqual(prompt_info["file_path"], str(prompt_path.resolve()))

    def test_resolve_prompt_files_always_prepends_baseline(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        prompt_files = resolve_prompt_files(repo_root, [Path("prompts/reid_focus.txt")])

        self.assertEqual(prompt_files[0], (repo_root / "prompts" / "baseline.txt").resolve())


class PromptDecisionTests(unittest.TestCase):
    def test_final_gate_requires_all_models_to_improve(self) -> None:
        per_model_deltas = {
            "google/gemma-4-31b-it": {"reid_useful_delta": 0.01, "average_delta": 0.0, "failure_delta": 0},
            "qwen/qwen3.5-9b": {"reid_useful_delta": 0.01, "average_delta": 0.0, "failure_delta": 0},
            "google/gemini-2.5-flash-lite": {"reid_useful_delta": 0.002, "average_delta": 0.0, "failure_delta": 0},
            "qwen/qwen3.5-35b-a3b": {"reid_useful_delta": -0.001, "average_delta": 0.0, "failure_delta": 0},
        }

        decision = decide_prompt_variant(per_model_deltas, average_tolerance=0.005, final_gate=True)

        self.assertEqual(decision, "rejected_anchor_no_gain")

    def test_final_gate_accepts_when_all_checks_pass(self) -> None:
        per_model_deltas = {
            "google/gemma-4-31b-it": {"reid_useful_delta": 0.01, "average_delta": -0.001, "failure_delta": 0},
            "qwen/qwen3.5-9b": {"reid_useful_delta": 0.005, "average_delta": 0.0, "failure_delta": 0},
            "google/gemini-2.5-flash-lite": {"reid_useful_delta": 0.004, "average_delta": -0.002, "failure_delta": 0},
            "qwen/qwen3.5-35b-a3b": {"reid_useful_delta": 0.003, "average_delta": 0.001, "failure_delta": 0},
        }

        decision = decide_prompt_variant(per_model_deltas, average_tolerance=0.005, final_gate=True)

        self.assertEqual(decision, "accepted")

    def test_pilot_gate_rejects_multiple_anchor_losses(self) -> None:
        per_model_deltas = {
            "google/gemma-4-31b-it": {"reid_useful_delta": 0.01, "average_delta": 0.0, "failure_delta": 0},
            "qwen/qwen3.5-9b": {"reid_useful_delta": -0.01, "average_delta": 0.0, "failure_delta": 0},
            "google/gemini-2.5-flash-lite": {"reid_useful_delta": -0.02, "average_delta": 0.0, "failure_delta": 0},
            "qwen/qwen3.5-35b-a3b": {"reid_useful_delta": 0.001, "average_delta": 0.0, "failure_delta": 0},
        }

        decision = decide_prompt_variant(per_model_deltas, average_tolerance=0.005, final_gate=False)

        self.assertEqual(decision, "rejected_anchor_no_gain")


class PromptPilotSelectionTests(unittest.TestCase):
    def test_build_age_balanced_pilot_items_fills_requested_limit(self) -> None:
        items = []
        for class_index in range(4):
            for item_index in range(3):
                items.append(
                    GalleryItem(
                        image_path=Path(f"class{class_index}_{item_index}.jpg"),
                        pid=class_index + 1,
                        class_index=class_index,
                        source_index=len(items),
                    )
                )

        test_split = type("Split", (), {"age": np.array([1, 2, 3, 4], dtype=np.int32)})()
        selected = build_age_balanced_pilot_items(items, test_split=test_split, limit=5, seed=7)

        self.assertEqual(len(selected), 5)
        selected_ages = [int(test_split.age[item.class_index]) for item in selected]
        age_counts = {age: selected_ages.count(age) for age in set(selected_ages)}
        self.assertEqual(sum(age_counts.values()), 5)
        self.assertTrue(all(count >= 1 for count in age_counts.values()))


if __name__ == "__main__":
    unittest.main()
