import json
import unittest
from pathlib import Path

from eval_qwen3_vl_openrouter import (
    InvalidFormatError,
    build_confusion_report,
    compute_analysis_report,
    parse_json_object,
    validate_prediction,
)

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
QWEN_PLUS_RUN_DIR = (
    REPO_ROOT
    / "eval_runs"
    / "20260418_184658_qwen__qwen3.5-plus-02-15"
    / "api_responses"
)


def load_message_content(filename: str, response_index: int = 0) -> str:
    payload = json.loads((QWEN_PLUS_RUN_DIR / filename).read_text(encoding="utf-8"))
    return payload["responses"][response_index]["response"]["choices"][0]["message"]["content"]


class ParseJsonObjectTests(unittest.TestCase):
    def test_intact_json_parses_unchanged(self) -> None:
        content = load_message_content("00001_0015_c6s1_001476_01.json")

        parsed = parse_json_object(content)

        self.assertEqual(parsed["gender"], "male")
        self.assertEqual(parsed["age"], "adult")
        self.assertEqual(parsed["upper_body_clothes_color"], "purple")

    def test_partial_first_key_recovers_gender(self) -> None:
        content = 'gender": "male", "hair": "short", "age": "adult"}'

        parsed = parse_json_object(content)

        self.assertEqual(
            parsed,
            {
                "gender": "male",
                "hair": "short",
                "age": "adult",
            },
        )

    def test_fully_truncated_first_key_recovers_gender(self) -> None:
        content = load_message_content("00002_0271_c1s1_059231_02.json", response_index=1)

        parsed = parse_json_object(content)

        self.assertEqual(parsed["gender"], "male")
        self.assertEqual(parsed["lower_body_clothes_color"], "white")

    def test_regex_fallback_recovers_deeper_truncation(self) -> None:
        content = 'der": "male", "hair": "short", "age": "adult", "hat": "no"}'

        parsed = parse_json_object(content)

        self.assertEqual(
            parsed,
            {
                "hair": "short",
                "age": "adult",
                "hat": "no",
            },
        )

    def test_regex_fallback_rejects_too_few_pairs(self) -> None:
        with self.assertRaises(InvalidFormatError):
            parse_json_object('der": "male"}')

    def test_parse_and_validate_saved_corrupted_response(self) -> None:
        content = load_message_content("00002_0271_c1s1_059231_02.json")

        validated = validate_prediction(parse_json_object(content))

        self.assertEqual(validated["gender"], 1)
        self.assertEqual(validated["age"], 1)
        self.assertEqual(validated["up"], 1)
        self.assertEqual(validated["down_color"], 4)

    def test_build_confusion_report_tracks_top_confusions(self) -> None:
        report = build_confusion_report(
            "age",
            np.asarray([1, 1, 2, 2], dtype=np.int32),
            np.asarray([1, 2, 2, 3], dtype=np.int32),
            {1: "child", 2: "teenager", 3: "adult"},
        )

        self.assertEqual(report["matrix_true_rows_pred_cols"][0], [1, 1, 0])
        self.assertEqual(report["matrix_true_rows_pred_cols"][1], [0, 1, 1])
        self.assertEqual(report["top_confusions"][0]["true_label"], "child")
        self.assertEqual(report["top_confusions"][0]["predicted_label"], "teenager")

    def test_compute_analysis_report_decodes_color_ground_truth(self) -> None:
        gallery = np.asarray(
            [
                [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 5],
                [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 3, 5],
            ],
            dtype=np.int32,
        )
        gt = {
            "age": np.asarray([1, 2], dtype=np.int32),
            "upblack": np.asarray([1, 1], dtype=np.int32),
            "upwhite": np.asarray([2, 1], dtype=np.int32),
            "upred": np.asarray([1, 2], dtype=np.int32),
            "uppurple": np.asarray([1, 1], dtype=np.int32),
            "upgray": np.asarray([1, 1], dtype=np.int32),
            "upblue": np.asarray([1, 1], dtype=np.int32),
            "upgreen": np.asarray([1, 1], dtype=np.int32),
            "upyellow": np.asarray([1, 1], dtype=np.int32),
            "downblack": np.asarray([1, 1], dtype=np.int32),
            "downwhite": np.asarray([1, 1], dtype=np.int32),
            "downpink": np.asarray([1, 1], dtype=np.int32),
            "downgray": np.asarray([1, 1], dtype=np.int32),
            "downblue": np.asarray([2, 2], dtype=np.int32),
            "downgreen": np.asarray([1, 1], dtype=np.int32),
            "downbrown": np.asarray([1, 1], dtype=np.int32),
            "downyellow": np.asarray([1, 1], dtype=np.int32),
            "downpurple": np.asarray([1, 1], dtype=np.int32),
        }

        report = compute_analysis_report(gallery, gt)

        self.assertEqual(report["up_color"]["labels"], ["black", "white", "red", "purple", "gray", "blue", "green", "yellow"])
        self.assertEqual(report["up_color"]["matrix_true_rows_pred_cols"][1][1], 1)
        self.assertEqual(report["up_color"]["matrix_true_rows_pred_cols"][2][2], 1)
        self.assertEqual(report["down_color"]["matrix_true_rows_pred_cols"][4][4], 2)


if __name__ == "__main__":
    unittest.main()
