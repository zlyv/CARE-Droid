import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_year_split_manifest.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_year_split_manifest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FixedTrainingYearSplitTest(unittest.TestCase):
    def test_fixed_training_years_do_not_accumulate_later_years(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            feature_path = tmp_path / "features.csv"
            blocking_path = tmp_path / "blocking.csv"
            output_path = tmp_path / "output"

            feature_rows = [
                {"sha256": "a", "metadata_year": "2017", "metadata_label": "malicious"},
                {"sha256": "b", "metadata_year": "2017", "metadata_label": "benign"},
                {"sha256": "c", "metadata_year": "2018", "metadata_label": "malicious"},
                {"sha256": "d", "metadata_year": "2018", "metadata_label": "benign"},
                {"sha256": "e", "metadata_year": "2019", "metadata_label": "malicious"},
                {"sha256": "f", "metadata_year": "2019", "metadata_label": "benign"},
            ]
            blocking_rows = [
                {"sha256": "a", "block_id": "g_a"},
                {"sha256": "b", "block_id": "g_b"},
                {"sha256": "c", "block_id": "g_c"},
                {"sha256": "d", "block_id": "g_d"},
                {"sha256": "e", "block_id": "g_e"},
                {"sha256": "f", "block_id": "g_b"},
            ]
            write_csv(feature_path, feature_rows, ["sha256", "metadata_year", "metadata_label"])
            write_csv(blocking_path, blocking_rows, ["sha256", "block_id"])

            module = load_script()
            original_argv = sys.argv[:]
            sys.argv = [
                "run_year_split_manifest.py",
                "--features",
                str(feature_path),
                "--blocking",
                str(blocking_path),
                "--output",
                str(output_path),
                "--fixed-train-years",
                "2017",
                "--test-years",
                "2018",
                "2019",
            ]
            try:
                module.main()
            finally:
                sys.argv = original_argv

            audit = json.loads((output_path / "split_manifest_audit.json").read_text())
            self.assertEqual(audit["training_policy"], "fixed_years")
            self.assertEqual(audit["fixed_train_years"], [2017])
            self.assertEqual([item["train_count"] for item in audit["split_summaries"]], [2, 2])

            with (output_path / "split_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

            split_2019 = [
                row for row in rows if row["split_name"] == "train_fixed_2017__test_2019"
            ]
            self.assertEqual(
                {row["sha256"] for row in split_2019 if row["partition"] == "train"},
                {"a", "b"},
            )
            self.assertEqual(
                {row["sha256"] for row in split_2019 if row["partition"] == "test"},
                {"e"},
            )
            self.assertEqual(
                {row["sha256"] for row in split_2019 if row["partition"] == "excluded"},
                {"f"},
            )
