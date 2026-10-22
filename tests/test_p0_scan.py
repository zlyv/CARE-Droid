from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tasa_droid.config import load_project_config  # noqa: E402
from tasa_droid.p0 import run_p0_scan  # noqa: E402


class P0ScanTest(unittest.TestCase):
    def test_scan_separates_sources_and_finds_exact_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "AndroZoo"
            old_apk = dataset / "old" / "2017" / "malware" / "a.apk"
            new_apk = dataset / "new" / "2018" / "benign" / "b.apk"
            old_apk.parent.mkdir(parents=True)
            new_apk.parent.mkdir(parents=True)
            old_apk.write_bytes(b"same-apk-content")
            new_apk.write_bytes(b"same-apk-content")

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset),
                        "sources": [
                            {
                                "id": "legacy",
                                "purpose": "feasibility_only",
                                "declared_pack_state": "mixed_or_packed",
                                "apk_roots": ["old"],
                            },
                            {
                                "id": "new",
                                "purpose": "formal_candidate_after_p0",
                                "declared_pack_state": "unpacked",
                                "apk_roots": ["new"],
                            },
                        ],
                        "scan": {
                            "extensions": [".apk"],
                            "year_regex": "(?<![0-9])(2017|2018|2019|2020|2021)(?![0-9])",
                            "label_rules": {
                                "malicious": ["(^|[/_-])malware([/_.-]|$)"],
                                "benign": ["(^|[/_-])benign([/_.-]|$)"],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            output = root / "artifacts"
            audit = run_p0_scan(
                load_project_config(config_path),
                output,
                hash_mode="sha256",
                workers=2,
            )

            self.assertEqual(audit["apk_count"], 2)
            self.assertEqual(audit["exact_duplicate_group_count"], 1)
            self.assertEqual(audit["cross_source_exact_duplicate_group_count"], 1)
            self.assertEqual(audit["unresolved_year_count"], 0)
            self.assertEqual(audit["unresolved_label_count"], 0)
            self.assertFalse(audit["formal_training_enabled"])
            self.assertEqual(audit["p0_status"], "NOT_AUDITED")
            self.assertTrue((output / "inventory.csv").is_file())
            self.assertTrue((output / "p0_audit.json").is_file())

    def test_source_root_name_does_not_infer_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "apks_new_nonpacked_2017_2021_b200_t100"
            apk = dataset / "2019" / "sample.apk"
            apk.parent.mkdir(parents=True)
            apk.write_bytes(b"apk")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(root),
                        "sources": [
                            {
                                "id": "new",
                                "purpose": "formal_candidate_after_p0",
                                "declared_pack_state": "unpacked",
                                "apk_roots": [dataset.name],
                            }
                        ],
                        "scan": {
                            "extensions": [".apk"],
                            "label_rules": {
                                "malicious": ["(^|[/_-])t100([/_.-]|$)"],
                                "benign": ["(^|[/_-])b200([/_.-]|$)"],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit = run_p0_scan(load_project_config(config_path), root / "out", workers=1)
            self.assertEqual(audit["unresolved_label_count"], 1)

    def test_year_can_be_inferred_from_year_apk_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            apk_root = root / "year2020"
            apk = apk_root / "benign" / "sample.apk"
            apk.parent.mkdir(parents=True)
            apk.write_bytes(b"apk")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(root),
                        "sources": [
                            {
                                "id": "year_root",
                                "purpose": "feasibility_only",
                                "declared_pack_state": "unknown",
                                "apk_roots": ["year2020"],
                            }
                        ],
                        "scan": {
                            "extensions": [".apk"],
                            "label_rules": {
                                "benign": ["(^|[/_-])benign([/_.-]|$)"]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit = run_p0_scan(load_project_config(config_path), root / "out", workers=1)
            self.assertEqual(audit["unresolved_year_count"], 0)


if __name__ == "__main__":
    unittest.main()
