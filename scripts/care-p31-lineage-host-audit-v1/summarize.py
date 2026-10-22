#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT


def main():
    decision = json.loads((OUTPUT / "final_decision.json").read_text(encoding="utf-8"))
    print((OUTPUT / "report.md").read_text(encoding="utf-8"))
    print("===== final_decision.json =====")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    for name in ("p31_exposure_performance.csv", "host_annotation_feasibility.json", "lineage_summary.json"):
        path = OUTPUT / name
        print(f"===== {name} =====")
        print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
