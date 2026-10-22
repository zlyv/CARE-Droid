#!/usr/bin/env python3
"""Pure contract tests for the capacity-matched block masks."""

from masked_host_model import ARMS, BLOCKS, correction_input_mask


def main() -> None:
    assert list(BLOCKS.values()) == [(0, 24), (24, 25), (25, 57), (57, 81), (81, 105), (105, 113)]
    assert int(correction_input_mask("FULL").sum()) == 113
    expected = {
        "DROP_GLOBAL24": 89, "DROP_STAGE1_LOGIT": 112,
        "DROP_ROLE_GRAPH": 81, "DROP_CRITICAL_HOST": 89,
        "DROP_ATTENDED_HOST": 89, "DROP_SCALAR_SUMMARY": 105,
        "STAGE1_CONTEXT_ONLY": 25, "STRUCTURE_ONLY": 88,
    }
    for arm, active in expected.items():
        mask = correction_input_mask(arm)
        assert mask.numel() == 113 and int(mask.sum()) == active
        assert set(mask.tolist()).issubset({0.0, 1.0})
    assert len(ARMS) == 9
    print("HOST_INPUT_ABLATION_SELFTEST_PASS")


if __name__ == "__main__":
    main()

