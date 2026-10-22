#!/usr/bin/env python3
import numpy as np

from prompt_v2_core import MODEL_ROLE_ORDER, reconstruct_primary_graph, render_user_prompt, summarize_evidence


def main():
    primary = np.asarray([3, 1, 17], dtype=np.int64)
    offsets = np.asarray([0, 1, 2, 2], dtype=np.int64)
    positions = np.asarray([1, 2], dtype=np.int64)
    matrix, edges = reconstruct_primary_graph(primary, offsets, positions)
    assert edges == {(0, 1), (1, 2)}
    assert matrix[3, 1] == 1 and matrix[1, 17] == 1 and matrix.sum() == 2
    roles = np.zeros(31, dtype=np.int64); roles[[1, 3, 17, 25]] = [1, 2, 3, 4]
    summary = {
        "entry_reachable_selected_endpoint_count": 3,
        "entry_unreachable_selected_endpoint_count": 7,
        "active_node_provenance_counts": {
            "BUNDLED_NON_PACKAGE": 8, "EXTERNAL_OR_FRAMEWORK": 2, "PACKAGE_ALIGNED": 0,
        },
    }
    evidence = summarize_evidence(roles, matrix, summary)
    prompt = render_user_prompt(0, 1, evidence)
    assert "command_execution" not in prompt
    assert "reflective_invocation -> dynamic_code_loading" in prompt
    assert "30.0%" in prompt and "70.0%" in prompt
    assert "BUNDLED_NON_PACKAGE: 8" not in prompt
    assert "FIXED" not in prompt and "BROKEN" not in prompt
    assert len(MODEL_ROLE_ORDER) == len(set(MODEL_ROLE_ORDER)) == 31
    print("LLM_PROMPT_V2_SELFTEST_PASS")
    print("llm_calls_performed = 0")


if __name__ == "__main__":
    main()
