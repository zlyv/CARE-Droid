#!/usr/bin/env python3
"""Synthetic static-analysis and statistical-contract regressions; no APK execution."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from argument_core import (analyze_method, command_effects, legacy_id, aggregate_events,
                           RUNTIME_REFS, MAX_STRING)
from argument_extract import decode_method, scan_one, validate_cache
from argument_probe import (feature_vector, fit_offset, row_shuffle, arm_features,
                            paired_test, adjust_holm, WIDTH, EFFECT_START, EFFECT_END, COARSE_END)
from argument_io import source_rows, write_json, read_json

EXEC = "Ljava/lang/Runtime;->exec(Ljava/lang/String;)Ljava/lang/Process;"
EXEC_ARRAY = "Ljava/lang/Runtime;->exec([Ljava/lang/String;)Ljava/lang/Process;"
PB = "Ljava/lang/ProcessBuilder;"
SB = "Ljava/lang/StringBuilder;"


def rows(*values):
    return [dict(offset=i * 2, **value) for i, value in enumerate(values)]


def const(register, text):
    return {"name": "const-string", "registers": [register], "string": text}


def invoke(reference=EXEC, registers=(0, 1), name="invoke-virtual"):
    return {"name": name, "reference": reference, "registers": list(registers)}


def first(*instructions, **kwargs):
    result = analyze_method(rows(*instructions), **kwargs)
    return result["events"][0]


class StaticArgumentTests(unittest.TestCase):
    def test_same_api_coarse_category_but_distinct_effect(self):
        query = first(const(1, "logcat -d"), invoke())
        privilege = first(const(1, "su -c id"), invoke())
        self.assertEqual(query["coarse_categories"], privilege["coarse_categories"])
        self.assertEqual(query["endpoint_id"], privilege["endpoint_id"])
        self.assertEqual(query["effects"], ["QUERY_COMMAND"])
        self.assertEqual(privilege["effects"], ["PRIVILEGE_HELPER"])

    def test_no_literal_export(self):
        event = first(const(1, "cat /data/private-token-ABC"), invoke())
        self.assertNotIn("private-token", json.dumps(event))
        self.assertNotIn("cat ", json.dumps(event))

    def test_framework_dispatch_is_case_sensitive(self):
        result = analyze_method(rows(const(1, "id"), invoke(EXEC.replace("Runtime", "runtime"))))
        self.assertEqual(result["events"], [])
        self.assertEqual(command_effects(["SU"])[0], ("OTHER_PROGRAM",))

    def test_runtime_string_is_not_a_shell(self):
        event = first(const(1, "echo hi; rm /tmp/not-executed"), invoke())
        self.assertEqual(event["effects"], ["OUTPUT_COMMAND"])

    def test_java_tokenizer_does_not_use_unicode_whitespace(self):
        event = first(const(1, "id\u00a0something"), invoke())
        self.assertEqual(event["effects"], ["UNRESOLVED_COMMAND"])

    def test_shell_simple_command(self):
        self.assertEqual(set(command_effects(["sh", "-c", "id"])[0]), {"SHELL_WRAPPER", "QUERY_COMMAND"})

    def test_shell_compound_and_expansion_are_unknown(self):
        for body in ("id; rm -rf /not-run", "echo $USER", "cat *.txt", "cat a >b", "id && ps"):
            effects, reason = command_effects(["sh", "-c", body])
            self.assertIn("UNRESOLVED_COMMAND", effects)
            self.assertIsNotNone(reason)

    def test_runtime_quotes_not_shell_quotes(self):
        event = first(const(1, "sh -c 'id -u'"), invoke())
        self.assertIn("UNRESOLVED_COMMAND", event["effects"])

    def test_shell_can_parse_one_quoted_word(self):
        event = first(const(1, "sh -c 'id'"), invoke())
        self.assertIn("QUERY_COMMAND", event["effects"])

    def test_wide_destination_kills_both_register_words(self):
        for write in ({"name": "const-wide/16", "registers": [1], "literal": 7},
                      {"name": "move-wide", "registers": [1, 9]},
                      {"name": "move-result-wide", "registers": [1]}):
            event = first(const(2, "id"), write, invoke(EXEC, (0, 2)))
            self.assertFalse(event["resolved_argv"])

    def test_truncated_invoke_register_list_stays_unknown(self):
        event = first(const(1, "id"), invoke(EXEC.replace(";)Ljava", ";[Ljava/lang/String;)Ljava")))
        self.assertFalse(event["resolved_argv"])

    def test_filled_array_and_shell(self):
        event = first(const(1, "sh"), const(2, "-c"), const(3, "chmod 600 file"),
                      {"name": "filled-new-array", "type": "[Ljava/lang/String;", "registers": [1, 2, 3]},
                      {"name": "move-result-object", "registers": [4]}, invoke(EXEC_ARRAY, (0, 4)))
        self.assertEqual(set(event["effects"]), {"SHELL_WRAPPER", "PERMISSION_CHANGE"})

    def test_array_alias_write(self):
        event = first({"name": "const/4", "registers": [1], "literal": 1},
                      {"name": "new-array", "registers": [2, 1], "type": "[Ljava/lang/String;"},
                      {"name": "move-object", "registers": [3, 2]}, const(4, "id"),
                      {"name": "const/4", "registers": [5], "literal": 0},
                      {"name": "aput-object", "registers": [4, 3, 5]}, invoke(EXEC_ARRAY, (0, 2)))
        self.assertEqual(event["effects"], ["QUERY_COMMAND"])

    def test_array_unknown_write_invalidates(self):
        event = first(const(1, "id"),
                      {"name": "filled-new-array", "type": "[Ljava/lang/String;", "registers": [1]},
                      {"name": "move-result-object", "registers": [2]},
                      {"name": "aput-object", "registers": [8, 2, 9]}, invoke(EXEC_ARRAY, (0, 2)))
        self.assertFalse(event["resolved_argv"])

    def test_process_builder(self):
        event = first(const(1, "id"),
                      {"name": "filled-new-array", "type": "[Ljava/lang/String;", "registers": [1]},
                      {"name": "move-result-object", "registers": [2]},
                      {"name": "new-instance", "type": PB, "registers": [3]},
                      invoke(PB + "-><init>([Ljava/lang/String;)V", (3, 2), "invoke-direct"),
                      invoke(PB + "->start()Ljava/lang/Process;", (3,)))
        self.assertEqual(event["channel"], "PROCESS_BUILDER")
        self.assertEqual(event["effects"], ["QUERY_COMMAND"])

    def test_string_builder(self):
        event = first({"name": "new-instance", "registers": [2], "type": SB},
                      invoke(SB + "-><init>()V", (2,), "invoke-direct"),
                      const(3, "ch"), invoke(SB + "->append(Ljava/lang/String;)Ljava/lang/StringBuilder;", (2, 3)),
                      {"name": "move-result-object", "registers": [4]}, const(3, "mod 600 file"),
                      invoke(SB + "->append(Ljava/lang/String;)Ljava/lang/StringBuilder;", (4, 3)),
                      invoke(SB + "->toString()Ljava/lang/String;", (2,)),
                      {"name": "move-result-object", "registers": [1]}, invoke())
        self.assertEqual(event["effects"], ["PERMISSION_CHANGE"])

    def test_string_concat(self):
        event = first(const(1, "get"), const(2, "prop"),
                      invoke("Ljava/lang/String;->concat(Ljava/lang/String;)Ljava/lang/String;", (1, 2)),
                      {"name": "move-result-object", "registers": [1]}, invoke())
        self.assertEqual(event["effects"], ["QUERY_COMMAND"])

    def test_unknown_call_invalidates_mutable_argument(self):
        event = first(const(1, "id"),
                      {"name": "filled-new-array", "type": "[Ljava/lang/String;", "registers": [1]},
                      {"name": "move-result-object", "registers": [2]},
                      invoke("Lapp/X;->mutate([Ljava/lang/String;)V", (2,), "invoke-static"),
                      invoke(EXEC_ARRAY, (0, 2)))
        self.assertFalse(event["resolved_argv"])

    def test_unknown_result_kills_previous_literal(self):
        event = first(const(1, "id"), invoke("Lapp/X;->unknown()Ljava/lang/String;", (), "invoke-static"),
                      {"name": "move-result-object", "registers": [1]}, invoke())
        self.assertFalse(event["resolved_argv"])

    def test_cfg_join_does_not_reuse_fallthrough_constant(self):
        event = first(const(1, "su"), {"name": "if-eqz", "registers": [9], "target": 8},
                      const(1, "id"), {"name": "goto", "target": 8}, invoke())
        self.assertFalse(event["resolved_argv"])

    def test_branch_target_without_control_instruction_at_join(self):
        # The incoming edge targets a nop, not the invoke itself.
        event = first(const(1, "su"), {"name": "if-eqz", "registers": [9], "target": 8},
                      const(1, "id"), {"name": "nop"}, {"name": "nop"}, invoke())
        self.assertFalse(event["resolved_argv"])

    def test_exception_handler_is_a_fresh_block(self):
        event = first(const(1, "id"), {"name": "nop"}, invoke(),
                      exception_regions=[{"start_offset_bytes": 0, "end_offset_bytes": 4,
                                          "handlers": [{"target_offset_bytes": 4}]}])
        self.assertFalse(event["resolved_argv"])

    def test_unresolved_exception_table_does_not_claim_resolution(self):
        result = analyze_method(rows(const(1, "id"), invoke()), exception_error="invalid")
        self.assertEqual(result["events"], [])
        self.assertEqual(result["status"], "PARTIAL")

    def test_missing_branch_target(self):
        result = analyze_method(rows({"name": "if-eqz", "registers": [0]}, const(1, "id"), invoke()))
        self.assertIn("CFG_TARGET_UNRESOLVED", result["unknown_reasons"])

    def test_unmodeled_write_is_a_barrier(self):
        event = first(const(1, "id"), {"name": "iget-object", "registers": [1, 9]}, invoke())
        self.assertFalse(event["resolved_argv"])

    def test_frozen_endpoint_filter(self):
        result = analyze_method(rows(const(1, "id"), invoke()), frozen_targets={"other"})
        self.assertEqual(result["events"], [])

    def test_literal_and_method_caps(self):
        self.assertFalse(first(const(1, "x" * (MAX_STRING + 1)), invoke())["resolved_argv"])
        result = analyze_method(rows(const(1, "id"), invoke()), max_instructions=1)
        self.assertEqual(result["status"], "RESOURCE_LIMIT")

    def test_unknown_and_known_events_have_separate_counts(self):
        record = aggregate_events([first(const(1, "id"), invoke()), first(invoke())])
        self.assertEqual((record["events"], record["resolved"]), (2, 1))
        self.assertEqual(record["effects"]["UNRESOLVED_COMMAND"], 1)


class ReadoutAndContractTests(unittest.TestCase):
    def test_feature_schema_and_same_coarse_different_semantics(self):
        def vector(command):
            event = first(const(1, command), invoke())
            return feature_vector({"summary": aggregate_events([event]), "hosts": []})
        a, b = vector("id"), vector("su")
        self.assertEqual(a.shape, (WIDTH,))
        np.testing.assert_array_equal(a[:EFFECT_START], b[:EFFECT_START])
        self.assertFalse(np.array_equal(a[EFFECT_START:], b[EFFECT_START:]))

    def test_row_control_deranges_apks_and_preserves_multiset(self):
        n = 12
        context = np.zeros((n, WIDTH))
        context[:, 0] = 1; context[:, 6] = 1; context[:, COARSE_END] = 1
        evidence = np.arange(n * 3).reshape(n, 3)
        shuffled, meta = row_shuffle(evidence, context, list(map(str, range(n))), 7)
        self.assertEqual(meta["changed_fraction"], 1.)
        self.assertEqual(sorted(map(tuple, evidence)), sorted(map(tuple, shuffled)))

    def test_null_train_and_target_do_not_share_donors(self):
        a = np.ones((6, WIDTH)); b = np.ones((4, WIDTH)) * 3
        _, _, meta = arm_features("EFFECT_SHUFFLE_0", a, b, list("abcdef"), list("wxyz"), 3)
        self.assertTrue(all(i < 6 for i in meta["train_null"]["donor_indices"]))
        self.assertTrue(all(i < 4 for i in meta["target_null"]["donor_indices"]))

    def test_offset_logistic_learns_new_evidence(self):
        x = np.linspace(-2, 2, 40)[:, None]
        y = (x[:, 0] > 0).astype(int)
        pred, fit = fit_offset(x, y, np.zeros(40), x, np.zeros(40))
        self.assertGreater(np.mean((pred >= .5) == y), .95)
        self.assertTrue(np.isfinite(fit["loss"]))

    def test_scaler_and_fit_do_not_learn_target_distribution(self):
        x = np.arange(30).reshape(10, 3).astype(float); y = np.array([0, 1] * 5)
        _, one = fit_offset(x, y, np.zeros(10), x, np.zeros(10))
        _, two = fit_offset(x, y, np.zeros(10), x + 10000, np.zeros(10))
        np.testing.assert_array_equal(one["mean"], two["mean"])
        np.testing.assert_allclose(one["weights"], two["weights"], rtol=0, atol=0)

    def test_paired_identical_predictions_are_not_supported(self):
        y = np.array([0, 0, 1, 1]); p = np.array([[.1, .7, .9, .4]] * 3)
        result = paired_test(y, p, p, 7, repeats=100)
        self.assertEqual(result["BA_difference"], 0.)
        self.assertEqual(result["p_one_sided"], 1.)
        self.assertEqual(result["ci95_low"], 0.)

    def test_holm_and_pseudocount(self):
        rows0 = [{"p_one_sided": .01}, {"p_one_sided": .03}, {"p_one_sided": .6}]
        adjust_holm(rows0)
        self.assertEqual(rows0[0]["p_holm"], .03)
        self.assertEqual(rows0[1]["p_holm"], .06)

    def test_future_rows_are_rejected_before_cohort_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "source_2017").mkdir()
            (root / "source_2017/source_oof_predictions.csv").write_text("year,arm\n2018,B0_W0\n")
            with self.assertRaisesRegex(ValueError, "non-2017"):
                source_rows(root)

    def test_existing_nonidentical_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json.gz"
            write_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"value": 2})
            self.assertEqual(read_json(path), {"value": 1})

    def test_cache_content_changes_are_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"SYNTHETIC_NO_TARGET_BYTE_FIXTURE"
            sid = hashlib.sha256(data).hexdigest()
            apk = root / "no-target.fixture"
            apk.write_bytes(data)
            view = root / "view.json"
            write_json(view, {"sha256": sid, "year": 2017, "host_ids": [], "pairs_by_host": {}})
            cache = root / "cache.json.gz"
            scan_one(sid, apk, view, cache, "TEST_CONTRACT")
            record = validate_cache(cache, sid, view, "TEST_CONTRACT")
            record["summary"]["events"] = 1
            write_json(cache, record, replace=True)
            with self.assertRaisesRegex(ValueError, "stale or foreign"):
                validate_cache(cache, sid, view, "TEST_CONTRACT")
            self.assertEqual(apk.read_bytes(), data)

    def test_range_decoder_expands_all_registers_and_keeps_case(self):
        class Ins:
            AA, CCCC = 2, 4
            def get_name(self): return "invoke-virtual/range"
            def get_operands(self): return [(0, 4), (256, 1, EXEC)]
            def get_output(self): return "v4 ... v5, " + EXEC
        class Code:
            def get_tries_size(self): return 0
        class Method:
            def get_code(self): return Code()
            def get_instructions_idx(self): return [(0, Ins())]
        decoded, regions, error = decode_method(Method())
        self.assertEqual(decoded[0]["registers"], [4, 5])
        self.assertEqual(decoded[0]["reference"], EXEC)
        self.assertEqual(regions, [])
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
