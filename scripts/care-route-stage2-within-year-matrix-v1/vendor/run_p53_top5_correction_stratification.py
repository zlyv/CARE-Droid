#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


TARGET_VARIANT = "compact_plus_global_support_error_guided_detail"
ROLE_PREFIX = "role_node__"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value):
    return int(float(value))


def year_from_split(split_name: str) -> str:
    m = re.search(r"test_(\d{4})", split_name)
    return m.group(1) if m else ""


def unique_keep_order(values):
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def role_from_item(item, available_roles):
    if isinstance(item, str):
        value = item.strip()
        if value in available_roles:
            return value
        return None

    if isinstance(item, dict):
        for key in (
            "role",
            "semantic_role",
            "role_name",
            "name",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.strip() in available_roles:
                return value.strip()

    return None


def roles_from_list(value, available_roles):
    if not isinstance(value, list):
        return []

    roles = []
    for item in value:
        role = role_from_item(item, available_roles)
        if role is not None:
            roles.append(role)
    return unique_keep_order(roles)


def load_selected_roles(path: Path, available_roles, k: int):
    payload = json.loads(path.read_text(encoding="utf-8"))

    # Most likely layouts first.
    if isinstance(payload, list):
        roles = roles_from_list(payload, available_roles)
        if len(roles) >= k:
            return roles[:k], "top_level_list"

    if isinstance(payload, dict):
        preferred_keys = (
            "error_guided",
            "selected_roles",
            "top_roles",
            "top_k_roles",
            "topk_roles",
            "selected",
            "roles_selected",
        )
        for key in preferred_keys:
            if key in payload:
                roles = roles_from_list(payload[key], available_roles)
                if len(roles) >= k:
                    return roles[:k], key

        # Search nested dictionaries for explicitly selected/top lists.
        queue = [payload]
        while queue:
            obj = queue.pop(0)
            if not isinstance(obj, dict):
                continue

            for key, value in obj.items():
                key_lower = str(key).lower()
                if isinstance(value, list) and (
                    "selected" in key_lower or
                    "top" in key_lower
                ):
                    roles = roles_from_list(value, available_roles)
                    if len(roles) >= k:
                        return roles[:k], f"nested:{key}"

                if isinstance(value, dict):
                    queue.append(value)

        # Ranked-list fallback.
        for key in (
            "ranked_roles",
            "ranking",
            "role_ranking",
            "ranked",
        ):
            if key in payload:
                roles = roles_from_list(payload[key], available_roles)
                if len(roles) >= k:
                    return roles[:k], f"ranked_fallback:{key}"

    raise RuntimeError(
        f"Could not recover {k} selected semantic roles from {path}. "
        "Please inspect the JSON or pass --role five times explicitly."
    )


class Confusion:
    def __init__(self):
        self.n = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, label, pred):
        self.n += 1
        if label == 1 and pred == 1:
            self.tp += 1
        elif label == 0 and pred == 0:
            self.tn += 1
        elif label == 0 and pred == 1:
            self.fp += 1
        elif label == 1 and pred == 0:
            self.fn += 1
        else:
            raise ValueError((label, pred))

    def accuracy(self):
        if self.n == 0:
            return float("nan")
        return (self.tp + self.tn) / self.n

    def error_rate(self):
        acc = self.accuracy()
        return 1.0 - acc if not math.isnan(acc) else float("nan")

    def tpr(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    def tnr(self):
        denom = self.tn + self.fp
        return self.tn / denom if denom else float("nan")

    def balanced_accuracy(self):
        tpr = self.tpr()
        tnr = self.tnr()
        if math.isnan(tpr) or math.isnan(tnr):
            return float("nan")
        return 0.5 * (tpr + tnr)


class GroupStats:
    def __init__(self):
        self.compact = Confusion()
        self.p53 = Confusion()

        self.corrected = 0
        self.regressed = 0
        self.still_correct = 0
        self.still_wrong = 0

        self.prob_delta_sum = 0.0
        self.abs_prob_delta_sum = 0.0

    @property
    def n(self):
        return self.compact.n

    def add(
        self,
        label,
        compact_pred,
        p53_pred,
        compact_probability,
        p53_probability,
    ):
        self.compact.add(label, compact_pred)
        self.p53.add(label, p53_pred)

        compact_correct = compact_pred == label
        p53_correct = p53_pred == label

        if not compact_correct and p53_correct:
            self.corrected += 1
        elif compact_correct and not p53_correct:
            self.regressed += 1
        elif compact_correct and p53_correct:
            self.still_correct += 1
        else:
            self.still_wrong += 1

        delta = p53_probability - compact_probability
        self.prob_delta_sum += delta
        self.abs_prob_delta_sum += abs(delta)

    def row(self):
        n = self.n
        compact_errors = self.corrected + self.still_wrong
        compact_corrects = self.regressed + self.still_correct

        compact_ba = self.compact.balanced_accuracy()
        p53_ba = self.p53.balanced_accuracy()
        compact_acc = self.compact.accuracy()
        p53_acc = self.p53.accuracy()

        return {
            "apk_count": n,

            "compact_accuracy": compact_acc,
            "p53_accuracy": p53_acc,
            "p53_minus_compact_accuracy_pp":
                (p53_acc - compact_acc) * 100.0
                if not math.isnan(compact_acc) and not math.isnan(p53_acc)
                else "",

            "compact_balanced_accuracy": compact_ba,
            "p53_balanced_accuracy": p53_ba,
            "p53_minus_compact_ba_pp":
                (p53_ba - compact_ba) * 100.0
                if not math.isnan(compact_ba) and not math.isnan(p53_ba)
                else "",

            "compact_error_rate": self.compact.error_rate(),
            "p53_error_rate": self.p53.error_rate(),

            "compact_tp": self.compact.tp,
            "compact_tn": self.compact.tn,
            "compact_fp": self.compact.fp,
            "compact_fn": self.compact.fn,

            "p53_tp": self.p53.tp,
            "p53_tn": self.p53.tn,
            "p53_fp": self.p53.fp,
            "p53_fn": self.p53.fn,

            "corrected": self.corrected,
            "regressed": self.regressed,
            "still_correct": self.still_correct,
            "still_wrong": self.still_wrong,
            "net_correction": self.corrected - self.regressed,

            "correction_rate_among_compact_errors":
                self.corrected / compact_errors
                if compact_errors else "",

            "regression_rate_among_compact_correct":
                self.regressed / compact_corrects
                if compact_corrects else "",

            "net_correction_per_apk":
                (self.corrected - self.regressed) / n
                if n else "",

            "mean_probability_shift":
                self.prob_delta_sum / n if n else "",

            "mean_absolute_probability_shift":
                self.abs_prob_delta_sum / n if n else "",
        }


def transition_name(label, compact_pred, p53_pred):
    compact_correct = compact_pred == label
    p53_correct = p53_pred == label

    if compact_correct and p53_correct:
        return "STILL_CORRECT"
    if not compact_correct and p53_correct:
        return "CORRECTED"
    if compact_correct and not p53_correct:
        return "REGRESSED"
    return "STILL_WRONG"


def write_group_summary(path, groups, dimension_name):
    rows = []

    for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        scope, split_name, year, dimension_value = key
        stats = groups[key]

        row = {
            "scope": scope,
            "split_name": split_name,
            "year": year,
            dimension_name: dimension_value,
        }
        row.update(stats.row())
        rows.append(row)

    fields = [
        "scope",
        "split_name",
        "year",
        dimension_name,
        "apk_count",

        "compact_accuracy",
        "p53_accuracy",
        "p53_minus_compact_accuracy_pp",

        "compact_balanced_accuracy",
        "p53_balanced_accuracy",
        "p53_minus_compact_ba_pp",

        "compact_error_rate",
        "p53_error_rate",

        "compact_tp",
        "compact_tn",
        "compact_fp",
        "compact_fn",

        "p53_tp",
        "p53_tn",
        "p53_fp",
        "p53_fn",

        "corrected",
        "regressed",
        "still_correct",
        "still_wrong",
        "net_correction",

        "correction_rate_among_compact_errors",
        "regression_rate_among_compact_correct",
        "net_correction_per_apk",

        "mean_probability_shift",
        "mean_absolute_probability_shift",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_headline(path: Path, target_variant: str):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == target_variant:
                return row
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/p1-features/features.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "artifacts/p53-global-coarse-selective-context/"
            "p53_predictions.csv"
        ),
    )
    parser.add_argument(
        "--selected-roles",
        type=Path,
        default=Path(
            "artifacts/p51-error-guided-entry/selected_roles.json"
        ),
    )
    parser.add_argument(
        "--headline",
        type=Path,
        default=Path(
            "artifacts/p53-global-coarse-selective-context/"
            "p53_headline_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/p53-top5-correction-stratification"
        ),
    )
    parser.add_argument(
        "--variant",
        default=TARGET_VARIANT,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Optional explicit selected role; repeat exactly top-k times.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    for path in (
        args.features,
        args.predictions,
        args.selected_roles,
    ):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    csv.field_size_limit(sys.maxsize)

    # ---------------------------------------------------------
    # 1. Inspect Compact feature header to get semantic-role vocabulary.
    # ---------------------------------------------------------
    with args.features.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        feature_header = next(reader)

    available_roles = {
        column[len(ROLE_PREFIX):]
        for column in feature_header
        if column.startswith(ROLE_PREFIX)
    }

    if not available_roles:
        raise SystemExit(
            f"No {ROLE_PREFIX}<role> columns found in {args.features}"
        )

    if args.role:
        selected_roles = unique_keep_order(args.role)
        parser_source = "explicit_cli_roles"

        if len(selected_roles) != args.top_k:
            raise SystemExit(
                f"--role supplied {len(selected_roles)} roles; "
                f"expected exactly {args.top_k}"
            )

        unknown = [
            role for role in selected_roles
            if role not in available_roles
        ]
        if unknown:
            raise SystemExit(f"Unknown roles in --role: {unknown}")
    else:
        selected_roles, parser_source = load_selected_roles(
            args.selected_roles,
            available_roles,
            args.top_k,
        )

    print("Selected Top roles:")
    for i, role in enumerate(selected_roles, 1):
        print(f"  {i}. {role}")
    print(f"role parser source: {parser_source}", flush=True)

    # ---------------------------------------------------------
    # 2. Read only the target P53 variant predictions.
    #    This is small: roughly the four future-year test sets.
    # ---------------------------------------------------------
    predictions = {}
    target_shas = set()

    with args.predictions.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "split_name",
            "sha256",
            "variant",
            "label",
            "compact_probability",
            "variant_probability",
            "compact_predicted_label",
            "variant_predicted_label",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"prediction CSV missing columns: {sorted(missing)}"
            )

        for row in reader:
            if row["variant"] != args.variant:
                continue

            split_name = row["split_name"]
            sha = row["sha256"].strip().lower()
            key = (split_name, sha)

            if key in predictions:
                raise SystemExit(
                    f"duplicate target prediction row: {key}"
                )

            predictions[key] = {
                "split_name": split_name,
                "year": year_from_split(split_name),
                "sha256": sha,
                "label": safe_int(row["label"]),
                "compact_probability":
                    safe_float(row["compact_probability"]),
                "p53_probability":
                    safe_float(row["variant_probability"]),
                "compact_pred":
                    safe_int(row["compact_predicted_label"]),
                "p53_pred":
                    safe_int(row["variant_predicted_label"]),
            }
            target_shas.add(sha)

    if not predictions:
        raise SystemExit(
            f"No rows found for variant={args.variant!r}"
        )

    print(
        f"target prediction rows: {len(predictions)} "
        f"unique APKs: {len(target_shas)}",
        flush=True,
    )

    # ---------------------------------------------------------
    # 3. Stream Compact features once.
    #    For target APKs only, record presence of the Top-K roles.
    # ---------------------------------------------------------
    role_columns = {
        role: f"{ROLE_PREFIX}{role}"
        for role in selected_roles
    }

    feature_presence = {}

    with args.features.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)

        if "sha256" not in (reader.fieldnames or []):
            raise SystemExit("features.csv has no sha256 column")

        missing_role_columns = [
            column for column in role_columns.values()
            if column not in (reader.fieldnames or [])
        ]
        if missing_role_columns:
            raise SystemExit(
                "missing selected role columns: "
                f"{missing_role_columns}"
            )

        for row in reader:
            sha = row["sha256"].strip().lower()
            if sha not in target_shas:
                continue

            if sha in feature_presence:
                raise SystemExit(
                    f"duplicate Compact feature row for sha={sha}"
                )

            present_roles = [
                role
                for role, column in role_columns.items()
                if safe_float(row.get(column, 0.0)) > 0.0
            ]

            feature_presence[sha] = present_roles

            if len(feature_presence) == len(target_shas):
                break

    missing_feature_shas = sorted(
        target_shas - set(feature_presence)
    )
    if missing_feature_shas:
        raise SystemExit(
            f"{len(missing_feature_shas)} prediction APKs are missing "
            f"from Compact features; first={missing_feature_shas[:5]}"
        )

    # ---------------------------------------------------------
    # 4. Stratify correction behavior.
    # ---------------------------------------------------------
    presence_groups = defaultdict(GroupStats)
    count_groups = defaultdict(GroupStats)
    role_groups = defaultdict(GroupStats)
    split_groups = defaultdict(GroupStats)

    apk_rows = []

    for key in sorted(predictions):
        pred = predictions[key]
        sha = pred["sha256"]
        split_name = pred["split_name"]
        year = pred["year"]

        present_roles = feature_presence[sha]
        top5_count = len(present_roles)
        top5_presence = "PRESENT" if top5_count > 0 else "ABSENT"

        label = pred["label"]
        compact_pred = pred["compact_pred"]
        p53_pred = pred["p53_pred"]
        compact_prob = pred["compact_probability"]
        p53_prob = pred["p53_probability"]

        transition = transition_name(
            label,
            compact_pred,
            p53_pred,
        )

        # Per split.
        split_groups[split_name].add(
            label,
            compact_pred,
            p53_pred,
            compact_prob,
            p53_prob,
        )

        # Top-5 present/absent: per year and pooled.
        for scope, sname, y in (
            ("PER_SPLIT", split_name, year),
            ("ALL_FUTURE", "ALL", "ALL"),
        ):
            presence_groups[
                (scope, sname, y, top5_presence)
            ].add(
                label,
                compact_pred,
                p53_pred,
                compact_prob,
                p53_prob,
            )

            count_groups[
                (scope, sname, y, str(top5_count))
            ].add(
                label,
                compact_pred,
                p53_pred,
                compact_prob,
                p53_prob,
            )

        # Specific selected role present.
        for role in present_roles:
            for scope, sname, y in (
                ("PER_SPLIT", split_name, year),
                ("ALL_FUTURE", "ALL", "ALL"),
            ):
                role_groups[
                    (scope, sname, y, role)
                ].add(
                    label,
                    compact_pred,
                    p53_pred,
                    compact_prob,
                    p53_prob,
                )

        apk_rows.append({
            "split_name": split_name,
            "year": year,
            "sha256": sha,
            "label": label,
            "compact_probability": compact_prob,
            "p53_probability": p53_prob,
            "compact_predicted_label": compact_pred,
            "p53_predicted_label": p53_pred,
            "transition": transition,
            "top5_count": top5_count,
            "top5_present": int(top5_count > 0),
            "present_top5_roles": "|".join(present_roles),
        })

    # ---------------------------------------------------------
    # 5. Write outputs.
    # ---------------------------------------------------------
    apk_path = args.output / "p53_top5_apk_level.csv"
    with apk_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "split_name",
            "year",
            "sha256",
            "label",
            "compact_probability",
            "p53_probability",
            "compact_predicted_label",
            "p53_predicted_label",
            "transition",
            "top5_count",
            "top5_present",
            "present_top5_roles",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(apk_rows)

    presence_path = (
        args.output / "p53_top5_presence_summary.csv"
    )
    count_path = (
        args.output / "p53_top5_count_summary.csv"
    )
    role_path = (
        args.output / "p53_top5_role_summary.csv"
    )

    write_group_summary(
        presence_path,
        presence_groups,
        "top5_presence",
    )
    write_group_summary(
        count_path,
        count_groups,
        "top5_count",
    )
    write_group_summary(
        role_path,
        role_groups,
        "semantic_role",
    )

    # ---------------------------------------------------------
    # 6. Reproduce P53 headline rescue/harm/net as a QC.
    # ---------------------------------------------------------
    split_qc = []
    for split_name in sorted(split_groups):
        stats = split_groups[split_name]
        row = {
            "split_name": split_name,
            "year": year_from_split(split_name),
            "apk_count": stats.n,
        }
        row.update(stats.row())
        split_qc.append(row)

    observed_rescue_mean = (
        sum(row["corrected"] for row in split_qc)
        / len(split_qc)
    )
    observed_harm_mean = (
        sum(row["regressed"] for row in split_qc)
        / len(split_qc)
    )
    observed_net_mean = (
        sum(row["net_correction"] for row in split_qc)
        / len(split_qc)
    )

    observed_ba_mean = (
        sum(
            row["p53_balanced_accuracy"]
            for row in split_qc
            if row["p53_balanced_accuracy"] != ""
            and not math.isnan(row["p53_balanced_accuracy"])
        )
        / len(split_qc)
    )

    headline = read_headline(
        args.headline,
        args.variant,
    )

    headline_check = None
    if headline is not None:
        headline_check = {
            "expected_balanced_accuracy_mean":
                safe_float(headline.get("balanced_accuracy_mean")),
            "observed_balanced_accuracy_mean":
                observed_ba_mean,
            "expected_rescue_mean":
                safe_float(headline.get("rescue_mean")),
            "observed_rescue_mean":
                observed_rescue_mean,
            "expected_harm_mean":
                safe_float(headline.get("harm_mean")),
            "observed_harm_mean":
                observed_harm_mean,
            "expected_net_rescue_mean":
                safe_float(headline.get("net_rescue_mean")),
            "observed_net_rescue_mean":
                observed_net_mean,
        }

    audit = {
        "phase": "P53_TOP5_CORRECTION_STRATIFICATION",
        "target_variant": args.variant,
        "top_k": args.top_k,
        "selected_roles": selected_roles,
        "selected_role_parser_source": parser_source,
        "prediction_row_count": len(predictions),
        "unique_target_sha_count": len(target_shas),
        "joined_feature_sha_count": len(feature_presence),
        "split_count": len(split_groups),
        "memory_policy": (
            "No APKs or graphs opened. Predictions are small and retained "
            "in memory; Compact features are streamed row-by-row."
        ),
        "role_presence_definition": (
            "role_node__<semantic_role> > 0 in raw Compact features"
        ),
        "interpretation_limit": (
            "Top-5 role presence does not itself prove that detailed Entry "
            "context was available/nonzero for that role. This analysis "
            "tests semantic hotspot presence, not Entry-support availability."
        ),
        "observed": {
            "balanced_accuracy_mean": observed_ba_mean,
            "rescue_mean": observed_rescue_mean,
            "harm_mean": observed_harm_mean,
            "net_rescue_mean": observed_net_mean,
        },
        "headline_check": headline_check,
        "inputs": {
            "features": str(args.features),
            "predictions": str(args.predictions),
            "selected_roles": str(args.selected_roles),
            "headline": str(args.headline),
        },
        "outputs": {
            "apk_level": str(apk_path),
            "presence_summary": str(presence_path),
            "count_summary": str(count_path),
            "role_summary": str(role_path),
        },
    }

    audit_path = args.output / "p53_top5_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=== QC ===")
    print(
        f"observed P53 mean BA: {observed_ba_mean:.6f}"
    )
    print(
        f"observed rescue/harm/net: "
        f"{observed_rescue_mean:.3f} / "
        f"{observed_harm_mean:.3f} / "
        f"{observed_net_mean:.3f}"
    )

    if headline_check is not None:
        print(
            f"headline rescue/harm/net: "
            f"{headline_check['expected_rescue_mean']:.3f} / "
            f"{headline_check['expected_harm_mean']:.3f} / "
            f"{headline_check['expected_net_rescue_mean']:.3f}"
        )

    print()
    print("=== IMPORTANT: pooled Top-5 presence ===")

    for presence in ("ABSENT", "PRESENT"):
        key = ("ALL_FUTURE", "ALL", "ALL", presence)
        stats = presence_groups.get(key)
        if stats is None:
            continue

        row = stats.row()

        print(
            f"{presence:7s} "
            f"N={row['apk_count']:4d} "
            f"CompactBA={row['compact_balanced_accuracy']:.4f} "
            f"P53BA={row['p53_balanced_accuracy']:.4f} "
            f"delta={row['p53_minus_compact_ba_pp']:+.2f}pp "
            f"corrected={row['corrected']} "
            f"regressed={row['regressed']} "
            f"net={row['net_correction']:+d}"
        )

    print()
    print("wrote:")
    print(f"  {apk_path}")
    print(f"  {presence_path}")
    print(f"  {count_path}")
    print(f"  {role_path}")
    print(f"  {audit_path}")


if __name__ == "__main__":
    main()
