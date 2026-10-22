#!/usr/bin/env python3

from pathlib import Path
import argparse
import collections
import csv
import json


SINK_ROLES = (
    "dynamic_code_loading",
    "network_transmission_or_upload",
    "command_execution",
    "sms_send_receive",
)


def safe_role(value):
    return str(value).strip().replace("-", "_")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--schema-out",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    unit_files = sorted(
        args.input.glob(
            "batch_*/result_centered_behavior_units.jsonl"
        )
    )

    print("unit files:", len(unit_files))

    apk_rows = {}
    verified_pairs = set()
    same_roles = set()

    # --------------------------------------------------------
    # First pass:
    # collect APK statistics and SOURCE-ONLY verified-pair schema.
    # --------------------------------------------------------

    for path in unit_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                wrapper = json.loads(line)

                sha = str(
                    wrapper.get("sha256", "")
                ).strip().lower()

                if not sha:
                    continue

                wrapper_status = str(
                    wrapper.get("status", "UNKNOWN")
                )

                row = apk_rows.setdefault(
                    sha,
                    {
                        "sha256": sha,
                        "analysis_status": wrapper_status,
                        "dep_analysis_ok": int(
                            wrapper_status == "OK"
                        ),
                    },
                )

                # Make sure a later duplicate wrapper cannot
                # accidentally turn a failure into success.
                if wrapper_status != "OK":
                    row["analysis_status"] = wrapper_status
                    row["dep_analysis_ok"] = 0

                units = wrapper.get("units", []) or []

                for unit in units:
                    status = str(
                        unit.get(
                            "dependency_status",
                            "",
                        )
                    )

                    sink = safe_role(
                        unit.get(
                            "sink_role",
                            "",
                        )
                    )

                    kind = str(
                        unit.get(
                            "sink_argument_kind",
                            "",
                        )
                    )

                    row["dep_total_unit_count"] = (
                        row.get(
                            "dep_total_unit_count",
                            0,
                        )
                        + 1
                    )

                    if status == "NON_PAYLOAD_ENDPOINT":
                        row[
                            "dep_non_payload_endpoint_count"
                        ] = (
                            row.get(
                                "dep_non_payload_endpoint_count",
                                0,
                            )
                            + 1
                        )

                    elif status == "UNRESOLVED":
                        row["dep_unresolved_count"] = (
                            row.get(
                                "dep_unresolved_count",
                                0,
                            )
                            + 1
                        )

                    elif (
                        status
                        == "RESOLVED_NO_SENSITIVE_SOURCE"
                    ):
                        row[
                            "dep_resolved_no_sensitive_count"
                        ] = (
                            row.get(
                                "dep_resolved_no_sensitive_count",
                                0,
                            )
                            + 1
                        )

                    elif (
                        status
                        == "VERIFIED_CROSS_ROLE_DEPENDENCY"
                    ):
                        row[
                            "dep_verified_cross_role_count"
                        ] = (
                            row.get(
                                "dep_verified_cross_role_count",
                                0,
                            )
                            + 1
                        )

                        upstream_roles = sorted(
                            {
                                safe_role(x)
                                for x in (
                                    unit.get(
                                        "upstream_roles",
                                        [],
                                    )
                                    or []
                                )
                                if safe_role(x)
                                and safe_role(x) != sink
                            }
                        )

                        for upstream in upstream_roles:
                            pair = (
                                upstream,
                                sink,
                            )

                            verified_pairs.add(pair)

                            key = (
                                "dep_verified_pair__"
                                f"{upstream}"
                                "__to__"
                                f"{sink}"
                            )

                            row[key] = (
                                row.get(key, 0)
                                + 1
                            )

                    elif (
                        status
                        == "VERIFIED_SAME_ROLE_DEPENDENCY"
                    ):
                        row[
                            "dep_verified_same_role_count"
                        ] = (
                            row.get(
                                "dep_verified_same_role_count",
                                0,
                            )
                            + 1
                        )

                        if sink:
                            same_roles.add(sink)

                            key = (
                                "dep_verified_same__"
                                f"{sink}"
                            )

                            row[key] = (
                                row.get(key, 0)
                                + 1
                            )

                    # Carrier is an analysis state, not verified evidence.
                    if kind in {
                        "argument_carrier",
                        "receiver_carrier",
                    }:
                        row[
                            "dep_carrier_unit_count"
                        ] = (
                            row.get(
                                "dep_carrier_unit_count",
                                0,
                            )
                            + 1
                        )

                        if status == "UNRESOLVED":
                            row[
                                "dep_carrier_unresolved_count"
                            ] = (
                                row.get(
                                    "dep_carrier_unresolved_count",
                                    0,
                                )
                                + 1
                            )

                    if kind == "direct_payload":
                        row[
                            "dep_direct_payload_count"
                        ] = (
                            row.get(
                                "dep_direct_payload_count",
                                0,
                            )
                            + 1
                        )

                    if sink in SINK_ROLES:
                        key = (
                            "dep_sink__"
                            f"{sink}"
                            "_count"
                        )

                        row[key] = (
                            row.get(key, 0)
                            + 1
                        )

    # --------------------------------------------------------
    # Freeze SOURCE-ONLY feature schema.
    # Future years must use this schema unchanged.
    # --------------------------------------------------------

    base_features = [
        "dep_analysis_ok",
        "dep_total_unit_count",
        "dep_non_payload_endpoint_count",
        "dep_payload_eligible_count",
        "dep_direct_payload_count",
        "dep_carrier_unit_count",
        "dep_carrier_unresolved_count",
        "dep_unresolved_count",
        "dep_resolved_no_sensitive_count",
        "dep_verified_cross_role_count",
        "dep_verified_same_role_count",
        "dep_verified_pair_diversity",
        "dep_resolution_rate",
        "dep_unresolved_rate",
    ]

    sink_features = [
        f"dep_sink__{role}_count"
        for role in SINK_ROLES
    ]

    pair_features = [
        (
            "dep_verified_pair__"
            f"{source}"
            "__to__"
            f"{sink}"
        )
        for source, sink in sorted(
            verified_pairs
        )
    ]

    same_features = [
        f"dep_verified_same__{role}"
        for role in sorted(
            same_roles
        )
    ]

    feature_columns = (
        base_features
        + sink_features
        + pair_features
        + same_features
    )

    # --------------------------------------------------------
    # Finalize ratios.
    #
    # NON_PAYLOAD_ENDPOINT is deliberately excluded from
    # payload-eligible coverage denominator.
    # --------------------------------------------------------

    for row in apk_rows.values():
        for key in feature_columns:
            if key == "dep_analysis_ok":
                continue

            row.setdefault(key, 0)

        total = int(
            row.get(
                "dep_total_unit_count",
                0,
            )
        )

        non_payload = int(
            row.get(
                "dep_non_payload_endpoint_count",
                0,
            )
        )

        payload_eligible = max(
            0,
            total - non_payload,
        )

        row[
            "dep_payload_eligible_count"
        ] = payload_eligible

        verified_cross = int(
            row.get(
                "dep_verified_cross_role_count",
                0,
            )
        )

        verified_same = int(
            row.get(
                "dep_verified_same_role_count",
                0,
            )
        )

        resolved_negative = int(
            row.get(
                "dep_resolved_no_sensitive_count",
                0,
            )
        )

        unresolved = int(
            row.get(
                "dep_unresolved_count",
                0,
            )
        )

        resolved_total = (
            verified_cross
            + verified_same
            + resolved_negative
        )

        if payload_eligible > 0:
            row[
                "dep_resolution_rate"
            ] = (
                resolved_total
                / payload_eligible
            )

            row[
                "dep_unresolved_rate"
            ] = (
                unresolved
                / payload_eligible
            )
        else:
            row[
                "dep_resolution_rate"
            ] = 0.0

            row[
                "dep_unresolved_rate"
            ] = 0.0

        pair_keys = [
            key
            for key in pair_features
            if row.get(key, 0) > 0
        ]

        row[
            "dep_verified_pair_diversity"
        ] = len(pair_keys)

    # --------------------------------------------------------
    # Persist.
    # --------------------------------------------------------

    rows = sorted(
        apk_rows.values(),
        key=lambda x: x["sha256"],
    )

    fieldnames = [
        "sha256",
        "analysis_status",
    ] + feature_columns

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key, 0)
                    for key in fieldnames
                }
            )

    schema = {
        "source_year": 2017,
        "feature_columns": feature_columns,
        "verified_pair_features": pair_features,
        "verified_same_role_features": same_features,
        "sink_volume_features": sink_features,
        "notes": {
            "future_schema_frozen": True,
            "non_payload_excluded_from_eligible_denominator": True,
            "pair_features_only_from_verified_cross_role_units": True,
        },
    }

    args.schema_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.schema_out.write_text(
        json.dumps(
            schema,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    ok_count = sum(
        int(
            row.get(
                "dep_analysis_ok",
                0,
            )
        )
        for row in rows
    )

    print("APKs:", len(rows))
    print("OK APKs:", ok_count)
    print(
        "Feature columns:",
        len(feature_columns),
    )
    print(
        "Verified pair features:",
        len(pair_features),
    )
    print(
        "Same-role features:",
        len(same_features),
    )

    print()
    print("Verified pairs:")

    if not verified_pairs:
        print("  (none)")
    else:
        for source, sink in sorted(
            verified_pairs
        ):
            print(
                f"  {source} -> {sink}"
            )


if __name__ == "__main__":
    main()
