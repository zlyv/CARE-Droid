#!/usr/bin/env python3

from pathlib import Path
import argparse
import collections
import csv
import json


def update_feature(features, unit):

    status = unit.get(
        "dependency_status",
        ""
    )

    sink = unit.get(
        "sink_role",
        ""
    )

    roles = unit.get(
        "roles",
        []
    )

    upstream = unit.get(
        "upstream_roles",
        []
    )

    key_roles = set(
        roles
        +
        upstream
    )


    # Global evidence statistics
    features[
        "dep_total_sink_count"
    ] += 1


    kind = unit.get(
        "sink_argument_kind",
        ""
    )


    if kind == "direct_payload":
        features[
            "dep_direct_payload_count"
        ] += 1


    if status == "NON_PAYLOAD_ENDPOINT":
        features[
            "dep_non_payload_endpoint_count"
        ] += 1


    if "CARRIER" in status:
        features[
            "dep_carrier_candidate_count"
        ] += 1


    if status == "VERIFIED_CROSS_ROLE_DEPENDENCY":
        features[
            "dep_verified_cross_role_count"
        ] += 1


    elif status == "VERIFIED_SAME_ROLE_DEPENDENCY":
        features[
            "dep_verified_same_role_count"
        ] += 1


    elif status == "RESOLVED_NO_SENSITIVE_SOURCE":
        features[
            "dep_resolved_no_sensitive_count"
        ] += 1


    elif status == "UNRESOLVED":
        features[
            "dep_unresolved_count"
        ] += 1


    if sink == "dynamic_code_loading":
        features[
            "dep_dynamic_loading_count"
        ] += 1


    if sink == "network_transmission_or_upload":
        features[
            "dep_network_upload_count"
        ] += 1


    if sink == "command_execution":
        features[
            "dep_command_execution_count"
        ] += 1


    if (
        "database_access" in key_roles
        and
        "dynamic_code_loading" in key_roles
    ):
        features[
            "dep_database_to_dynamic_loading"
        ] += 1


    if (
        "reflective_invocation" in key_roles
        and
        "dynamic_code_loading" in key_roles
    ):
        features[
            "dep_reflection_to_dynamic_loading"
        ] += 1


    if (
        "command_execution" in key_roles
        and
        "dynamic_code_loading" in key_roles
    ):
        features[
            "dep_command_to_dynamic_loading"
        ] += 1


    if (
        "network_transmission_or_upload"
        in key_roles
        and
        "dynamic_code_loading"
        in key_roles
    ):
        features[
            "dep_network_to_dynamic_loading"
        ] += 1


    if (
        "database_access" in key_roles
        and
        "network_transmission_or_upload"
        in key_roles
    ):
        features[
            "dep_database_to_network"
        ] += 1


    if (
        "file_read" in key_roles
        and
        "network_transmission_or_upload"
        in key_roles
    ):
        features[
            "dep_file_to_network"
        ] += 1


    if (
        "device_identifier_access"
        in key_roles
        and
        "network_transmission_or_upload"
        in key_roles
    ):
        features[
            "dep_device_to_network"
        ] += 1


    if (
        "location_access"
        in key_roles
        and
        "network_transmission_or_upload"
        in key_roles
    ):
        features[
            "dep_location_to_network"
        ] += 1



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

    args = parser.parse_args()


    apk_features = {}


    files = sorted(
        args.input.glob(
            "batch_*/result_centered_behavior_units.jsonl"
        )
    )


    print(
        "unit files:",
        len(files)
    )


    for path in files:

        with path.open(
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                wrapper = json.loads(line)

                sha = wrapper["sha256"]

                if sha not in apk_features:

                    apk_features[sha] = collections.Counter()

                for unit in wrapper.get(
                    "units",
                    []
                ):

                    update_feature(
                        apk_features[sha],
                        unit
                    )


    rows = []

    for sha, feat in apk_features.items():

        row = {
            "sha256": sha
        }

        row.update(feat)

        row[
            "dep_uncertainty_ratio"
        ] = (
            feat[
                "dep_unresolved_count"
            ]
            /
            max(
                1,
                sum(feat.values())
            )
        )

        rows.append(row)


    # fill missing
    keys = set()

    for row in rows:
        keys.update(row.keys())


    for row in rows:

        for k in keys:

            if k not in row:
                row[k] = 0


    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=sorted(keys)
        )

        writer.writeheader()
        writer.writerows(rows)


    print(
        "APKs:",
        len(rows)
    )

    print(
        "Features:",
        len(keys)-1
    )


if __name__ == "__main__":
    main()
