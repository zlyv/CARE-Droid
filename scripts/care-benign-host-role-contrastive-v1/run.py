#!/usr/bin/env python3
"""Train a Host-level, benign-only Role contrastive auxiliary loss.

Reads frozen CARE feature caches and original Direct-C43 checkpoints. Never
reads APKs or changes formal code. Writes exclusively under a new experiment
output directory. This is development research, not a deployed classifier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
OLD = PROJECT / "scripts/care-existing-three-background-contrastive-v2"
sys.path.insert(0, str(OLD))
import contrastive as legacy  # noqa: E402
from common import BASE, original  # noqa: E402

OUTPUT = PROJECT / "outputs/care_benign_host_role_contrastive_v1"
TAGS = ("C0R_2017", "T2_C1", "T3_C1", "T4_C1")
SEEDS = (900000, 900001, 900002)
ARMS = ("H1_TRUE_ROLE", "H2_ROLE_SHUFFLE")
ROLE_CONFIDENCE = 0.8
HOSTS_PER_APK = 4
TEMPERATURE = 0.1
LOSS_WEIGHT = 0.05


def roles_from_raw(data, train_index, bags):
    from train_stage2 import raw_bags

    raw = raw_bags(data, train_index)
    result = []
    for original_bag, transformed in zip(raw, bags):
        if len(original_bag) != len(transformed):
            raise RuntimeError("Host order/count mismatch")
        w = np.maximum(np.asarray(original_bag[:, 43:74], np.float32), 0)
        sums = w.sum(1, keepdims=True)
        w = np.divide(w, sums, out=np.zeros_like(w), where=sums > 0)
        good = np.flatnonzero(w.max(1) >= ROLE_CONFIDENCE)
        result.append((good, w.argmax(1).astype(np.int16)))
    return result


def host_supcon(torch, nn, vectors, roles, owners):
    n = len(vectors)
    if n < 3:
        return vectors.sum() * 0, 0, 0
    same_apk = owners[:, None] == owners[None, :]
    valid_pair = ~same_apk
    same_role = roles[:, None] == roles[None, :]
    pos = valid_pair & same_role
    neg = valid_pair & ~same_role
    valid_anchor = pos.any(1) & neg.any(1)
    if not bool(valid_anchor.any()):
        return vectors.sum() * 0, 0, 0
    z = nn.functional.normalize(vectors, dim=1, eps=1e-8)
    sim = z @ z.T / TEMPERATURE
    sim = sim.masked_fill(~valid_pair, -1e9)
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(logprob * pos.float()).sum(1) / pos.sum(1).clamp_min(1)
    return loss[valid_anchor].mean(), int(valid_anchor.sum()), int(pos.sum())


def run(tag: str, seed: int, arm: str, device_name: str, pilot: bool) -> None:
    if tag not in TAGS or seed not in SEEDS or arm not in ARMS:
        raise RuntimeError("Unregistered tag/seed/arm")
    mc, v4, _, data = original()
    torch = v4.torch
    nn = v4.nn
    device = torch.device(device_name)
    if device.type == "cuda" and torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free GPU")
    obj, args, tr, va, glob, bags, graphs, _ = legacy.inputs(mc, v4, data, tag, seed)
    if pilot:
        labels = np.asarray(data["y"][tr], int)
        chosen = np.r_[
            np.flatnonzero(labels == 0)[:64],
            np.flatnonzero(labels == 1)[:64],
        ]
        tr = tr[chosen]
        glob = (glob[0][chosen], glob[1])
        bags = ([bags[0][int(i)] for i in chosen], bags[1])
        graphs = ({k: v[chosen] for k, v in graphs[0].items()}, graphs[1])
    labels = np.asarray(data["y"][tr], int)
    role_meta = roles_from_raw(data, tr, bags[0])
    role_apk_counts = Counter()
    eligible_hosts = 0
    for y, (eligible, dominant) in zip(labels, role_meta):
        if y != 0:
            continue
        eligible_hosts += len(eligible)
        role_apk_counts.update(set(map(int, dominant[eligible])))
    effective_roles = sum(count >= 2 for count in role_apk_counts.values())
    print(
        "BHRC_PREFLIGHT", tag, seed, arm,
        f"train={len(tr)} benign_eligible_hosts={eligible_hosts}",
        f"roles_in_two_or_more_APKs={effective_roles}", flush=True,
    )
    if eligible_hosts < 50 or effective_roles < 3:
        raise RuntimeError("Not enough cross-APK benign Role-positive candidates")

    v4.seed_everything(seed)
    model = v4.RoleGraphConditionedEvidence(24, "RGCondAttn", args).to(device)
    # Keep the original model initialization RNG identical to the frozen BCE.
    with torch.random.fork_rng(
        devices=[torch.cuda.current_device()] if device.type == "cuda" else []
    ):
        torch.manual_seed(seed + 1907)
        projection = nn.Sequential(
            nn.Linear(int(args.embedding_dim), 24),
            nn.ReLU(),
            nn.Linear(24, 16),
        ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projection.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    gt = torch.as_tensor(glob[0], dtype=torch.float32, device=device)
    graph_t = v4.graph_tensors(graphs[0], device)
    bag_t = [torch.as_tensor(b, dtype=torch.float32, device=device) for b in bags[0]]
    y_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
    g_val = v4.graph_tensors(graphs[1], device)
    b_val = [torch.as_tensor(b, dtype=torch.float32, device=device) for b in bags[1]]
    x_val = torch.as_tensor(glob[1], dtype=torch.float32, device=device)
    rng = np.random.RandomState(seed + 3301 + 811)
    epochs = 1 if pilot else int(obj["audit"]["selected_epochs"])
    counts = Counter()
    projection_gradient_seen = False
    t0 = time.monotonic()
    for epoch in range(epochs):
        model.train()
        projection.train()
        order = rng.permutation(len(tr))
        epoch_loss = 0.0
        for start in range(0, len(order), int(args.batch_size)):
            batch = order[start:start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = []
            local_vectors = []
            local_roles = []
            local_owners = []
            for position in batch:
                i = int(position)
                one_graph = {k: val[i] for k, val in graph_t.items()}
                logit, _, _ = model.forward_with_trace(
                    gt[i], torch.zeros((), device=device), one_graph, bag_t[i]
                )
                logits.append(logit)
                if labels[i] != 0:
                    continue
                good, dominant = role_meta[i]
                if not len(good):
                    continue
                if len(good) > HOSTS_PER_APK:
                    selected = np.sort(
                        np.random.RandomState(
                            seed + epoch * 1000003 + start * 1009 + i
                        ).choice(good, HOSTS_PER_APK, replace=False)
                    )
                else:
                    selected = good
                selected_t = torch.as_tensor(
                    selected, dtype=torch.long, device=device
                )
                # Only 107 Compact + four frozen background responses enter
                # this encoder; explicit Role31 weights are NOT its input.
                z = model.region_encoder(bag_t[i][selected_t, :111])
                local_vectors.append(projection(z))
                local_roles.extend(map(int, dominant[selected]))
                local_owners.extend([i] * len(selected))
            logits_t = torch.stack(logits)
            loss_bce = nn.functional.binary_cross_entropy_with_logits(
                logits_t, y_t[batch]
            )
            if local_vectors:
                vector_t = torch.cat(local_vectors, dim=0)
                role_values = np.asarray(local_roles, dtype=np.int64)
                if arm == "H2_ROLE_SHUFFLE":
                    role_values = np.random.RandomState(
                        seed + epoch * 1000003 + start + 88123
                    ).permutation(role_values)
                role_t = torch.as_tensor(
                    role_values, dtype=torch.long, device=device
                )
                owner_t = torch.as_tensor(
                    local_owners, dtype=torch.long, device=device
                )
                aux, anchors, positive_pairs = host_supcon(
                    torch, nn, vector_t, role_t, owner_t
                )
            else:
                aux = loss_bce * 0
                anchors = positive_pairs = 0
            loss = loss_bce + LOSS_WEIGHT * aux
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Host contrastive training loss")
            loss.backward()
            if anchors and any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
                for parameter in projection.parameters()
            ):
                projection_gradient_seen = True
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(projection.parameters()),
                args.grad_clip,
            )
            optimizer.step()
            counts["anchors"] += anchors
            counts["positive_pairs"] += positive_pairs
            counts["steps"] += 1
            epoch_loss += float(loss.detach().cpu()) * len(batch)
        print(
            "BHRC_EPOCH_DONE", tag, seed, arm, epoch + 1, epochs,
            f"loss={epoch_loss/len(tr):.6f}",
            f"anchors={counts['anchors']}",
            flush=True,
        )
    if (
        counts["anchors"] == 0
        or counts["positive_pairs"] == 0
        or not projection_gradient_seen
    ):
        raise RuntimeError("Host Role contrastive objective was never evaluable")
    model.eval()
    probability = []
    with torch.no_grad():
        for i in range(len(va)):
            one_graph = {k: v[i] for k, v in g_val.items()}
            logit, _, _ = model.forward_with_trace(
                x_val[i], torch.zeros((), device=device), one_graph, b_val[i]
            )
            probability.append(float(torch.sigmoid(logit).cpu()))
    probability = np.asarray(probability)
    if not np.isfinite(probability).all():
        raise RuntimeError("Nonfinite validation predictions")
    elapsed = time.monotonic() - t0
    destination = OUTPUT / ("pilot" if pilot else "full") / tag / str(seed)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination / arm
    with (prefix.with_name(prefix.name + "_predictions.npz")).open("xb") as f:
        np.savez_compressed(
            f,
            sha256=np.asarray(obj["metadata"]["validation_sha"]),
            probability=probability,
        )
    if not pilot:
        with (prefix.with_name(prefix.name + "_checkpoint.pt")).open("xb") as f:
            torch.save({
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "projection_state": {
                    k: v.detach().cpu() for k, v in projection.state_dict().items()
                },
                "tag": tag,
                "seed": seed,
                "arm": arm,
                "base_checkpoint": str(BASE / "models/stage2" / tag / f"{seed}.pkl"),
            }, f)
    audit = {
        "status": "PILOT_PASS" if pilot else "TRAINING_DONE",
        "tag": tag,
        "seed": seed,
        "arm": arm,
        "train_APKs": len(tr),
        "validation_APKs": len(va),
        "epochs": epochs,
        "benign_eligible_hosts": eligible_hosts,
        "role_apk_counts": dict(role_apk_counts),
        "anchors": counts["anchors"],
        "positive_pairs": counts["positive_pairs"],
        "projection_gradient_seen": projection_gradient_seen,
        "steps": counts["steps"],
        "seconds": elapsed,
        "loss_weight": LOSS_WEIGHT,
        "role_confidence": ROLE_CONFIDENCE,
        "hosts_per_APK": HOSTS_PER_APK,
        "temperature": TEMPERATURE,
        "model_parameters": sum(v.numel() for v in model.parameters()),
        "projection_parameters": sum(v.numel() for v in projection.parameters()),
        "APK_files_opened": 0,
        "future_labels_used_for_training": False,
    }
    with (prefix.with_name(prefix.name + "_audit.json")).open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(audit, f, indent=2, sort_keys=True)
        f.write("\n")
    print("BHRC_ARM_DONE", tag, seed, arm, audit["status"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", choices=TAGS, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--pilot", action="store_true")
    cli = p.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    run(cli.tag, cli.seed, cli.arm, cli.device, cli.pilot)
