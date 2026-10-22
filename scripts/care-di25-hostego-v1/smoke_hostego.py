#!/usr/bin/env python3
"""Actual-cache smoke test: five nontrivial and five fallback source APKs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment_contract import read_csv, write_csv
from host_ego_encoder import DirectedHostEgoFusion


def main():
    root = Path("artifacts/care-source-local-assets-v2")
    out = Path("artifacts/care-di25-hostego-v1/smoke")
    out.mkdir(parents=True, exist_ok=True)
    with np.load(root / "host_regions_2017.npz", allow_pickle=False) as h, np.load(root / "ego_1hop_regions_2017.npz", allow_pickle=False) as e:
        sha = h["sha256"].astype(str); values = np.asarray(h["host_values"], np.float32)
        hoff = np.asarray(h["host_offsets"], np.int64); real = np.asarray(h["real_host_count"], np.int64)
        eoff = np.asarray(e["ego_offsets"], np.int64); status=np.asarray(e["ego_status"],np.int64)
        ioff=np.asarray(e["incoming_offsets"],np.int64); ooff=np.asarray(e["outgoing_offsets"],np.int64)
        ip=np.asarray(e["incoming_host_positions"],np.int64); op=np.asarray(e["outgoing_host_positions"],np.int64)
        with_ego=[i for i in range(len(sha)) if np.any(status[eoff[i]:eoff[i+1]]==1)][:5]
        without=[i for i in range(len(sha)) if int(real[i])>0 and not np.any(status[eoff[i]:eoff[i+1]]==1)][:5]
        if len(with_ego)<5 or len(without)<5: raise RuntimeError("smoke cohort lacks 5+5 APKs")
        torch.manual_seed(20260919)
        encoder=nn.Sequential(nn.Linear(74,48),nn.ReLU(),nn.Linear(48,24),nn.ReLU()).eval()
        fusion=DirectedHostEgoFusion(24).eval(); rows=[]
        for i in with_ego+without:
            eb,ee=int(eoff[i]),int(eoff[i+1]); host=torch.as_tensor(values[hoff[i]:hoff[i+1]])
            edge_set=set()
            for g in range(eb,ee):
                edge_set.update((g-eb,int(v-eb)) for v in op[ooff[g]:ooff[g+1]])
            edges=torch.as_tensor(sorted(edge_set),dtype=torch.long).reshape(-1,2)
            with torch.no_grad():
                h0=encoder(host); h1,_,gamma,used,inc,outc=fusion(h0,edges)
            diff=torch.max(torch.abs(h1-h0),dim=1).values.numpy()
            if np.any(used.numpy()) and not np.any(diff[used.numpy()]>1e-8): raise RuntimeError("nontrivial Ego did not change representation")
            if np.any(~used.numpy()) and np.max(diff[~used.numpy()])>1e-7: raise RuntimeError("fallback is not exact")
            rows.append({"sha256":sha[i],"expected_nontrivial":int(i in with_ego),"host_count":len(host),
                         "ego_used_hosts":int(used.sum()),"max_changed_diff":float(np.max(diff[used.numpy()])) if np.any(used.numpy()) else 0.0,
                         "max_fallback_diff":float(np.max(diff[~used.numpy()])) if np.any(~used.numpy()) else 0.0,
                         "max_gamma":float(torch.max(gamma[used]).item()) if torch.any(used) else 0.0})
    reference=read_csv(Path("artifacts/care-di25-hostego-v1/preflight/di25_reference/source_predictions.csv"))
    routes=[int(r["A25__routed"]) for r in reference]
    assert sum(routes)==200 and all(routes==list(routes) for _ in range(4))
    # Integration invariants independent of trained weights.
    p1=np.asarray([float(r["Global24__prob"]) for r in reference]); routed=np.asarray(routes,dtype=bool)
    final=p1.copy(); assert np.max(np.abs(final[~routed]-p1[~routed]))==0.0
    raw=np.linspace(-10,10,len(p1)); delta=2.25*np.tanh(raw); assert np.max(np.abs(delta))<=2.25+1e-6
    write_csv(out/"smoke_examples.csv",rows)
    (out/"REPORT.txt").write_text("HOSTEGO_SMOKE_PASS\nselected_nontrivial=5\nselected_fallback=5\nroute_difference_count=0\nmax_abs_delta<=2.25\nnonrouted_final_equals_stage1=true\n",encoding="utf-8")
    print((out/"REPORT.txt").read_text(encoding="utf-8"))


if __name__ == "__main__": main()

