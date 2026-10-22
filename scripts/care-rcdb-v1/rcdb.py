#!/usr/bin/env python3
"""RCDB bank/response implementation; reads cached Host tensors, writes nowhere."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROLE_N = 31
PER_CLASS = 12
CAP_PER_ROLE_APK = 8
MIN_APK_SUPPORT = 5


def split_raw(raw, mean, std):
    import haig_background as hb
    core, weights, rel = hb.split(raw)
    z = [np.asarray((x - mean) / std, np.float32) for x in core]
    role = []
    for w in weights:
        a = np.argmax(w, axis=1).astype(np.int16)
        a[np.sum(w, axis=1) == 0] = -1
        role.append(a)
    return z, role


def _sample(z, role, seed):
    rng = np.random.RandomState(int(seed) + 2481)
    samples = []
    for apk, (bag, rid) in enumerate(zip(z, role)):
        for r in np.unique(rid):
            if r < 0:
                continue
            idx = np.flatnonzero(rid == r)
            if len(idx) > CAP_PER_ROLE_APK:
                idx = np.sort(rng.choice(idx, CAP_PER_ROLE_APK, replace=False))
            for j in idx:
                samples.append((apk, int(r), bag[j]))
    return samples


def _rms(x, c):
    import haig_background as hb
    return hb.v2.pairwise_rms(x, c) if hasattr(hb, 'v2') else np.sqrt(
        np.maximum(np.mean((x[:, None, :] - c[None, :, :]) ** 2, axis=2), 0))


def fit_banks(raw_train, y, prep, seed):
    """One candidate algorithm and exact 12+12 quotas for both treatment arms."""
    z, roles = split_raw(raw_train, prep['mean'], prep['std'])
    y = np.asarray(y, int)
    samples = _sample(z, roles, seed)
    apk = np.asarray([t[0] for t in samples], int)
    rid = np.asarray([t[1] for t in samples], int)
    x = np.asarray([t[2] for t in samples], np.float32)
    if x.shape[1] != 107:
        raise RuntimeError('RCDB candidate space is not Compact107')
    counts = {}
    for r in range(ROLE_N):
        counts[r] = tuple(len(np.unique(apk[(rid == r) & (y[apk] == cls)])) for cls in (0, 1))
    eligible = [r for r in range(ROLE_N) if min(counts[r]) >= MIN_APK_SUPPORT]
    eligible.sort(key=lambda r: (-min(counts[r]), -(counts[r][0] + counts[r][1]), r))
    chosen = eligible[:PER_CLASS]
    if len(chosen) != PER_CLASS:
        raise RuntimeError('fewer than 12 reliable two-sided Role strata; cannot fill 24 slots')
    shared = np.asarray(prep['state']['shared_centers'], np.float32)
    sp = np.asarray(prep['sp'], np.float32)
    result = {arm: {0: [], 1: []} for arm in ('M2_RCDB', 'M3_BALANCED_PROTOTYPE')}
    manifest = []
    support = []
    for r in chosen:
        rx = x[rid == r]
        ra = apk[rid == r]
        hp = np.argmax(sp, axis=1)
        compatible = np.flatnonzero((np.sum(sp, axis=1) > 0) & (hp == r))
        if len(compatible) == 0:
            compatible = np.arange(len(shared))
        shared_dist = np.min(_rms(rx, shared[compatible]), axis=1)
        shared_tau = float(np.median(shared_dist))
        shared_supported = set(int(v) for v in np.unique(ra[shared_dist <= shared_tau]))
        for cls in (0, 1):
            own = (rid == r) & (y[apk] == cls)
            own_x = x[own]
            own_apk = apk[own]
            k = min(4, len(own_x))
            km = MiniBatchKMeans(n_clusters=k, random_state=int(seed) + 300 + 11*r + cls,
                                 batch_size=min(2048, max(256, len(own_x))), n_init=10,
                                 max_iter=200, reassignment_ratio=0.01).fit(own_x)
            centers = np.asarray(km.cluster_centers_, np.float32)
            nearest = np.min(_rms(own_x, centers), axis=1)
            tau = max(float(np.median(nearest)), 1e-6)
            n_own = max(int(np.sum(y == cls)), 1)
            n_other = max(int(np.sum(y != cls)), 1)
            candidate_scores = []
            for ci, c in enumerate(centers):
                d = _rms(rx, c[None, :])[:, 0]
                covered = set(int(v) for v in np.unique(ra[d <= tau]))
                good = {a for a in covered if y[a] == cls}
                bad = {a for a in covered if y[a] != cls}
                own_rate = len(good) / n_own
                other_rate = len(bad) / n_other
                unique_rate = len(good - shared_supported) / n_own
                redundancy = len(good & shared_supported) / n_own
                # Prespecified source-only objective; no future-year tuning.
                score = unique_rate + .25*own_rate - other_rate - .25*redundancy
                candidate_scores.append(dict(candidate=ci, score=score, own_rate=own_rate,
                                             other_rate=other_rate, unique_rate=unique_rate,
                                             redundancy=redundancy, own_n=len(good),
                                             other_n=len(bad), tau=tau))
            m2 = max(candidate_scores, key=lambda q: (q['score'], q['own_rate'], -q['candidate']))
            m3 = max(candidate_scores, key=lambda q: (q['own_rate'], -q['candidate']))
            for arm, sel in (('M2_RCDB', m2), ('M3_BALANCED_PROTOTYPE', m3)):
                result[arm][cls].append((r, centers[sel['candidate']]))
                manifest.append(dict(arm=arm, role=r, class_reference=cls, **sel,
                                     candidate_algorithm='MiniBatchKMeans_k4_per_class_role',
                                     candidate_count=k, role_apk_benign=counts[r][0],
                                     role_apk_malware=counts[r][1]))
            support.append(dict(role=r, benign_apks=counts[r][0], malware_apks=counts[r][1],
                                shared_supported_apks=len(shared_supported), candidates_per_class=k))
    if any(len(result[arm][cls]) != PER_CLASS for arm in result for cls in (0, 1)):
        raise RuntimeError('24-prototype capacity broken')
    return result, manifest, support, (z, roles)


def contrast(z, roles, bank):
    benign = dict(bank[0]); malware = dict(bank[1])
    out = []
    observed = []
    for bag, rid in zip(z, roles):
        delta = np.zeros(len(bag), np.float32)
        valid = np.zeros(len(bag), bool)
        for r in set(int(v) for v in rid if v >= 0):
            idx = np.flatnonzero(rid == r)
            if r not in benign or r not in malware:
                continue
            b = _rms(bag[idx], benign[r][None, :])[:, 0]
            m = _rms(bag[idx], malware[r][None, :])[:, 0]
            delta[idx] = b - m
            valid[idx] = True
        out.append(delta)
        observed.append(valid)
    return out, observed


def calibrate(original_train, train_z, train_roles, m3_bank):
    d, mask = contrast(train_z, train_roles, m3_bank)
    c = np.concatenate([v[m] for v, m in zip(d, mask)])
    if len(c) < 100:
        raise RuntimeError('insufficient train-only contrast calibration')
    scale = max(float(np.quantile(np.abs(c), .75)), 1e-6)
    ext = np.concatenate([b[:, 107] for b in original_train])
    center = float(np.median(ext))
    amplitude = min(center, max(float(np.quantile(ext, .75) - np.quantile(ext, .25)), 1e-6))
    if not np.isfinite([scale, center, amplitude]).all() or center < 0:
        raise RuntimeError('invalid nonnegative adapter')
    return dict(scale=scale, center=center, amplitude=amplitude,
                meaning='nonnegative bounded adapter of signed dB-dM; not original External distance')


def transform(formal, raw, bank, prep, adapter):
    z, roles = split_raw(raw, prep['mean'], prep['std'])
    d, mask = contrast(z, roles, bank)
    out = []
    for base, delta, valid in zip(formal, d, mask):
        a = np.asarray(base, np.float32).copy()
        replacement = adapter['center'] + adapter['amplitude'] * np.tanh(delta / adapter['scale'])
        replacement[~valid] = adapter['center']
        a[:, 107] = np.maximum(replacement, 0)
        a[:, 110] = (1 - np.clip(a[:, 109], 0, 1))*np.sqrt(
            np.maximum(a[:, 107], 0)*np.maximum(a[:, 108], 0))
        if a.shape[1] != 142 or not np.isfinite(a).all():
            raise RuntimeError('RCDB invalid 142D Host input')
        if not np.array_equal(a[:, 108:110], np.asarray(base)[:, 108:110]):
            raise RuntimeError('Shared/Internal branch changed')
        out.append(a)
    return out
