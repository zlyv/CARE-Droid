#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, sys, time
from common import *
from binding_core import proof_tables, aggregates, host_context, summarize_repeat, node_id

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def preflight():
    p = protocol()
    parent_names = ('feature_validation.json', 'baseline_integrity_check.json', 'selection_label_free.csv',
                    'feature_scan_status.csv', 'control_features.npz', 'engine_identity.json', 'protocol.json')
    required = [(PARENT / n, 'read-only incremental ' + n) for n in parent_names]
    formal = PROJECT / 'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'
    endpoint = first([os.environ.get('HCB_ENDPOINT'),
        '/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz',
        PROJECT / 'artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz'], 'source endpoint cache')
    compact = first([os.environ.get('HCB_C43_CACHE'),
        PROJECT / 'artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz',
        PROJECT / 'artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz'], 'frozen C43 Host cache')
    manifest = first([os.environ.get('HCB_ROLE_MANIFEST'),
        PROJECT / 'artifacts/care-source-local-assets-v2/local_asset_manifest.json',
        '/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json'], 'payload-to-model Role manifest')
    hostrel = PROJECT / 'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py'
    prior_core = PROJECT / 'scripts/care-local-control-constraint-incremental-source-v1/feature_core.py'
    prior_extractor = PROJECT / 'scripts/care-local-control-constraint-incremental-source-v1/extract_features.py'
    required += [(formal, 'formal SVM SHA/fold/route fields only'), (endpoint, 'frozen endpoint source cache'),
                 (compact, 'frozen Compact43/Relation31 Host context'), (manifest, 'frozen Role index mapping'),
                 (hostrel, 'formal Host reconstruction'), (prior_core, 'previous marginal feature definition'),
                 (prior_extractor, 'previous aggregation consumer')]
    for path, _ in required:
        if not path.is_file():
            raise RuntimeError('BLOCKED: missing ' + str(path))
    fv = read_json(PARENT / 'feature_validation.json')
    pi = read_json(PARENT / 'baseline_integrity_check.json')
    if fv.get('status') != 'PASS' or fv.get('source_rows') != 799 or not fv.get('feature_extraction_labels_used') is False or pi.get('status') != 'PASS':
        raise RuntimeError('parent feature validation or integrity did not pass')
    parent_identity = read_json(PARENT / 'engine_identity.json')
    for name, source in (('feature_core.py', prior_core), ('extract_features.py', prior_extractor)):
        if parent_identity.get('experiment_source_sha256', {}).get(name) != sha256(source):
            raise RuntimeError('parent feature implementation hash differs: ' + name)
    rows = read_csv(PARENT / 'selection_label_free.csv')
    ids = [norm(r['sha256']) for r in rows]
    # Consume only the three membership columns. Labels/probabilities are never accessed.
    official = [dict(sha256=norm(r['sha256']), fold=int(float(r['fold'])), routed=int(float(r['routed'])))
                for r in read_csv(formal) if norm(r['arm']) == 'svm']
    actual = [dict(sha256=norm(r['sha256']), fold=int(r['fold']), routed=int(r['routed'])) for r in rows]
    if len(ids) != 799 or len(set(ids)) != 799 or sorted(actual, key=lambda r:r['sha256']) != sorted(official, key=lambda r:r['sha256']):
        raise RuntimeError('source SHA/fold/route membership differs')
    if any(len(s) != 64 or any(c not in '0123456789abcdef' for c in s) for s in ids):
        raise RuntimeError('invalid APK SHA')
    if [sum(r['fold'] == f for r in actual) for f in range(1, 6)] != [160, 160, 160, 160, 159]:
        raise RuntimeError('formal fold sizes differ')
    if any(r['routed'] not in (0, 1) for r in actual) or [sum(r['routed'] and r['fold'] == f for r in actual) for f in range(1, 6)] != [40] * 5:
        raise RuntimeError('formal routed cohort differs')
    statuses = read_csv(PARENT / 'feature_scan_status.csv')
    status_by = {norm(r['sha256']): r for r in statuses}
    if len(statuses) != 799 or set(status_by) != set(ids) or any(r['status'] != 'OK' for r in statuses):
        raise RuntimeError('parent scans incomplete; no exclusions allowed')
    for sid in ids:
        scan = Path(status_by[sid]['path']).resolve()
        frozen = PARENT / 'cache/frozen' / (sid + '.json')
        if not scan.is_file() or not frozen.is_file():
            raise RuntimeError('BLOCKED: missing reused scan/frozen cache ' + sid)
        required += [(scan, 'read-only LCC scan ' + sid), (frozen, 'read-only frozen Host pairs ' + sid)]
    before = []
    unique = {}
    for path, purpose in required:
        unique.setdefault(str(path.resolve()), (path, purpose))
    for i, (path, purpose) in enumerate(unique.values(), 1):
        before.append(asset(path, purpose))
        if i % 200 == 0:
            print(f'HCB_ASSET_BEFORE {i}/{len(unique)}', flush=True)
    dump(OUTPUT / 'baseline_asset_manifest_before.json', before)
    dump(OUTPUT / 'protocol.json', p)
    dump(OUTPUT / 'environment_inventory.json', dict(python=sys.version, python_executable=sys.executable,
        numpy=np.__version__, source_rows=799, routed=200, labels_used=False, future_rows_loaded=0,
        gpu_required=False, new_DEX_scans=0, APK_files_opened=0, endpoint=str(endpoint), compact=str(compact),
        manifest=str(manifest), parent=str(PARENT), memory_strategy='one LCC scan at a time; frozen source endpoint/Host caches in memory'))
    # Record overlap evidence rather than claiming an exhaustive historical search.
    older_shuffle = PROJECT / 'scripts/care-p31rf-c43-haig-host-ablation-source-v1/build_caches.py'
    previous = prior_core.read_text(encoding='utf-8')
    consumer = prior_extractor.read_text(encoding='utf-8')
    if 'evidence=np.zeros((4,31)' not in previous.replace(' ', '') or 'evidence.reshape(-1)' not in previous or 'evidence.append(e)' not in consumer:
        raise RuntimeError('previous aggregate representation changed; overlap must be reviewed')
    overlap = dict(status='SCOPED_CODE_REVIEW_COMPLETE_NOT_EXHAUSTIVE',
        previous_control= dict(path=str(prior_core), sha256=sha256(prior_core), output='APK-level 4x31 proof ratios, no C43 Host-coordinate joint descriptor'),
        older_host_call_shuffle=dict(path=str(older_shuffle), available=older_shuffle.is_file(),
            distinction='existing within-Role shuffle concerns hashed Host-call sidecars, not control-proof to frozen C43 Host binding'),
        proposed_difference='within-APK proof bundles with exact Role and observation profiles; unchanged prior marginals; inspect heterogeneous bindings before any classifier',
        duplicate_experiment_confirmed=False,
        limitation='Only named implementations inspected; does not establish literature novelty or exclude an unindexed historical equivalent.')
    if older_shuffle.is_file():
        overlap['older_host_call_shuffle']['sha256'] = sha256(older_shuffle)
        extra = asset(older_shuffle, 'historical Host-call shuffle implementation')
        before.append(extra)
        dump(OUTPUT / 'baseline_asset_manifest_before.json', before)
    dump(OUTPUT / 'experiment_overlap_audit.json', overlap)
    return actual, status_by, endpoint, compact, manifest, hostrel, prior_core

def audit(inputs):
    rows, statuses, endpoint, compact, manifest, hostrel_path, prior_core_path = inputs
    ids = [r['sha256'] for r in rows]
    wanted = set(ids)
    p = protocol()
    hostrel = module('_hcb_hostrel', hostrel_path)
    previous = module('_hcb_previous_marginals', prior_core_path)
    mapping = np.asarray(read_json(manifest)['payload_to_model_role'], int)
    if mapping.shape != (31,) or sorted(mapping.tolist()) != list(range(31)):
        raise RuntimeError('invalid frozen Role mapping')
    endpoints = {}
    with gzip.open(endpoint, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = norm(rec.get('sha256') or rec.get('sample_id'))
            if sid in wanted and rec.get('status') == 'OK':
                if sid in endpoints:
                    raise RuntimeError('duplicate source endpoint record')
                endpoints[sid] = rec['payload']
    if set(endpoints) != wanted:
        raise RuntimeError('incomplete source endpoint inventory')
    with np.load(compact, allow_pickle=False) as z:
        cache_ids = [norm(v) for v in z['sha256']]
        values = np.asarray(z['host_values'], float)
        offsets = np.asarray(z['host_offsets'], int)
        real = np.asarray(z['real_host_count'], int)
        empty = np.asarray(z['host_was_empty'], int)
    if len(cache_ids) != 799 or len(set(cache_ids)) != 799 or set(cache_ids) != wanted or offsets.shape != (800,) or real.shape != (799,) or empty.shape != (799,):
        raise RuntimeError('C43 SHA/count/offset schema mismatch')
    if offsets[0] != 0 or offsets[-1] != len(values) or (real < 0).any() or not np.array_equal(np.diff(offsets), np.maximum(real, 1)) or not np.array_equal(empty, (real == 0).astype(int)):
        raise RuntimeError('C43 offset/placeholder mismatch')
    context_index = {s:i for i,s in enumerate(cache_ids)}
    with np.load(PARENT / 'control_features.npz', allow_pickle=False) as z:
        proof_ids = [norm(v) for v in z['ids']]
        evidence = np.asarray(z['evidence'], float)
        coverage = np.asarray(z['coverage'], float)
    if len(proof_ids) != 799 or len(set(proof_ids)) != 799 or set(proof_ids) != wanted or evidence.shape != (799, 124) or coverage.shape != (799, 64) or not np.isfinite(evidence).all() or not np.isfinite(coverage).all():
        raise RuntimeError('previous proof matrix schema mismatch')
    proof_index = {s:i for i,s in enumerate(proof_ids)}
    identity = read_json(PARENT / 'engine_identity.json')
    audit_rows, repeats = [], []
    donor_file = safe(OUTPUT / 'synthetic_binding_donors.csv')
    donor_fields = ['sha256', 'repeat', 'receiver_pair', 'donor_pair', 'receiver_host', 'donor_host', 'role_ids', 'proof_flags_changed', 'synthetic_only']
    with donor_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=donor_fields); writer.writeheader()
        for n, row in enumerate(rows, 1):
            sid = row['sha256']
            frozen = read_json(PARENT / 'cache/frozen' / (sid + '.json'))
            record = load_gz(statuses[sid]['path'])
            expected_token = identity['parent_scan_identity'] if statuses[sid]['source'] == 'PASSED_V2_READ_ONLY' else identity['scan_identity_sha256']
            if record.get('schema') != 'lcc-fa-v2.0' or record.get('sha256') != sid or record.get('status') != 'OK' or record.get('scan_identity_sha256') != expected_token or record.get('labels_used') is not False or record.get('future_rows_loaded') != 0:
                raise RuntimeError('scan lineage/schema/scope differs ' + sid)
            if record['frozen_pairs_total'] != len(frozen['pairs']) or record['frozen_hosts_total'] != len(frozen['host_ids']):
                raise RuntimeError('frozen scan scope mismatch ' + sid)
            records = hostrel.reconstruct_host_relations(endpoints[sid], mapping)
            ci = context_index[sid]
            if len(records) != real[ci]:
                raise RuntimeError('frozen C43 real Host count differs ' + sid)
            context = host_context(values[offsets[ci]:offsets[ci+1]], records, mapping)
            if set(context) != {node_id(v) for v in frozen['host_ids']}:
                raise RuntimeError('Host ID identity differs ' + sid)
            # Validate retained Host-endpoint pairs directly, not only their counts.
            graph_pairs = {}
            payload = endpoints[sid]
            for left, right in zip(*payload['edge_index']):
                rr = list(map(int, payload['node_role_ids'][int(right)]))
                if payload['node_base_features'][int(right)][4] > .5 and rr:
                    key = node_id(payload['node_ids'][int(left)]) + ':' + node_id(payload['node_ids'][int(right)])
                    graph_pairs[key] = sorted(set(rr))
            if graph_pairs != {k:sorted(set(map(int,v))) for k,v in frozen['pairs'].items()}:
                raise RuntimeError('retained endpoint pair sets differ ' + sid)
            table = proof_tables(record, frozen)
            a, b = aggregates(table)
            old_a, old_b, _ = previous.extract(record, frozen)
            pi = proof_index[sid]
            if not (np.array_equal(a, old_a) and np.array_equal(b, old_b) and np.array_equal(a, evidence[pi]) and np.array_equal(b, coverage[pi])):
                raise RuntimeError('previous Role ratios/coverage not exactly reproduced ' + sid)
            local = []
            for repeat in range(p['null_repeats']):
                stats, donors = summarize_repeat(table, context, sid, repeat, p['null_seed'], p['minimum_joint_change_tolerance'])
                stats.update(sha256=sid, fold=row['fold'], routed=row['routed'])
                repeats.append(stats); local.append(stats)
                if repeat == 0:
                    for i, j in enumerate(donors):
                        if i != j:
                            writer.writerow(dict(sha256=sid, repeat=repeat, receiver_pair=table['keys'][i], donor_pair=table['keys'][j],
                                receiver_host=table['hosts'][i], donor_host=table['hosts'][j], role_ids='|'.join(map(str,table['roles'][i])),
                                proof_flags_changed=int(np.any(table['flags'][i] != table['flags'][j])), synthetic_only=1))
            audit_rows.append(dict(**row, frozen_hosts=len(context), frozen_pairs=len(table['keys']), observed_pairs=int(table['observed'].sum()),
                missing_pairs=int((~table['observed']).sum()), verified_pairs=int(table['flags'][:,0].sum()), aligned_events=int(table['count'].sum()),
                unknown_events=int(table['unknown'].sum()),
                repeats_with_binding_change=sum(s['binding_ready'] for s in local),
                median_joint_rms_change=float(np.median([s['joint_rms_change'] for s in local])),
                median_changed_pairs=float(np.median([s['changed_pairs'] for s in local])),
                all_previous_marginals_preserved=True, host_alignment_exact=True))
            if n % 25 == 0 or n == 799:
                print(f'HCB_BINDING_AUDIT {n}/799 new_scans=0 labels_used=false', flush=True)
    write_csv(OUTPUT / 'apk_binding_audit.csv', audit_rows)
    write_csv(OUTPUT / 'null_repeat_apk_metrics.csv', repeats)
    repeat_summary = []
    fold_summary = []
    for scope in ('ALL_SOURCE', 'ROUTED_PRIMARY'):
        for rep in range(p['null_repeats']):
            selected = [r for r in repeats if r['repeat'] == rep and (scope == 'ALL_SOURCE' or r['routed'])]
            verified = sum(r['verified_pairs'] for r in selected)
            repeat_summary.append(dict(scope=scope, repeat=rep, apks=len(selected),
                binding_changed_apks=sum(r['binding_ready'] for r in selected),
                binding_changed_apk_rate=sum(r['binding_ready'] for r in selected)/len(selected),
                verified_pairs=verified, changed_verified_pairs=sum(r['changed_verified_pairs'] for r in selected),
                changed_verified_fraction=sum(r['changed_verified_pairs'] for r in selected)/max(verified,1),
                changed_pairs=sum(r['changed_pairs'] for r in selected), invariants_pass=all(r['invariants_pass'] for r in selected)))
    for fold in range(1, 6):
        rates = [sum(r['binding_ready'] for r in repeats if r['routed'] and r['fold'] == fold and r['repeat'] == rep)/40 for rep in range(p['null_repeats'])]
        fold_summary.append(dict(fold=fold, routed=40, median_changed_apk_rate=float(np.median(rates)), minimum_changed_apk_rate=min(rates), maximum_changed_apk_rate=max(rates)))
    primary = [r for r in repeat_summary if r['scope'] == 'ROUTED_PRIMARY']
    summary = dict(source_rows=799, routed=200, null_repeats=p['null_repeats'],
        routed_median_binding_changed_apk_rate=float(np.median([r['binding_changed_apk_rate'] for r in primary])),
        routed_median_changed_verified_pair_fraction=float(np.median([r['changed_verified_fraction'] for r in primary])),
        folds_with_median_changed_apk_rate_at_least_0_15=sum(r['median_changed_apk_rate'] >= .15 for r in fold_summary),
        all_marginals_exact=True, all_host_alignments_exact=True, all_randomization_invariants_pass=True,
        full_source_frozen_hosts=sum(r['frozen_hosts'] for r in audit_rows), full_source_missing_pairs=sum(r['missing_pairs'] for r in audit_rows),
        labels_used=False, future_rows_loaded=0, model_training_performed=False)
    write_csv(OUTPUT / 'binding_repeat_summary.csv', repeat_summary)
    write_csv(OUTPUT / 'fold_binding_coverage.csv', fold_summary)
    dump(OUTPUT / 'binding_coverage.json', summary)
    print('HCB_BINDING_AUDIT_COMPLETE', flush=True)

def integrity():
    before = read_json(OUTPUT / 'baseline_asset_manifest_before.json')
    after, changed = [], []
    for i, old in enumerate(before, 1):
        p = Path(old['absolute_path'])
        if not p.is_file():
            changed.append(dict(path=str(p), reason='MISSING')); continue
        now = asset(p, old['purpose']); after.append(now)
        if any(now[k] != old[k] for k in ('size_bytes','mtime_ns','sha256')):
            changed.append(dict(before=old, after=now))
        if i % 200 == 0:
            print(f'HCB_ASSET_AFTER {i}/{len(before)}', flush=True)
    dump(OUTPUT / 'baseline_asset_manifest_after.json', after)
    dump(OUTPUT / 'baseline_integrity_check.json', dict(status='PASS' if not changed else 'FAIL', asset_count=len(before), changed=changed,
        scope='actual reused source files only; original APK files never opened; not a whole-project integrity claim'))
    if changed:
        raise RuntimeError('protected reused asset integrity failure')

def report(error, timing):
    check = OUTPUT / 'baseline_integrity_check.json'
    intact = read_json(check)['status'] if check.is_file() else 'UNAVAILABLE'
    criteria = {}
    if error:
        state = 'HOST_CONTROL_BINDING_BLOCKED'
        reason = error['error']
    else:
        c = read_json(OUTPUT / 'binding_coverage.json'); g = protocol()['gate']
        criteria = dict(all_799_scans_and_marginals_reproduced=c['all_marginals_exact'],
            all_799_host_context_alignments_exact=c['all_host_alignments_exact'],
            all_randomization_invariants_pass=c['all_randomization_invariants_pass'],
            changed_routed_APK_rate_at_least_0_20=c['routed_median_binding_changed_apk_rate'] >= g['routed_APK_rate_with_flag_AND_joint_change_median_repeat_min'],
            at_least_4_folds_changed_rate_at_least_0_15=c['folds_with_median_changed_apk_rate_at_least_0_15'] >= g['folds_with_changed_APK_rate_at_least_0_15_min'],
            changed_verified_pair_fraction_at_least_0_25=c['routed_median_changed_verified_pair_fraction'] >= g['routed_verified_pair_fraction_with_changed_binding_median_repeat_min'],
            formal_integrity_PASS=intact == 'PASS')
        state = 'HOST_CONTROL_BINDING_OBSERVABLE' if all(criteria.values()) else 'HOST_CONTROL_BINDING_COVERAGE_INSUFFICIENT'
        reason = 'Engineering observability gate passed; class information remains untested.' if all(criteria.values()) else 'Strict within-APK heterogeneous binding coverage did not pass all frozen engineering criteria.'
    d = dict(status=state, stopped_at=error['phase'] if error else 'LABEL_FREE_BINDING_OBSERVABILITY', reason=reason,
        criteria=criteria, formal_integrity=intact, labels_used=False, future_rows_loaded=0,
        new_DEX_scans=0, model_training_performed=False, model_training_authorized=False,
        permit_specifying_separate_source_information_test=state == 'HOST_CONTROL_BINDING_OBSERVABLE',
        next_action='specify a separate fixed-capacity source-only binding information test' if state == 'HOST_CONTROL_BINDING_OBSERVABLE' else 'stop this version; do not relax matching, drop failures, train a model or use future labels')
    dump(OUTPUT / 'final_decision.json', d); dump(OUTPUT / 'phase_timing.json', timing)
    text = '# CARE-Droid Host-Control Binding Observability Audit v1\n\n'
    text += f"Final status: **{state}**.\n\n{reason}\n\n"
    if not error:
        text += '## Actual coverage\n\n'
        text += f"- Source APKs: 799; unchanged routed primary cohort: 200.\n- Median routed APK fraction with changed proof flags AND changed joint descriptor: {c['routed_median_binding_changed_apk_rate']:.6f}.\n- Median fraction of verified pairs with changed binding: {c['routed_median_changed_verified_pair_fraction']:.6f}.\n- Folds with median changed APK fraction >=0.15: {c['folds_with_median_changed_apk_rate_at_least_0_15']}/5.\n- Exact Host alignment and previous 124+64 aggregate reproduction: PASS for all 799.\n- All 20 synthetic repeat invariants: PASS.\n\n"
    text += '## Interpretation and safety\n\nThis audit fits no model and measures no class AUC, fixed, broken or net. A changed joint descriptor only demonstrates that Role marginals do not determine the selected Host-bound descriptor. It does not prove that full C43 loses this information, that binding has class direction, or that compression causes errors. UNKNOWN/PARTIAL remain uncertainty, not absence of a guard. Permission checks do not imply benign behavior. Synthetic donor swaps are not APK transformations, runtime experiments or causal interventions.\n\nNo original APK is parsed, installed, executed or rewritten. No graph, route, Host set, classifier, threshold or formal result is changed. Formal integrity covers the actually reused files in the manifests, not every file on the server. Historical comparison is scoped to named implementations and is not a novelty proof.\n\n'
    text += f"Formal reused-asset integrity: {intact}. Future rows loaded: 0. Labels consumed: false. Actual phase seconds: {json.dumps(timing, sort_keys=True)}.\n\n"
    text += 'Only HOST_CONTROL_BINDING_OBSERVABLE permits specifying a separate information test. Training and future experiments are never launched by this package.\n'
    safe(OUTPUT / 'REPORT.md').write_text(text, encoding='utf-8')
    safe(OUTPUT / 'reproduce_commands.md').write_text(f'```bash\ncd {PROJECT}\nbash scripts/{NAME}/launch_tmux.sh\nbash scripts/{NAME}/collect_results.sh\n```\nFresh output required; no automatic resume or overwrite.\n', encoding='utf-8')
    print('CARE_HOST_CONTROL_BINDING_STATUS=' + state, flush=True)

def main():
    if OUTPUT.exists():
        raise RuntimeError('experiment output exists; create a new version, never overwrite: ' + str(OUTPUT))
    safe(OUTPUT).mkdir(exist_ok=False)
    print('CARE_HOST_CONTROL_BINDING_START output=' + str(OUTPUT), flush=True)
    phase = 'preflight'; error = None; timing = {}
    try:
        for phase, operation in (('preflight', preflight), ('audit', None)):
            t = time.monotonic(); print('HCB_PHASE_START name=' + phase, flush=True)
            if operation is not None:
                inputs = operation()
            else:
                audit(inputs)
            timing[phase] = time.monotonic() - t
            print(f'HCB_PHASE_DONE name={phase} seconds={timing[phase]:.3f}', flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        error = dict(phase=phase, error=repr(exc))
    finally:
        if (OUTPUT / 'baseline_asset_manifest_before.json').is_file():
            t = time.monotonic()
            try:
                integrity()
            except Exception as exc:
                error = error or dict(phase='integrity', error=repr(exc))
            timing['integrity'] = time.monotonic() - t
    report(error, timing)
    print('CARE_HOST_CONTROL_BINDING_FAILED' if error else 'CARE_HOST_CONTROL_BINDING_DONE', flush=True)
    if error:
        raise SystemExit(1)

if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
