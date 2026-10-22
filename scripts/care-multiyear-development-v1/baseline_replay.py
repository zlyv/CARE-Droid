"""Check historical source assets and cascade arithmetic without claiming a new 799-APK neural retraining replay."""
import numpy as np
from myd_common import *
from metrics import cascade

def replay():
    first=PROJECT/'outputs/care-svm-c43-remove-bounded-correction-v1/source/U2_DIRECT_STAGE2/source_predictions.csv'
    second=PROJECT/'outputs/care-direct-c43-route-conditioned-training-v2/source/B0_ORIGINAL/source_predictions.csv'
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json');seen={r['path'] for r in before}
    for p in [first,second]:
        if not p.exists():dump(OUTPUT/'baseline_replay.json',dict(status='UNAVAILABLE',missing=str(p)));return
        if str(p.resolve()) not in seen:before.append(asset(p,'historical source Direct-C43 reference replay'))
    dump(OUTPUT/'baseline_asset_manifest_before.json',before)
    a=read_csv(first);b={norm(r['sha256']):r for r in read_csv(second)}
    def col(rows,suffix):
        keys=[k for k in rows[0] if k.endswith(suffix) and ('Direct' in k or 'B0' in k)]
        if len(keys)!=1:raise RuntimeError('ambiguous historical replay column '+repr(keys))
        return keys[0]
    ca=col(a,'__expert_prob');cb=col(list(b.values()),'__expert_prob');diff=max(abs(float(r[ca])-float(b[norm(r['sha256'])][cb])) for r in a)
    y=np.asarray([int(r['label']) for r in a]);base=np.asarray([float(r['Global24__prob']) for r in a]);e=np.asarray([float(r[ca]) for r in a]);route=np.asarray([int(r['A25__routed']) for r in a],bool)
    m=cascade(y,base,e,route);passed=diff<=1e-8 and m['Fixed']==49 and m['Broken']==17 and abs(m['BA']-.9186152882205514)<1e-12
    dump(OUTPUT/'baseline_replay.json',dict(status='ASSET_REPLAY_PASS' if passed else 'FAIL',scope='historical independently completed B0 source probabilities and cascade arithmetic; NOT new full neural training',prediction_max_difference=diff,rows=len(a),metrics=m,new_neural_baseline_retrained=False))
    if not passed:raise RuntimeError('historical Direct-C43 asset replay differs')
    print('MYD_BASELINE_ASSET_REPLAY_PASS',m['Fixed'],m['Broken'],flush=True)

if __name__=='__main__':replay()
