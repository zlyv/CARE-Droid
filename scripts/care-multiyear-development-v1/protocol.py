"""Rolling development windows, deterministic SHA deduplication and leakage guards."""
from __future__ import annotations
import numpy as np
WINDOWS={'T1':((2017,),2018),'T2':((2017,2018),2019),'T3':((2017,2018,2019),2020),'T4':((2017,2018,2019,2020),2021)}
def deduplicate(records):
    grouped={}
    for r in records:grouped.setdefault(r['sha256'],[]).append(r)
    kept=[];excluded=[];conflicts=[]
    for sid,rows in grouped.items():
        if len({int(r['label']) for r in rows})!=1:conflicts.append(dict(sha256=sid,rows=rows));continue
        rows=sorted(rows,key=lambda r:(int(r['year']),int(r['order'])))
        kept.append(rows[0]);excluded.extend(dict(r,reason='same SHA retained in earliest year',retained_year=rows[0]['year']) for r in rows[1:])
    if conflicts:raise RuntimeError('same-SHA label conflicts: '+str(conflicts[:5]))
    return sorted(kept,key=lambda r:(r['year'],r['order'])),excluded
def split_window(records,window):
    train_years,val_year=WINDOWS[window]
    train=np.asarray([i for i,r in enumerate(records) if r['year'] in train_years],int)
    val=np.asarray([i for i,r in enumerate(records) if r['year']==val_year],int)
    audit_scope(records,train,val,train_years,val_year)
    return train,val
def audit_scope(records,train,val,train_years,val_year):
    if not len(train) or not len(val):raise RuntimeError('empty rolling train/validation cohort')
    tr={records[i]['sha256'] for i in train};va={records[i]['sha256'] for i in val}
    if tr&va or set(train)&set(val):raise RuntimeError('train/validation SHA overlap')
    if any(records[i]['year'] not in train_years or records[i]['year']>=val_year for i in train):raise RuntimeError('validation/later year enters training')
    if any(records[i]['year']!=val_year for i in val):raise RuntimeError('validation year mismatch')
def mean_opposition(member,anchor):
    a=np.asarray(member,float);p=np.asarray(anchor,float)
    if a.shape!=(len(p),4) or not np.isfinite(a).all() or not np.isfinite(p).all():raise RuntimeError('invalid committee/anchor')
    if not np.allclose(a[:,3],p,rtol=0,atol=1e-14):raise RuntimeError('committee SVM is not current anchor')
    return np.where((p>=.5)[:,None],1-a,a).mean(axis=1)
def top_budget(scores,ids,budget=.25):
    scores=np.asarray(scores,float);ids=np.asarray(ids).astype(str)
    if len(set(ids))!=len(ids) or not np.isfinite(scores).all():raise RuntimeError('invalid ranking inputs')
    mask=np.zeros(len(ids),bool);mask[np.lexsort((ids,-scores))[:int(round(len(ids)*budget))]]=True
    return mask
def year_balanced_order(years,rng):
    years=np.asarray(years);unique=np.unique(years);n=len(years)
    if len(unique)==1:return rng.permutation(n)
    counts=np.full(len(unique),n//len(unique),int);counts[rng.permutation(len(unique))[:n%len(unique)]]+=1
    order=np.concatenate([rng.choice(np.flatnonzero(years==y),int(k),replace=True) for y,k in zip(unique,counts)])
    rng.shuffle(order);return order
