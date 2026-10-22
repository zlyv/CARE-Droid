#!/usr/bin/env python3
from __future__ import annotations
import unittest
import numpy as np
from binding_core import *

def event(eid, host=1, target=9, role_ids=None, tier='VERIFIED_CONTROL', bypass=False):
    return dict(event_id=eid, host_node_id=host, endpoint_node_id=target, role_ids=role_ids or [0],
        frozen_aligned=True, evidence_tier=tier,
        controls=[dict(event_postdominates_allowing_successor=True, blocked_successors_cannot_reach_event=True,
                       branch_dominates_event=not bypass, bypass_path_possible=bypass)] if tier in VERIFIED else [])

def fixture():
    f = dict(pairs={'1:9':[0], '2:9':[0]}, host_ids=['1','2'])
    record = dict(events=[event('a'), event('b',host=2,tier='PARTIAL')])
    t = proof_tables(record, f)
    context = {'1':np.r_[1.,np.zeros(73)], '2':np.zeros(74)}
    return t, context

class Tests(unittest.TestCase):
    def test_unknown_is_not_unguarded(self):
        t = proof_tables(dict(events=[event('a',tier='UNKNOWN')]),dict(pairs={'1:9':[0]}))
        a,b = aggregates(t)
        self.assertFalse(a.any()); self.assertEqual(b[31],1); self.assertEqual(b[-1],1)
    def test_missing_edge_is_not_observed(self):
        t = proof_tables(dict(events=[]),dict(pairs={'1:9':[0]}))
        a,b = aggregates(t)
        self.assertFalse(a.any()); self.assertFalse(b.any()); self.assertFalse(t['observed'].any())
    def test_empty_apk(self):
        t = proof_tables(dict(events=[]),dict(pairs={}))
        a,b = aggregates(t)
        self.assertEqual(a.shape,(124,));self.assertEqual(b.shape,(64,))
        s,d = summarize_repeat(t,{},'empty',0,1,1e-10)
        self.assertFalse(s['binding_ready']);self.assertEqual(len(d),0)
    def test_verified_needs_proof(self):
        e=event('a');e['controls']=[]
        with self.assertRaises(RuntimeError):proof_tables(dict(events=[e]),dict(pairs={'1:9':[0]}))
    def test_bad_control_rejected(self):
        e=event('a');e['controls'][0]['event_postdominates_allowing_successor']=False
        with self.assertRaises(RuntimeError):proof_tables(dict(events=[e]),dict(pairs={'1:9':[0]}))
    def test_duplicate_id_rejected(self):
        with self.assertRaises(RuntimeError):proof_tables(dict(events=[event('a'),event('a')]),dict(pairs={'1:9':[0]}))
    def test_role_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):proof_tables(dict(events=[event('a',role_ids=[1])]),dict(pairs={'1:9':[0]}))
    def test_large_uint64_host_id_not_float_cast(self):
        n=18446744073709551613
        t=proof_tables(dict(events=[event('a',host=n)]),dict(pairs={f'{n}:9':[0]}))
        self.assertEqual(t['hosts'][0],str(n))
    def test_same_marginals_different_binding(self):
        t,x=fixture();d,e=permutation(t,10)
        self.assertEqual(d.tolist(),[1,0]);self.assertTrue(e.all())
        for a,b in zip(aggregates(t),aggregates(t,t['flags'][d])):self.assertTrue(np.array_equal(a,b))
        self.assertGreater(np.max(np.abs(joint_descriptor(t,x)-joint_descriptor(t,x,t['flags'][d]))),0)
    def test_heterogeneous_proof_with_identical_host_context_is_not_ready(self):
        t,x=fixture();x['1']=x['2'].copy()
        s,_=summarize_repeat(t,x,'test',0,3,1e-10)
        self.assertEqual(s['changed_pairs'],2);self.assertFalse(s['binding_ready'])
    def test_homogeneous_facts_not_fake_mobility(self):
        t,x=fixture();t['flags'][1]=t['flags'][0]
        s,_=summarize_repeat(t,x,'test',0,3,1e-10)
        self.assertEqual(s['moved_pairs'],2);self.assertEqual(s['changed_pairs'],0);self.assertFalse(s['binding_ready'])
    def test_same_host_forbidden(self):
        t=proof_tables(dict(events=[event('a'),event('b',target=10,tier='PARTIAL')]),dict(pairs={'1:9':[0],'1:10':[0]}))
        d,e=permutation(t,1);self.assertEqual(d.tolist(),[0,1]);self.assertFalse(e.any())
    def test_exact_role_set_not_just_dominant_role(self):
        t=proof_tables(dict(events=[event('a'),event('b',host=2,role_ids=[0,1],tier='PARTIAL')]),dict(pairs={'1:9':[0],'2:9':[0,1]}))
        d,_=permutation(t,1);self.assertEqual(d.tolist(),[0,1])
    def test_observation_profile_not_changed(self):
        t=proof_tables(dict(events=[event('a'),event('b',host=2),event('c',host=2,tier='UNKNOWN')]),dict(pairs={'1:9':[0],'2:9':[0]}))
        d,_=permutation(t,1);self.assertEqual(d.tolist(),[0,1])
    def test_separate_unknown_stratum(self):
        t=proof_tables(dict(events=[event('a'),event('b',host=2,tier='UNKNOWN')]),dict(pairs={'1:9':[0],'2:9':[0]}))
        d,_=permutation(t,1);self.assertEqual(d.tolist(),[0,1])
    def test_unequal_host_pool_leaves_honest_residual(self):
        events=[event(str(i),target=i+10,tier='PARTIAL') for i in range(5)]+[event('b',host=2,target=20)]
        f=dict(pairs={f'1:{i+10}':[0] for i in range(5)});f['pairs']['2:20']=[0]
        t=proof_tables(dict(events=events),f);d,e=permutation(t,17)
        self.assertEqual(int((d!=np.arange(len(d))).sum()),2);self.assertTrue(e.all())
    def test_permutation_reproducible(self):
        t,_=fixture();self.assertTrue(np.array_equal(permutation(t,42)[0],permutation(t,42)[0]))
    def test_bad_donor_rejected(self):
        t,_=fixture()
        with self.assertRaises(RuntimeError):validate_permutation(t,[1,1])
    def test_exact_api_names_and_apk_labels_unused(self):
        t,x=fixture();f=dict(pairs={'1:9':[0],'2:9':[0]})
        es=[event('a'),event('b',host=2,tier='PARTIAL')]
        es[0]['sensitive_reference']='anything';es[1]['label']=1
        u=proof_tables(dict(events=es,label=0),f)
        self.assertTrue(np.array_equal(t['flags'],u['flags']))
        self.assertTrue(np.array_equal(joint_descriptor(t,x),joint_descriptor(u,x)))
    def test_host_context_alignment_and_padding(self):
        p=np.zeros(31);p[0]=1
        r=dict(host_id=1,presence=p,endpoint_counts=p,directed_neighbor_role=np.zeros(31))
        v=np.zeros((1,169));v[0,:31]=p;v[0,43:74]=p;v[0,31]=100
        c=host_context(v,[r],np.arange(31));self.assertEqual(c['1'].shape,(74,));self.assertLess(c['1'][31],1)
        v[0,105]=1
        with self.assertRaises(RuntimeError):host_context(v,[r],np.arange(31))
    def test_counterfeit_host_order_rejected(self):
        p=np.zeros(31);p[1]=1
        r=dict(host_id=1,presence=p,endpoint_counts=p,directed_neighbor_role=np.zeros(31))
        with self.assertRaises(RuntimeError):host_context(np.zeros((1,105)),[r],np.arange(31))
    def test_randomized_invariants_multirole_missing_duplicates(self):
        es=[];pairs={}
        for h in range(1,8):
            for j in range(h):
                target=100+h*10+j; rr=[0,3] if j%2 else [0]
                pairs[f'{h}:{target}']=rr
                if j%4!=3:
                    es.append(event(f'{h}:{j}',host=h,target=target,role_ids=rr,tier='PARTIAL' if h%2 else 'VERIFIED_CONTROL',bypass=bool(h%3)))
        t=proof_tables(dict(events=es),dict(pairs=pairs))
        for seed in range(50):
            d,_=permutation(t,seed);validate_permutation(t,d)
            self.assertTrue(np.array_equal(d[~t['observed']],np.arange(len(d))[~t['observed']]))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)
    print(f'HCB_SELFTEST_PASS tests={result.testsRun} labels_used=false training=false')
