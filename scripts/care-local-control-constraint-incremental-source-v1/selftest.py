#!/usr/bin/env python3
from __future__ import annotations
import unittest
import numpy as np
from feature_core import extract,apply_role_mask
from model_core import donor_permutation,conditional_shuffle,placebo,fit_predict
from statistics_core import strata_indices,paired_intervals
from common import protocol,metrics,auc

def event(eid,tier='VERIFIED_CONTROL',roles=None,bypass=False):
    return dict(event_id=eid,host_node_id=1,endpoint_node_id=2,role_ids=roles or [0],frozen_aligned=True,evidence_tier=tier,
        controls=[dict(event_postdominates_allowing_successor=True,blocked_successors_cannot_reach_event=True,branch_dominates_event=not bypass,bypass_path_possible=bypass)] if tier=='VERIFIED_CONTROL' else [])
class Tests(unittest.TestCase):
    def frozen(self,roles=None):return dict(pairs={'1:2':roles or [0]})
    def test_role_edge_proof_ratios(self):
        e,c,a=extract(dict(events=[event('a'),event('b','PARTIAL')]),self.frozen());e=e.reshape(4,31)
        self.assertEqual(e[:,0].tolist(),[1,0,1,0]);self.assertEqual(a['aligned_events'],2);self.assertEqual(c.shape,(64,))
    def test_unknown_never_becomes_unguarded(self):
        e,c,a=extract(dict(events=[event('a','UNKNOWN')]),self.frozen())
        self.assertFalse(e.any());self.assertEqual(c[31],1);self.assertEqual(c[-1],1)
    def test_empty_host_is_missing_not_error(self):
        e,c,a=extract(dict(events=[]),dict(pairs={}));self.assertFalse(e.any());self.assertFalse(c.any())
    def test_multiple_roles_consistent(self):
        e,c,a=extract(dict(events=[event('a',roles=[0,7])]),self.frozen([0,7]));self.assertEqual(e.reshape(4,31)[:,0].tolist(),e.reshape(4,31)[:,7].tolist())
    def test_bypass_is_fact_not_class_label(self):
        e,c,a=extract(dict(events=[event('a',bypass=True)]),self.frozen());self.assertEqual(e.reshape(4,31)[:,0].tolist(),[1,1,0,1])
    def test_duplicate_event_rejected(self):
        with self.assertRaises(RuntimeError):extract(dict(events=[event('a'),event('a')]),self.frozen())
    def test_invalid_proof_rejected(self):
        e=event('a');e['controls'][0]['blocked_successors_cannot_reach_event']=False
        with self.assertRaises(RuntimeError):extract(dict(events=[e]),self.frozen())
    def test_no_exact_API_feature(self):
        a=event('a');a['sensitive_reference']='arbitrary/exact/api';b={**a,'sensitive_reference':'completely/different/api'}
        self.assertTrue(np.array_equal(extract(dict(events=[a]),self.frozen())[0],extract(dict(events=[b]),self.frozen())[0]))
    def test_receiver_role_mask(self):
        c=np.zeros((2,64));c[0,0]=1;c[1,3]=1;out=apply_role_mask(np.ones((2,124)),c).reshape(2,4,31)
        self.assertEqual(int(np.count_nonzero(out)),8);self.assertFalse(out[0,:,3].any())
    def test_donors_stay_in_groups(self):
        g=np.array([[0,0],[0,0],[1,1],[1,1],[9,9]]);d,m=donor_permutation(g,np.random.default_rng(11))
        self.assertTrue(np.array_equal(g,g[d]));self.assertFalse(m[-1]);self.assertEqual(d[-1],4);self.assertTrue(np.all(d[:4]!=np.arange(4)))
    def test_shuffle_independent_repeat(self):
        e=np.arange(10*124).reshape(10,124)/2000;c=np.zeros((10,64));c[:,:31]=1;g=np.zeros((10,4),int)
        a=conditional_shuffle(e,c,g,7)[0];b=conditional_shuffle(e,c,g,7)[0];self.assertTrue(np.array_equal(a,b))
    def test_placebo_does_not_use_test_labels_or_moments(self):
        e=np.tile(np.linspace(0,1,124),(5,1));c=np.zeros((5,64));c[:,:31]=1
        a,b=placebo(e,c,c,np.array(list('abcde')),np.array(list('fghij')),3);self.assertEqual(a.shape,b.shape);self.assertTrue(np.isfinite(b).all())
    def test_fixed_readout_works(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(40,8));y=(x[:,0]>0).astype(int)
        p=fit_predict(x,y,x[:7],protocol()['auxiliary_model']);self.assertEqual(p.shape,(7,));self.assertTrue(((p>=0)&(p<=1)).all())
    def test_paired_identity_intervals_zero(self):
        n=40;y=np.arange(n)%2;f=np.arange(n)%5+1;base=np.where(np.arange(n)%3,0.7,0.3);route=np.ones(n,bool);prob=np.linspace(.1,.9,n)
        keys=['M0_DIRECT_C43','M1_HIDDEN_READOUT','M2_HIDDEN_CONTROL','M3_CAPACITY_PLACEBO__r00','M4_CONDITIONAL_SHUFFLE__r00']
        rows=paired_intervals(y,f,base,route,{k:prob.copy() for k in keys},30,77)
        self.assertEqual(len(rows),10);self.assertTrue(all(r['multiplicity_adjusted_low']==0 and r['difference']==0 for r in rows))
    def test_APK_strata_not_events(self):
        y=np.arange(40)%2;f=np.arange(40)%5+1;groups=strata_indices(y,f,np.linspace(0,1,40),np.ones(40,bool))
        self.assertEqual(sum(map(len,groups)),40);self.assertEqual(len(set(np.concatenate(groups))),40)
    def test_route_metrics(self):
        y=np.array([0,1,0,1]);base=np.array([.9,.1,.1,.9]);final=np.array([.1,.9,.9,.9]);r=metrics(y,base,final,np.ones(4,bool))
        self.assertEqual((r['fixed'],r['broken'],r['net']),(2,1,1));self.assertEqual(auc(y,np.array([.1,.9,.1,.9])),1)
    def test_protocol_has_no_selection(self):
        p=protocol();self.assertEqual(p['auxiliary_model']['C'],.1);self.assertEqual(p['threshold'],.5);self.assertIn('NONE',p['model_selection']);self.assertIn('permission features',p['prohibited'])
if __name__=='__main__':unittest.main(verbosity=2)
