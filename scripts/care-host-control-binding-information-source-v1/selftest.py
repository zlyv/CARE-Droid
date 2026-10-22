#!/usr/bin/env python3
import unittest
import numpy as np
from common import protocol,auc
from model_core import projection_map,sketch,variable_proof_mask,placebo,fit_predict
from statistics_core import endpoints,strata_indices,paired_intervals,gate,PRIMARY,COMPARATORS

class Tests(unittest.TestCase):
    def test_fixed_projection_reproducible(self):
        b,s=projection_map(64,20261016);bb,ss=projection_map(64,20261016)
        self.assertEqual(len(b),9176);self.assertTrue(np.array_equal(b,bb));self.assertTrue(np.array_equal(s,ss))
        self.assertEqual(set(b),set(range(64)));self.assertEqual(set(s),{-1.,1.})
    def test_sketch_zero(self):
        b,s=projection_map(64,1);self.assertTrue(np.array_equal(sketch(np.zeros((31,4,74)),b,s,64),np.zeros(64)))
    def test_sketch_linear(self):
        b,s=projection_map(64,1);rng=np.random.default_rng(11);a=rng.normal(size=(31,4,74));c=rng.normal(size=a.shape)
        self.assertTrue(np.allclose(sketch(a+c,b,s,64),sketch(a,b,s,64)+sketch(c,b,s,64)))
    def test_shape_rejected(self):
        b,s=projection_map(64,1)
        with self.assertRaises(RuntimeError):sketch(np.zeros(100),b,s,64)
    def test_variable_proof_support(self):
        e=np.zeros((3,124));e[0,0]=1;e[1,0]=.5
        self.assertEqual(variable_proof_mask(e).tolist(),[False,True,False])
    def test_placebo_has_no_test_moments(self):
        x=np.arange(6*64).reshape(6,64)/1000;ids=np.array(list('abcdef'));test=np.array(list('ghi'))
        a,b=placebo(x,ids,test,np.ones(6,bool),np.array([True,False,True]),4)
        aa,bb=placebo(x,ids,test,np.ones(6,bool),np.array([True,False,True]),4)
        self.assertTrue(np.array_equal(a,aa));self.assertTrue(np.array_equal(b,bb));self.assertFalse(b[1].any())
        self.assertEqual(b.shape,(3,64));self.assertFalse(np.array_equal(b[0],b[2]))
    def test_fit_uses_fixed_readout(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(40,7));y=(x[:,0]>0).astype(int)
        a=fit_predict(x,y,x[:6],protocol()['auxiliary_model'])
        self.assertEqual(a.shape,(6,));self.assertTrue(((a>=0)&(a<=1)).all())
    def test_nan_fit_rejected(self):
        with self.assertRaises(RuntimeError):fit_predict(np.array([[0],[np.nan]]),[0,1],np.array([[0]]),protocol()['auxiliary_model'])
    def test_one_class_conditional_side_is_unavailable(self):
        y=np.array([0,1,0,1]);base=np.array([.1,.9,.1,.9]);p=np.array([.2,.8,.3,.7]);route=np.ones(4,bool)
        ep=endpoints(y,p,base,route);self.assertIsNone(ep['FN_vs_TN_AUC']);self.assertIsNone(ep['TP_vs_FP_AUC'])
    def test_APKs_not_hosts_are_bootstrap_units(self):
        y=np.arange(40)%2;folds=np.arange(40)%5+1;base=np.linspace(0,1,40);route=np.arange(40)%3==0
        groups=strata_indices(y,folds,base,route)
        self.assertEqual(sum(map(len,groups)),40);self.assertEqual(len(set(np.concatenate(groups))),40)
    def test_identity_paired_intervals_exact_zero(self):
        n=40;y=np.arange(n)%2;f=np.arange(n)%5+1;base=np.where(np.arange(n)%3,.8,.2);route=np.ones(n,bool);pred=np.linspace(.1,.9,n)
        scores={a:pred.copy() for a in COMPARATORS+('B0_DIRECT_C43',PRIMARY)}
        rows=paired_intervals(y,f,base,route,scores,30,1)
        self.assertEqual(len(rows),16);self.assertEqual(sum(r['primary_family'] for r in rows),10)
        self.assertTrue(all(r['difference']==0 and r['paired_95_low']==0 and r['paired_95_high']==0 for r in rows))
        self.assertTrue(all(r['multiplicity_adjusted_low']==0 for r in rows if r['primary_family']))
    def test_gate_stops_when_controls_equal_real(self):
        names=COMPARATORS+('B0_DIRECT_C43',PRIMARY)
        value=dict(BA=.9,FN_vs_TN_AUC=.8,TP_vs_FP_AUC=.8,fixed=49,broken=17,net=32)
        points={a:value.copy() for a in names};fps={(a,f):value.copy() for a in names for f in range(1,6)}
        ims=[dict(comparator=a,endpoint=e,multiplicity_adjusted_low=0) for a in COMPARATORS for e in ('BA','FN_vs_TN_AUC','TP_vs_FP_AUC')]+[dict(comparator='B0_DIRECT_C43',endpoint='BA',multiplicity_adjusted_low=0)]
        null={a:{k:value[k] for k in ('BA','FN_vs_TN_AUC','TP_vs_FP_AUC')} for a in ('B3_WITHIN_APK_SHUFFLE','B4_CAPACITY_PLACEBO')}
        d=gate(points,fps,fps,ims,null,protocol(),True)
        self.assertEqual(d['status'],'HOST_BINDING_INFORMATION_NOT_SUPPORTED');self.assertFalse(d['permit_specifying_separate_frozen_diagnostic'])
        d=gate(points,fps,fps,ims,null,protocol(),False);self.assertEqual(d['status'],'HOST_BINDING_INFORMATION_INCONCLUSIVE')
    def test_gate_requires_both_sides_and_controlled_broken(self):
        names=COMPARATORS+('B0_DIRECT_C43',PRIMARY)
        base=dict(BA=.90,FN_vs_TN_AUC=.7,TP_vs_FP_AUC=.7,fixed=49,broken=17,net=32)
        real=dict(BA=.95,FN_vs_TN_AUC=.85,TP_vs_FP_AUC=.85,fixed=55,broken=10,net=45)
        points={a:base.copy() for a in names};points[PRIMARY]=real
        fps={(a,f):points[a].copy() for a in names for f in range(1,6)}
        ims=[dict(comparator=a,endpoint=e,multiplicity_adjusted_low=.01) for a in COMPARATORS for e in ('BA','FN_vs_TN_AUC','TP_vs_FP_AUC')]+[dict(comparator='B0_DIRECT_C43',endpoint='BA',multiplicity_adjusted_low=.01)]
        null={a:{k:base[k] for k in ('BA','FN_vs_TN_AUC','TP_vs_FP_AUC')} for a in ('B3_WITHIN_APK_SHUFFLE','B4_CAPACITY_PLACEBO')}
        d=gate(points,fps,fps,ims,null,protocol(),True);self.assertEqual(d['status'],'HOST_BINDING_INFORMATION_SUPPORTED_SOURCE_ONLY')
        points[PRIMARY]['broken']=18;d=gate(points,fps,fps,ims,null,protocol(),True)
        self.assertEqual(d['status'],'HOST_BINDING_DIRECTION_SUPPORTED_INTEGRATION_NOT_CONFIRMED');self.assertFalse(d['permit_specifying_separate_frozen_diagnostic'])
        points[PRIMARY]['TP_vs_FP_AUC']=.70;d=gate(points,fps,fps,ims,null,protocol(),True)
        self.assertEqual(d['status'],'PARTIALLY_SUPPORTED')
    def test_protocol_has_no_search_or_future(self):
        p=protocol();self.assertEqual(p['projection_width'],64);self.assertEqual(p['auxiliary_model']['C'],.1)
        self.assertEqual(p['threshold'],.5);self.assertEqual(p['future_rows_loaded'],0);self.assertIn('NONE',p['model_selection'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)
    print(f'HBI_SELFTEST_PASS tests={result.testsRun} projection=64 no_hyperparameter_search=true')
