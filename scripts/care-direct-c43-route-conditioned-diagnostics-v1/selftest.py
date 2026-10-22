from __future__ import annotations
import argparse,ast,contextlib,io,sys,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import diag_common as dc
import run_mode as rm
from summarize import compare_rows,paired_intervals,outcome_totals

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=dc.PARENT_SCRIPT if dc.PARENT_SCRIPT.is_dir() else Path(__file__).resolve().parent.parent/'care-direct-c43-route-conditioned-training-v2'
        dc.PARENT_SCRIPT=root;cls.core,cls.rt,cls.w,cls.ta=dc.parent_modules()
    def test_frozen_and_within_seed_rules(self):
        a=SimpleNamespace(split_seed=20260904)
        self.assertEqual(rm.context_seed('frozen',2017,0,a),(20260904+1701,20261017))
        self.assertEqual(rm.context_seed('within',2021,5,a),(20260904+2021*10000+5*100+1701,20261022))
    def test_ordinary_outcome_matches_original_metrics(self):
        rng=np.random.RandomState(1);y=np.arange(120)%2;b=rng.uniform(.1,.9,120);p=rng.uniform(.1,.9,120);r=np.arange(120)%3==0
        fast=outcome_totals(y,b,p,r);m=self.core.metrics(y,b,p,r)
        for k in fast:self.assertAlmostEqual(fast[k],m[k])
    def rows(self):
        labels={};rows=[];counter=0
        for year,n in dc.COUNTS.items():
            for i in range(n):
                s=f'{counter:064x}';counter+=1;label=i%2;labels[s]=label;fold=i%5+1;base=.3 if label==0 else .7;route=int(i%4==0);expert=.55
                for arm in dc.ARMS:rows.append(dict(sha256=s,year=year,fold=fold,arm=arm,stage1_prob=base,routed=route,expert_prob=expert,final_prob=expert if route else base))
        return rows,labels
    def test_all_years_macro_and_pooled(self):
        rows,labels=self.rows();annual,folds,data=compare_rows('within',rows,labels,self.core)
        self.assertEqual(len(annual),24);self.assertEqual(len(folds),80);self.assertEqual(len(data[0]),1989)
        for arm in dc.ARMS:
            r=next(r for r in annual if r['arm']==arm and r['year']=='macro_year_mean')
            years=[r for r in annual if r['arm']==arm and r['year'] not in ('pooled','macro_year_mean')]
            self.assertAlmostEqual(r['Stage2_BA'],np.mean([r['Stage2_BA'] for r in years]))
            self.assertNotIn('fixed',r)
    def test_changed_deployment_route_is_rejected(self):
        rows,labels=self.rows();rows[1]['routed']=1-rows[1]['routed']
        with self.assertRaises(RuntimeError):compare_rows('frozen',rows,labels,self.core)
    def test_missing_arm_is_rejected(self):
        rows,labels=self.rows()
        with self.assertRaises(RuntimeError):compare_rows('frozen',rows[:-1],labels,self.core)
    def test_paired_identical_controls_zero_interval(self):
        rows,labels=self.rows();_,_,data=compare_rows('within',rows,labels,self.core)
        c=paired_intervals('within',data,self.core,repeats=24);self.assertEqual(len(c),45)
        for r in c:self.assertEqual(r['difference'],0);self.assertEqual(r['ci95_low'],0);self.assertEqual(r['ci95_high'],0)
    def test_original_source_weights_reused(self):
        n=160;rng=np.random.RandomState(3);y=np.arange(n)%2;ids=np.asarray([str(i) for i in range(n)]);p=rng.uniform(.1,.9,(n,4));c=rng.randint(1,60,n)
        w,r,a=self.w.make_weights(y,ids,p,c,seed=20261018,groups=np.arange(n)%4)
        self.assertAlmostEqual(w['B1_ROUTE_WEIGHTED'][r][0]/w['B1_ROUTE_WEIGHTED'][~r][0],4)
        self.assertAlmostEqual(w['B2_CLASS_PRIOR'][y==0].sum(),w['B1_ROUTE_WEIGHTED'][y==0].sum())
        self.assertTrue(np.array_equal(np.sort(w['B3_CONDITIONAL_SHUFFLE']),np.sort(w['B1_ROUTE_WEIGHTED'])))
    def test_four_arm_callback_training_only_and_same_epochs(self):
        n=160;m=8;rng=np.random.RandomState(4);member=rng.uniform(.1,.9,(n,4));y=np.arange(n)%2
        ids=np.asarray(['s'+str(i) for i in range(n)]);test=np.asarray(['t'+str(i) for i in range(m)]);x=np.column_stack([np.arange(n),np.ones((n,30))])
        root=(Path.cwd()/'work' if not dc.OUTPUT.exists() else dc.OUTPUT/'selftest-fixtures')/('rd'+uuid.uuid4().hex[:8]);root.mkdir(parents=True,exist_ok=False)
        previous=(dc.OUTPUT,rm.OUTPUT,self.core.OUTPUT,rm.aligned_p31,rm.read_csv)
        seen=[];seeds=[900000,900001,900002];args=SimpleNamespace(split_seed=20260904,jobs=1,keep_weight=1.,fix_weight=1.,keep_consistency=0.,batch_size=16)
        def fit_committee(fit,yfit,hold,seed,inner_folds,jobs):
            self.assertFalse(set(fit[:,0])&set(hold[:,0]));self.assertEqual(inner_folds,3)
            return member[hold[:,0].astype(int)]
        def original(*a):
            np.testing.assert_array_equal(a[8],y);return np.full(m,.55),np.ones((m,1)),[dict(seed=s,selected_epochs=2,parameter_n=7) for s in seeds]
        def fixed(v4,inputs,weights,records,seeds,args,device):
            np.testing.assert_array_equal(inputs[8],y);self.assertEqual(len(weights),n);self.assertEqual([r['selected_epochs'] for r in records],[2]*3)
            seen.append(weights.copy());a=[dict(seed=s,selected_epochs=2,parameter_n=7,optimizer_steps=20) for s in seeds]
            return np.full(m,.55),None,a,np.full((3,m),.55)
        state=rm.Expert.__new__(rm.Expert);state.mode='frozen';state.count=0;state.source_ids=ids;state.target_ids=test;state.source_y=y;state.contexts={}
        state.w=self.w;state.rt=SimpleNamespace(family_module=lambda name:SimpleNamespace(SOURCE_P31='unused',fit_calibrated_committee=fit_committee))
        state.ta=SimpleNamespace(validate_epoch_records=self.ta.validate_epoch_records,fingerprint=self.ta.fingerprint,fit_fixed_epochs=fixed)
        state.v4=SimpleNamespace(RoleGraphConditionedEvidence=SimpleNamespace(correction_semantics='direct_stage2_logit_without_stage1_logit_input'));state.original=original
        try:
            dc.OUTPUT=rm.OUTPUT=self.core.OUTPUT=root
            rm.aligned_p31=lambda path,wanted:x
            rm.read_csv=lambda path:[dict(sha256=s,direct_c43_prob=.55) for s in test]
            with contextlib.redirect_stdout(io.StringIO()):
                p,_,a=state.fit(np.zeros((n,2)),np.zeros((m,2)),np.zeros((n,2)),np.zeros((m,2)),
                    [np.ones((i%60+1,169)) for i in range(n)],[np.zeros((1,169))]*m,member[:,3],np.full(m,.4),y,'RGCondAttn',seeds,args,SimpleNamespace(type='cpu'))
            self.assertEqual(len(seen),3);self.assertEqual(state.count,1)
            saved=dc.read_json(root/'training_routes/frozen/train2017.json')
            for split in saved['crossfit']:self.assertFalse(set(split['fit_sha'])&set(split['hold_sha']))
            self.assertFalse(set(saved['train_sha'])&set(saved['test_sha']));self.assertTrue(np.all(p==.55))
        finally:dc.OUTPUT,rm.OUTPUT,self.core.OUTPUT,rm.aligned_p31,rm.read_csv=previous
    def test_within_replaces_bounded_parser_and_installer(self):
        from diagnostic_runtime import direct_installer,direct_parser
        calls=[]
        def a0(v,method):calls.append(('a0',method));v.head='bounded'
        def install(v,name):self.assertEqual(name,'U2_DIRECT_STAGE2');calls.append(('direct',name));v.head='direct'
        v=SimpleNamespace(head=None);direct_installer(SimpleNamespace(install_a0=a0),SimpleNamespace(install_variant=install))(v,'SVM')
        self.assertEqual(v.head,'direct');self.assertEqual(len(calls),2)
        def builder():
            p=argparse.ArgumentParser();p.add_argument('--keep-weight',type=float,default=2.);p.add_argument('--fix-weight',type=float,default=1.);p.add_argument('--keep-consistency',type=float,default=.5);return p
        p=direct_parser(builder)().parse_args([]);self.assertEqual((p.keep_weight,p.fix_weight,p.keep_consistency),(1.,1.,0.))
    def test_replay_drift_or_decision_change_stops(self):
        self.assertEqual(rm.replay_checks(np.array([.1,.8]),np.array([.1,.8]))['status'],'PASS')
        self.assertEqual(rm.replay_checks(np.array([.1,.8]),np.array([.2,.8]))['status'],'FAIL')
        self.assertEqual(rm.replay_checks(np.array([.49999]),np.array([.50001]))['status'],'FAIL')
    def test_no_source_status_promotion_in_summary(self):
        text=(dc.HERE/'summarize.py').read_text();self.assertIn("source_conclusion_reversed=False",text);self.assertIn("source_status='ROUTE_CONDITIONED_TRAINING_NOT_SUPPORTED'",text)

if __name__=='__main__':unittest.main(verbosity=2)
