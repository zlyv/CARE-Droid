from __future__ import annotations
import argparse, ast, contextlib, hashlib, io, json, os, sys, unittest, uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pilot_common as pc
from weighting import make_weights,opposition,top25
from prepare_routes import crossfit_committee
from summarize import decide,fast_auc,paired_bootstrap
from train_adapter import fingerprint,validate_epoch_records,fit_fixed_epochs,TrainingState
import train_adapter as ta
from runtime import wrap_source_loader,read_only_bags
import runtime as rt

def fixture_root():
    path=Path(os.environ.get('RCT_TEST_ROOT',str(Path.cwd()/'work')))/('rt'+uuid.uuid4().hex[:8])
    path.mkdir(parents=True,exist_ok=False)
    return path

class WeightTests(unittest.TestCase):
    def data(self,n=160):
        rng=np.random.RandomState(4); ids=np.asarray([f'{i:064x}' for i in range(n)]); y=np.arange(n)%2
        p=rng.uniform(.1,.9,(n,4)); counts=rng.randint(1,100,n)
        return y,ids,p,counts
    def test_opposition_original_formula(self):
        _,_,p,_=self.data(); from_expected=np.mean(np.where(p[:,3:4]>=.5,1-p,p),axis=1)
        np.testing.assert_array_equal(opposition(p),from_expected)
    def test_top25_ties_SHA(self):
        ids=np.asarray(['d','b','c','a']); np.testing.assert_array_equal(top25(np.ones(4),ids),[False,False,False,True])
    def test_real_and_controls(self):
        y,ids,p,c=self.data(); w,r,a=make_weights(y,ids,p,c)
        self.assertEqual(r.sum(),40); self.assertTrue(np.all(w['B0_ORIGINAL']==1)); self.assertAlmostEqual(w['B1_ROUTE_WEIGHTED'].mean(),1)
        self.assertAlmostEqual(w['B1_ROUTE_WEIGHTED'][r][0]/w['B1_ROUTE_WEIGHTED'][~r][0],4)
        for label in (0,1):
            self.assertEqual(len(np.unique(w['B2_CLASS_PRIOR'][y==label])),1)
            for arm in ('B2_CLASS_PRIOR','B3_CONDITIONAL_SHUFFLE'):
                self.assertAlmostEqual(w[arm][y==label].sum(),w['B1_ROUTE_WEIGHTED'][y==label].sum())
        self.assertGreater(a['changed_fraction'],.1)
        self.assertAlmostEqual(a['effective_sample_size']['B3_CONDITIONAL_SHUFFLE'],a['effective_sample_size']['B1_ROUTE_WEIGHTED'])
    def test_top25_separate_inner_holdouts(self):
        y,ids,p,c=self.data(); groups=np.repeat(np.arange(4),40)
        _,r,_=make_weights(y,ids,p,c,groups=groups)
        for g in range(4): self.assertEqual(int(r[groups==g].sum()),10)
    def test_row_order_invariance(self):
        y,ids,p,c=self.data(); a,_,_=make_weights(y,ids,p,c); ix=np.random.RandomState(6).permutation(len(y)); b,_,_=make_weights(y[ix],ids[ix],p[ix],c[ix])
        for arm in a: np.testing.assert_array_equal(a[arm][ix],b[arm])
    def test_no_self_error_label_real_weights(self):
        y,ids,p,c=self.data(); a,_,_=make_weights(y,ids,p,c); b,_,_=make_weights(1-y,ids,p,c)
        np.testing.assert_array_equal(a['B1_ROUTE_WEIGHTED'],b['B1_ROUTE_WEIGHTED'])
    def test_invalid_probabilities_rejected(self):
        y,ids,p,c=self.data(); p[0,1]=np.nan
        with self.assertRaises(ValueError): make_weights(y,ids,p,c)
    def test_hardcoded_protocol_no_search(self):
        p=pc.protocol(); self.assertEqual(p['route_weight'],4); self.assertFalse(p['automatic_future_run']); self.assertFalse(p['tuning_authorized'])

class CrossfitTests(unittest.TestCase):
    def test_excludes_each_hold_label(self):
        from sklearn.model_selection import StratifiedKFold
        n=48; x=np.arange(n).reshape(n,1); y=np.arange(n)%2; ids=np.asarray([str(i) for i in range(n)]); seen=[]
        def fit(xfit,yfit,target,seed,inner_folds,jobs):
            self.assertFalse(set(xfit[:,0])&set(target[:,0])); self.assertEqual(inner_folds,3)
            seen.append((xfit[:,0].tolist(),target[:,0].tolist())); return np.tile([.1,.2,.3,.4],(len(target),1))
        p,fold,a=crossfit_committee(x,y,ids,21,fit,StratifiedKFold(4,shuffle=True,random_state=21))
        self.assertEqual(len(seen),4); np.testing.assert_array_equal(np.unique(fold),[1,2,3,4]); self.assertEqual(p.shape,(48,4))
        for row in a: self.assertFalse(set(row['fit_sha'])&set(row['hold_sha']))
    def test_incomplete_crossfit_rejected(self):
        class Incomplete:
            def split(self,x,y): yield np.arange(4,8),np.arange(4)
        def fit(*args,**kw): return np.full((4,4),.5)
        with self.assertRaises(RuntimeError): crossfit_committee(np.zeros((8,1)),np.arange(8)%2,np.asarray([str(i) for i in range(8)]),1,fit,Incomplete())

class DecisionTests(unittest.TestCase):
    def input(self):
        pooled={a:dict(fixed=49,broken=17,net=32) for a in pc.ARMS}; pooled['B1_ROUTE_WEIGHTED']=dict(fixed=51,broken=15,net=36)
        rows=[]
        for a in pc.ARMS:
            for f in range(1,6): rows.append(dict(arm=a,fold=f,BA=.92 if a=='B1_ROUTE_WEIGHTED' else .90,FN_vs_TN_AUC=.82 if a=='B1_ROUTE_WEIGHTED' else .78,TP_vs_FP_AUC=.86 if a=='B1_ROUTE_WEIGHTED' else .82))
        c=[dict(comparison='B1-'+a,endpoint=e,difference=.02 if e=='end_to_end_BA' else .04,family_adjusted_low=.001) for a in ('B0_ORIGINAL','B2_CLASS_PRIOR','B3_CONDITIONAL_SHUFFLE') for e in pc.protocol()['primary_endpoints']]
        return pooled,rows,c
    def test_all_gates_required(self):
        po,fo,co=self.input(); d=decide(po,fo,co,pc.protocol()); self.assertTrue(d['pilot_passed']); self.assertFalse(d['automatic_future_run'])
    def test_null_wins_stop(self):
        po,fo,co=self.input(); co[-1]['difference']=-.01; co[-1]['family_adjusted_low']=-.02
        d=decide(po,fo,co,pc.protocol()); self.assertFalse(d['pilot_passed']); self.assertFalse(d['tuning_authorized'])
    def test_interval_crossing_zero_stops(self):
        po,fo,co=self.input(); co[0]['family_adjusted_low']=-.001; self.assertFalse(decide(po,fo,co,pc.protocol())['pilot_passed'])
    def test_broken_damage_stops(self):
        po,fo,co=self.input(); po['B1_ROUTE_WEIGHTED']['broken']=18; self.assertFalse(decide(po,fo,co,pc.protocol())['pilot_passed'])
    def test_one_fold_driver_stops(self):
        po,fo,co=self.input()
        for r in fo:
            if r['arm']=='B1_ROUTE_WEIGHTED' and r['fold']>1:r['BA']=.89
        self.assertFalse(decide(po,fo,co,pc.protocol())['pilot_passed'])
    def test_auc_ties_against_sklearn(self):
        from sklearn.metrics import roc_auc_score
        y=np.array([0,1,1,0]); score=np.array([.4,.4,.8,.9]); self.assertAlmostEqual(fast_auc(y,score),roc_auc_score(y,score))
    def test_paired_identical_bootstrap_zero(self):
        y=np.tile([0,1],60); f=np.repeat(np.arange(1,6),24); b=np.tile([.2,.7,.6,.4],30); r=np.ones(120,bool)
        pred={a:np.linspace(.1,.9,120) for a in pc.ARMS}; p=pc.protocol(); p['bootstrap_repeats']=24
        c=paired_bootstrap(y,f,b,r,pred,p)
        self.assertEqual(len(c),9)
        for v in c:self.assertEqual(v['difference'],0);self.assertEqual(v['family_adjusted_low'],0);self.assertEqual(v['family_adjusted_high'],0)

class TrainingTests(unittest.TestCase):
    def test_input_fingerprint_order_and_dtype(self):
        x=np.arange(8,dtype=np.float32)
        self.assertNotEqual(fingerprint([x]),fingerprint([x.astype(np.float64)])); self.assertNotEqual(fingerprint([x]),fingerprint([x[::-1]]))
    def test_epoch_seed_mismatch_rejected(self):
        with self.assertRaises(RuntimeError): validate_epoch_records([dict(seed=1,selected_epochs=4)],[2],1)
    def test_epoch_out_of_bound(self):
        with self.assertRaises(RuntimeError): validate_epoch_records([dict(seed=1,selected_epochs=31)],[1],1)
    def test_fixed_training_uses_weights_and_epoch_budget(self):
        calls=[]
        class TensorModule:
            float32='float32'
            @staticmethod
            def as_tensor(x,**kw): return np.asarray(x)
            optim=SimpleNamespace(Adam=lambda params,**kw:object())
        class Parameter:
            def numel(self): return 7
        class Model:
            def to(self,d): return self
            def parameters(self): return [Parameter()]
        def train(model,opt,gt,base,graph,bags,y,weights,correct,epochs,args,seed):
            calls.append(dict(y=y.copy(),weights=weights.copy(),epochs=epochs,seed=seed)); return .3
        v=SimpleNamespace(torch=TensorModule,v3=SimpleNamespace(fit_scaled_dualproto_bags=lambda a,b,y,args,seed:(a,b,{})),
            fit_role_graph_scaler=lambda a:None,transform_role_graphs=lambda a,s:a,graph_tensors=lambda a,d:a,
            seed_everything=lambda seed:None,RoleGraphConditionedEvidence=lambda *a:Model(),train_epochs=train,
            predict_model=lambda model,gv,base,graph,bags,d:(np.full(len(gv),.5),np.ones((len(gv),1))))
        n=12; y=np.arange(n)%2; inputs=(np.zeros((n,2)),np.zeros((4,2)),np.zeros((n,2)),np.zeros((4,2)),[np.zeros((1,169))]*n,[np.zeros((1,169))]*4,np.full(n,.4),np.full(4,.4),y,'RGCondAttn')
        weights=np.linspace(.5,2,n); args=SimpleNamespace(learning_rate=.0008,weight_decay=.0005,batch_size=4)
        p,d,a,seeds=fit_fixed_epochs(v,inputs,weights,[dict(seed=100000,selected_epochs=7)],[100000],args,SimpleNamespace(type='cpu'))
        np.testing.assert_array_equal(calls[0]['y'],y); np.testing.assert_array_equal(calls[0]['weights'],weights.astype(np.float32))
        self.assertEqual(calls[0]['epochs'],7);self.assertEqual(calls[0]['seed'],103301);self.assertEqual(a[0]['optimizer_steps'],21);self.assertEqual(a[0]['parameter_n'],7)
    def test_original_parser_prefix_real_source(self):
        path=Path(__file__).resolve().parent.parent/'role_api_deps/scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py'
        if not path.is_file():self.skipTest('local original-source mirror unavailable; server runtime preflight parses active source')
        module=ast.parse(path.read_text(encoding='utf-8')); node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main'); prefix=[]
        for s in node.body:
            if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in s.targets): break
            prefix.append(s)
        env=dict(argparse=argparse,Path=Path,MODEL_VARIANTS=('RGCondAttn',))
        exec(compile(ast.Module(body=prefix,type_ignores=[]),'original parser','exec'),env)
        a=env['parser'].parse_args(['--variants','RGCondAttn','--seeds','0,1,2','--keep-weight','1','--fix-weight','1','--keep-consistency','0'])
        self.assertEqual(a.batch_size,16); self.assertEqual(a.max_epochs,30);self.assertEqual(a.split_seed,20260901)

class IntegrationTests(unittest.TestCase):
    def test_svm_module_bypasses_loaded_RF_only_namesake(self):
        family=rt.FAMILY if rt.FAMILY.is_dir() else Path(__file__).resolve().parent.parent/'care-p31-classifier-family-stage2-v1'
        if not (family/'nested_stage1.py').is_file(): self.fail('family NestedStage1 source unavailable')
        names=('common','stage1_core','nested_stage1','_rct_svm_nested_stage1')
        saved={n:sys.modules.get(n) for n in names}; old_family=rt.FAMILY
        try:
            rt.FAMILY=family
            class RFOnly:
                def __init__(self,mode,*a):
                    if mode not in ('RF','MEAN'): raise ValueError(mode)
            wrong=SimpleNamespace(NestedStage1=RFOnly)
            sys.modules['nested_stage1']=wrong; sys.modules.pop('_rct_svm_nested_stage1',None)
            common=rt.family_module('common'); old_cache=common.SOURCE_P31
            path=fixture_root()/'p31.npz'
            np.savez(path,sha256=np.asarray([f'{i:064x}' for i in range(799)]),role_prevalence=np.ones((799,31))/31)
            common.SOURCE_P31=path
            try: actual=rt.svm_nested_module()
            finally: common.SOURCE_P31=old_cache
            instance=actual.NestedStage1('SVM',path.with_suffix('.json'),4)
            self.assertEqual(instance.arm,'SVM'); self.assertEqual(len(instance.ids),799)
            self.assertEqual(Path(actual.__file__).resolve(),(family/'nested_stage1.py').resolve())
            self.assertIs(sys.modules['nested_stage1'],wrong)
        finally:
            rt.FAMILY=old_family
            for n,value in saved.items():
                if value is None: sys.modules.pop(n,None)
                else: sys.modules[n]=value
    def test_full_799_fold_training_and_summary(self):
        # Fake neural backend validates protocol plumbing only, not GPU numerics.
        from summarize import main as summary_main
        import summarize as sm
        path=fixture_root(); prior=(pc.OUTPUT,ta.OUTPUT,sm.OUTPUT); pc.OUTPUT=ta.OUTPUT=sm.OUTPUT=path
        try:
            n=799; ids=np.asarray([f'{i:064x}' for i in range(n)]); folds=np.repeat(np.arange(1,6),[160,160,160,160,159]); y=np.arange(n)%2
            rng=np.random.RandomState(4); member=rng.uniform(.1,.9,(n,4)); base=member[:,3]; route=np.zeros(n,bool)
            for f in range(1,6): ix=np.flatnonzero(folds==f); route[ix]=top25(opposition(member[ix]),ids[ix])
            d=dict(ids=ids,y=y,folds=folds,base=base,route=route,expert=np.full(n,.55),host_count=rng.randint(1,100,n))
            pc.save_npz(path/'source_index.npz',**d)
            for f in range(1,6):
                ix=np.flatnonzero(folds!=f); groups=np.arange(len(ix))%4
                w,r,_=make_weights(y[ix],ids[ix],member[ix],d['host_count'][ix],pc.protocol()['shuffle_seed']+f,groups=groups)
                pc.save_npz(path/'routes'/f'f{f}.npz',ids=ids[ix],train_index=ix,member=member[ix],route=r,**w)
            class T:
                float32='float32'
                as_tensor=staticmethod(lambda x,**kw:np.asarray(x))
                optim=SimpleNamespace(Adam=lambda params,**kw:object())
            class Param:
                def numel(self): return 7
            class Model:
                def to(self,d): return self
                def parameters(self): return [Param()]
            seen=[]
            def training(model,opt,gt,base,graph,bags,labels,weights,correct,epochs,args,seed):
                self.assertIn(len(labels),(639,640)); self.assertEqual(epochs,7); seen.append((len(labels),epochs,float(weights.mean())))
                return .2
            def original(gt,gv,s7t,s7v,bt,bv,baset,basev,labels,variant,seeds,args,device):
                return np.full(len(gv),.55),np.ones((len(gv),1)),[dict(seed=int(s),selected_epochs=7,parameter_n=7,selection_loss=.3) for s in seeds]
            v=SimpleNamespace(torch=T,fit_predict_variant=original,v3=SimpleNamespace(fit_scaled_dualproto_bags=lambda a,b,y,args,seed:(a,b,{})),
                fit_role_graph_scaler=lambda a:None,transform_role_graphs=lambda a,s:a,graph_tensors=lambda a,d:a,seed_everything=lambda seed:None,
                RoleGraphConditionedEvidence=lambda *a:Model(),train_epochs=training,
                predict_model=lambda model,gv,base,graph,bags,dev:(np.full(len(gv),.55),np.ones((len(gv),1))))
            args=SimpleNamespace(learning_rate=.0008,weight_decay=.0005,batch_size=16); device=SimpleNamespace(type='cpu')
            for arm in pc.ARMS:
                state=TrainingState(v,arm,d); p=np.full(n,np.nan)
                for f in range(1,6):
                    tr=np.flatnonzero(folds!=f); te=np.flatnonzero(folds==f)
                    with contextlib.redirect_stdout(io.StringIO()):
                        p[te],_,_=state.fit_predict(np.zeros((len(tr),2),np.float32),np.zeros((len(te),2),np.float32),
                            np.zeros((len(tr),2)),np.zeros((len(te),2)),[np.zeros((1,169),np.float32)]*len(tr),[np.zeros((1,169),np.float32)]*len(te),
                            base[tr],base[te],y[tr],'RGCondAttn',[100000*f+s for s in (0,1,2)],args,device)
                self.assertEqual(state.fold,5); pc.save_npz(path/'predictions'/f'{arm}.npz',ids=ids,expert=p)
            self.assertEqual(len(seen),45)
            pc.dump(path/'baseline_integrity_check.json',dict(status='PASS'))
            pc.dump(path/'baseline_replay_validation.json',dict(status='PASS'))
            # Use full frozen 4000-replicate statistical code on this fixture.
            with contextlib.redirect_stdout(io.StringIO()): summary_main()
            dec=pc.read_json(path/'final_decision.json'); self.assertEqual(dec['status'],'ROUTE_CONDITIONED_TRAINING_NOT_SUPPORTED')
            self.assertFalse(dec['automatic_future_run']); self.assertEqual(len(pc.read_csv(path/'source_fold_results.csv')),20)
            self.assertEqual(len(pc.read_csv(path/'paired_comparisons.csv')),9); self.assertEqual(len(pc.read_csv(path/'source_predictions.csv')),799)
        finally: pc.OUTPUT,ta.OUTPUT,sm.OUTPUT=prior
    def test_source_feature_filter_restores_reader(self):
        import csv
        path=fixture_root()/'features.csv'
        with path.open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=['sha256','metadata_year']); w.writeheader(); w.writerows([dict(sha256='a',metadata_year=2017),dict(sha256='b',metadata_year=2021)])
        namespace={'read_csv':pc.read_csv}
        exec('def loader(args):\n    return read_csv(args.features)\n',namespace)
        reader=namespace['read_csv']; wrapped=wrap_source_loader(namespace['loader'],{'a'})
        rows=wrapped(SimpleNamespace(features=path)); self.assertEqual(len(rows),1); self.assertEqual(rows[0]['metadata_year'],'2017'); self.assertIs(namespace['read_csv'],reader)
    def test_cache_mismatch_never_rebuilds(self):
        path=fixture_root()/'cache.npz'; np.savez(path,sha256=np.asarray(['a']),host_values=np.zeros((1,169)),host_offsets=np.array([0,1]))
        with self.assertRaises(RuntimeError): read_only_bags([dict(sha256='b')],None,SimpleNamespace(region_cache=path,rebuild_region_cache=False))
        self.assertFalse(path.with_name('cache_source_only.npz').exists())
    def test_safe_write_rejects_formal_file(self):
        with self.assertRaises(RuntimeError): pc.safe(pc.PROJECT/'artifacts/forbidden.json')

if __name__=='__main__':
    unittest.main(verbosity=2)
