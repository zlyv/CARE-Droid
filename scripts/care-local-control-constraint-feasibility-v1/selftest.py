#!/usr/bin/env python3
"""Proof soundness fixtures and exhaustive finite-path oracle comparisons."""
from __future__ import annotations
import itertools,random,struct,unittest
from types import SimpleNamespace
from control_core import *
from dex_adapter import decode_method,alignment,framework_permission_target

def row(off,name,**kw):return dict(offset=off,name=name,length=2,registers=[],**kw)
def paths(cfg,start):
    if start==EXIT:return [[EXIT]]
    return [[start]+p for nxt in cfg['succ'][start] for p in paths(cfg,nxt)]
class ProofTests(unittest.TestCase):
    def simple(self):return [row(0,'if-eqz',target=6),row(2,'invoke-static',reference='x'),row(4,'return-void'),row(6,'return-void')]
    def test_diamond(self):
        c=build_cfg(self.simple());p=analyze_event(c,2)
        self.assertEqual(p['evidence_tier'],'VERIFIED_CONTROL');self.assertFalse(p['controls'][0]['bypass_path_possible'])
    def test_both_branches_reach(self):
        c=build_cfg([row(0,'if-eqz',target=4),row(2,'goto',target=6),row(4,'nop'),row(6,'invoke-static'),row(8,'return-void')])
        self.assertEqual(analyze_event(c,6)['evidence_tier'],'PARTIAL')
    def test_implicit_exception_before_call(self):
        c=build_cfg([row(0,'if-eqz',target=8),row(2,'invoke-static'),row(4,'invoke-static'),row(6,'return-void'),row(8,'return-void')])
        self.assertEqual(analyze_event(c,4)['evidence_tier'],'PARTIAL')
    def test_throw_exit(self):
        c=build_cfg([row(0,'if-eqz',target=6),row(2,'invoke-static'),row(4,'return-void'),row(6,'throw')])
        self.assertEqual(analyze_event(c,2)['evidence_tier'],'VERIFIED_CONTROL')
    def test_loops_unknown(self):
        self.assertEqual(build_cfg([row(0,'if-eqz',target=0),row(2,'return-void')])['reason'],'REACHABLE_LOOP_OR_CYCLE')
    def test_try_unknown(self):self.assertEqual(build_cfg(self.simple(),tries_size=1)['status'],'UNKNOWN')
    def test_bad_jump(self):self.assertEqual(build_cfg([row(0,'goto',target=3)])['status'],'UNKNOWN')
    def test_unsupported(self):self.assertEqual(build_cfg([row(0,'unused'),row(2,'return-void')])['status'],'UNKNOWN')
    def test_cap(self):self.assertEqual(build_cfg(self.simple(),max_nodes=2)['reason'],'INSTRUCTION_CAP')
    def test_switch(self):
        c=build_cfg([row(0,'packed-switch',switch_targets=[6,8],switch_keys=[1,2]),row(2,'invoke-static'),row(4,'return-void'),row(6,'return-void'),row(8,'return-void')])
        self.assertEqual(analyze_event(c,2)['evidence_tier'],'VERIFIED_CONTROL')
    def permission(self,op='if-nez',dispatch=True):
        r=[row(0,'const-string',string='android.permission.READ_CONTACTS'),row(2,'invoke-super',reference='landroid/content/context;->checkselfpermission(ljava/lang/string;)i',framework_target_when_receiver_this=dispatch),
           row(4,'move-result'),row(6,op,target=12),row(8,'invoke-static'),row(10,'return-void'),row(12,'return-void')]
        r[0]['registers']=[1];r[2]['registers']=[2];r[1]['registers']=[0,1];r[3]['registers']=[2]
        c=build_cfg(r);c['initial_symbols']={0:('THIS',)};return c
    def test_permission_granted(self):
        p=analyze_event(self.permission(),8);self.assertEqual(p['evidence_tier'],'VERIFIED_PERMISSION_GUARD');self.assertTrue(p['permissions'][0]['allows_permission_granted'])
        self.assertIn(0,p['permissions'][0]['def_use_instruction_offsets'])
    def test_permission_denied(self):
        p=analyze_event(self.permission('if-eqz'),8);self.assertTrue(p['permissions'][0]['allows_permission_denied'])
    def test_permission_dispatch_unknown(self):
        p=analyze_event(self.permission(dispatch=False),8);self.assertEqual(p['evidence_tier'],'VERIFIED_CONTROL');self.assertEqual(p['permissions'][0]['status'],'PARTIAL')
    def test_redefined_result(self):
        c=self.permission();rr=list(c['rows'].values());rr.insert(3,row(5,'const/4',literal=0));rr[3]['registers']=[2]
        # Use contiguous valid offsets after insertion.
        for i,r in enumerate(rr):r['offset']=2*i
        rr[4]['target']=14
        c=build_cfg(rr);c['initial_symbols']={0:('THIS',)}
        self.assertNotEqual(analyze_event(c,10)['evidence_tier'],'VERIFIED_PERMISSION_GUARD')
    def test_permission_join_kills_conflict(self):
        r=[row(0,'const-string',string='android.permission.READ_CONTACTS'),row(2,'invoke-super',reference='landroid/content/context;->checkselfpermission(ljava/lang/string;)i',framework_target_when_receiver_this=True),
           row(4,'move-result'),row(6,'if-eqz',target=12),row(8,'const/4',literal=0),row(10,'goto',target=14),row(12,'nop'),
           row(14,'if-nez',target=20),row(16,'invoke-static'),row(18,'return-void'),row(20,'return-void')]
        r[0]['registers']=[1];r[1]['registers']=[0,1];r[2]['registers']=[2];r[3]['registers']=[3];r[4]['registers']=[2];r[7]['registers']=[2]
        c=build_cfg(r);c['initial_symbols']={0:('THIS',)};p=analyze_event(c,16)
        self.assertNotEqual(p['evidence_tier'],'VERIFIED_PERMISSION_GUARD')
    def test_permission_bypass(self):
        r=[row(0,'if-eqz',target=6),row(2,'goto',target=18),row(4,'return-void'),row(6,'const-string',string='android.permission.READ_CONTACTS'),
           row(8,'invoke-super',reference='landroid/content/context;->checkselfpermission(ljava/lang/string;)i',framework_target_when_receiver_this=True),
           row(10,'move-result'),row(12,'if-nez',target=16),row(14,'goto',target=18),row(16,'return-void'),row(18,'invoke-static'),row(20,'return-void')]
        r[3]['registers']=[1];r[4]['registers']=[0,1];r[5]['registers']=[2];r[6]['registers']=[2]
        c=build_cfg(r);c['initial_symbols']={0:('THIS',)};p=analyze_event(c,18)
        self.assertNotEqual(p['evidence_tier'],'VERIFIED_PERMISSION_GUARD')
    def test_finite_path_oracle(self):
        rng=random.Random(177)
        for trial in range(100):
            rr=[]
            for i in range(7):
                if i==6:rr.append(row(2*i,'return-void'))
                elif rng.random()<.5:rr.append(row(2*i,'if-eqz',target=2*rng.randrange(i+1,7)))
                else:rr.append(row(2*i,'nop'))
            cfg=build_cfg(rr)
            for event in cfg['reachable']-{EXIT}:
                actual={(p['branch_offset'],p['allowing_successor_offset']) for p in controlling_branches(cfg,event)};expected=set()
                for branch,labels in cfg['branch'].items():
                    if branch not in cfg['reachable']:continue
                    ss=set(labels.values())
                    if len(ss)<2:continue
                    for s in ss:
                        if all(event in p for p in paths(cfg,s)) and any(all(event not in p for p in paths(cfg,b)) for b in ss-{s}) and not all(event in p for p in paths(cfg,branch)):
                            expected.add((branch,s))
                self.assertEqual(actual,expected)
    def test_alignment_denominators(self):
        f=dict(pairs={'11:22':[3]},nodes={'11':{'endpoint':False},'22':{'endpoint':True}},host_ids=['11'],summary={})
        e=dict(host_node_id=11,endpoint_node_id=22,role_ids=[3]);self.assertEqual(alignment(e,f,{'11':1}),'EXACT_FROZEN_HOST_ENDPOINT_EDGE')
        e['endpoint_node_id']=33;self.assertIn('ENDPOINT_NOT_SELECTED',alignment(e,f,{'11':1}))
        e['endpoint_node_id']=22;self.assertEqual(alignment(e,f,{'11':2}),'AMBIGUOUS_CANONICAL_HOST_ID')
    def test_exact_permission_resolution(self):
        info={'lapp/c;':dict(final=True,super='landroid/app/activity;',methods=set())}
        ref='landroid/content/context;->checkselfpermission(ljava/lang/string;)i'
        self.assertTrue(framework_permission_target('lapp/c;',ref,'invoke-virtual',info));info['lapp/c;']['final']=False
        self.assertFalse(framework_permission_target('lapp/c;',ref,'invoke-virtual',info));self.assertTrue(framework_permission_target('lapp/c;',ref,'invoke-super',info))
    def test_real_androguard_branch_units(self):
        from androguard.core.dex import Instruction21t,Instruction10x
        class Packers:
            def __getitem__(self,key):return struct.Struct('<'+key)
        cm=SimpleNamespace(packer=Packers());obj=Instruction21t(cm,struct.pack('<BBh',0x38,1,3));ret=Instruction10x(cm,b'\x0e\x00')
        class Method:
            def get_code(self):return SimpleNamespace(get_tries_size=lambda:0,get_registers_size=lambda:2,get_ins_size=lambda:0)
            def get_instructions_idx(self):return [(0,obj),(4,ret),(6,ret)]
        rr,_=decode_method(Method());self.assertEqual(rr[0]['target'],6);self.assertEqual(rr[0]['registers'],[1]);self.assertEqual(build_cfg(rr)['status'],'PARSED')
    def test_real_androguard_permission_chain(self):
        from androguard.core.dex import Instruction21c,Instruction35c,Instruction11x,Instruction21t,Instruction10x
        class Packers:
            def __getitem__(self,key):return struct.Struct('<'+key)
        class Manager:
            packer=Packers()
            def get_string(self,index):return 'android.permission.READ_CONTACTS'
            def get_method_ref(self,index):
                if index==0:return SimpleNamespace(get_class_name=lambda:'Landroid/content/Context;',get_name=lambda:'checkSelfPermission',get_descriptor=lambda:'(Ljava/lang/String;)I')
                return SimpleNamespace(get_class_name=lambda:'Ljava/net/Socket;',get_name=lambda:'connect',get_descriptor=lambda:'()V')
        cm=Manager();objs=[(0,Instruction21c(cm,struct.pack('<BBH',0x1a,1,0))),
             (4,Instruction35c(cm,struct.pack('<HHH',0x206f,0,0x10))),
             (10,Instruction11x(cm,b'\x0a\x02')),
             (12,Instruction21t(cm,struct.pack('<BBh',0x39,2,6))),
             (16,Instruction35c(cm,struct.pack('<HHH',0x71,1,0))),
             (22,Instruction10x(cm,b'\x0e\x00')),(24,Instruction10x(cm,b'\x0e\x00'))]
        class Method:
            def get_code(self):return SimpleNamespace(get_tries_size=lambda:0,get_registers_size=lambda:3,get_ins_size=lambda:1)
            def get_instructions_idx(self):return objs
        rr,_=decode_method(Method());self.assertEqual(rr[1]['registers'],[0,1]);self.assertEqual(rr[3]['target'],24)
        rr[1]['framework_target_when_receiver_this']=True;c=build_cfg(rr);c['initial_symbols']={0:('THIS',)}
        p=analyze_event(c,16);self.assertEqual(p['evidence_tier'],'VERIFIED_PERMISSION_GUARD')
        self.assertTrue(p['permissions'][0]['allows_permission_granted'])
    def test_real_switch_payload_units(self):
        from androguard.core.dex import Instruction31t,Instruction10x,PackedSwitch
        class Packers:
            def __getitem__(self,key):return struct.Struct('<'+key)
        cm=SimpleNamespace(packer=Packers());obj=Instruction31t(cm,struct.pack('<BBi',0x2b,0,6))
        payload=PackedSwitch(cm,struct.pack('<HHi',0x0100,1,7)+struct.pack('<i',4));ret=Instruction10x(cm,b'\x0e\x00')
        class Method:
            def get_code(self):return SimpleNamespace(get_tries_size=lambda:0,get_registers_size=lambda:1,get_ins_size=lambda:0)
            def get_instructions_idx(self):return [(0,obj),(6,ret),(8,ret),(12,payload)]
        rr,_=decode_method(Method());self.assertEqual(rr[0]['switch_targets'],[8]);self.assertEqual(rr[0]['switch_keys'],[7]);self.assertTrue(rr[-1]['payload'])
    def test_scanner_with_typed_dex_fixture(self):
        from unittest.mock import patch
        from pathlib import Path
        from common import stable_id
        from scan_one import scan
        from androguard.core.dex import Instruction21c,Instruction35c,Instruction11x,Instruction21t,Instruction10x
        class Packers:
            def __getitem__(self,key):return struct.Struct('<'+key)
        class Manager:
            packer=Packers()
            def get_string(self,index):return 'android.permission.READ_CONTACTS'
            def get_method_ref(self,index):
                if index==0:return SimpleNamespace(get_class_name=lambda:'Landroid/content/Context;',get_name=lambda:'checkSelfPermission',get_descriptor=lambda:'(Ljava/lang/String;)I')
                return SimpleNamespace(get_class_name=lambda:'Ljava/net/Socket;',get_name=lambda:'connect',get_descriptor=lambda:'()V')
        cm=Manager();objs=[(0,Instruction21c(cm,struct.pack('<BBH',0x1a,1,0))),
             (4,Instruction35c(cm,struct.pack('<HHH',0x206f,0,0x10))),(10,Instruction11x(cm,b'\x0a\x02')),
             (12,Instruction21t(cm,struct.pack('<BBh',0x39,2,6))),(16,Instruction35c(cm,struct.pack('<HHH',0x71,1,0))),
             (22,Instruction10x(cm,b'\x0e\x00')),(24,Instruction10x(cm,b'\x0e\x00'))]
        method=SimpleNamespace(get_name=lambda:'test',get_descriptor=lambda:'(Ljava/lang/String;I)V',get_access_flags=lambda:0,
           get_code=lambda:SimpleNamespace(get_tries_size=lambda:0,get_registers_size=lambda:3,get_ins_size=lambda:3),get_instructions_idx=lambda:objs)
        cls=SimpleNamespace(get_name=lambda:'Lapp/C;',get_superclassname=lambda:'Landroid/content/Context;',get_access_flags=lambda:0x10,get_methods=lambda:[method])
        apk=SimpleNamespace(get_dex_names=lambda:['classes.dex'],get_file=lambda name:b'SYNTHETIC_DEX_FIXTURE')
        dex=SimpleNamespace(get_classes=lambda:[cls]);h=stable_id('Lapp/C;->test(Ljava/lang/String;I)V');t=stable_id('Ljava/net/Socket;->connect()V')
        frozen=dict(pairs={str(h)+':'+str(t):[0]},host_ids=[str(h)],nodes={str(h):dict(endpoint=False),str(t):dict(endpoint=True)},summary={})
        mapping=dict(roles=['network'],rules=[dict(role='network',class_prefixes=['Ljava/net/Socket;'],method_names=['connect'])])
        with patch('androguard.core.apk.APK',return_value=apk),patch('androguard.core.dex.DEX',return_value=dex):
            r=scan(Path('synthetic_fixture.apk'),'1'*64,mapping,frozen,'2'*64)
        self.assertEqual(r['frozen_pairs_recovered'],1);self.assertEqual(r['events'][0]['evidence_tier'],'VERIFIED_PERMISSION_GUARD')
        self.assertEqual(r['events'][0]['invoke_offset_bytes'],16);self.assertEqual(r['events'][0]['class_descriptor'],'Lapp/C;')
        self.assertEqual(r['events'][0]['permissions'][0]['def_use_instruction_offsets'],[0,4,10,12])
if __name__=='__main__':unittest.main(verbosity=2)
