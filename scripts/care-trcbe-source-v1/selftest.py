#!/usr/bin/env python3
from __future__ import annotations
from collections import Counter,deque
from common import canonical,stable_id
from taxonomy import classify
from run_coverage import classify_host,multi_source_distances,shuffle_root_types,shuffle_root_host,type_closure

def main():
    sig="Lx/A;->f ( I ) V";assert canonical(sig)=="lx/a;->f(i)v";assert stable_id(sig)==stable_id("lx/a;->f(i)v")
    classes={canonical("Lx/Base;"):{"super":canonical("Landroid/app/Activity;"),"interfaces":[]},canonical("Lx/Child;"):{"super":canonical("Lx/Base;"),"interfaces":[canonical("Landroid/view/View$OnClickListener;")]}}
    closure=type_closure(canonical("Lx/Child;"),classes,{})
    assert canonical("Landroid/app/Activity;") in closure and canonical("Landroid/view/View$OnClickListener;") in closure
    user=classify(closure,"onClick","(Landroid/view/View;)V")
    assert user and user["category"].startswith("USER")
    auto=classify({canonical("Landroid/content/BroadcastReceiver;")},"onReceive","(Landroid/content/Context;Landroid/content/Intent;)V")
    assert auto and auto["category"].startswith("AUTONOMOUS")
    assert classify({canonical("Lx/Fake;")},"onClick","(Landroid/view/View;)V") is None
    neutral=classify({canonical("Landroid/app/Activity;")},"onCreate","(Landroid/os/Bundle;)V");assert neutral and neutral["category"]=="NEUTRAL_OR_AMBIGUOUS"
    adj={0:[1],1:[2],3:[1],4:[5]};u=multi_source_distances(adj,{0});a=multi_source_distances(adj,{3});n=multi_source_distances(adj,{4});assert classify_host(2,u,a,n)[0]=="BOTH"
    assert classify_host(2,u,{},{} )[0]=="USER_ONLY";assert classify_host(2,{},a,{})[0]=="AUTONOMOUS_ONLY";assert classify_host(9,u,a,{})[0]=="UNREACHABLE"
    assert classify_host(5,{}, {},n)[0]=="NEUTRAL_ONLY"
    # Per-APK graph objects are separate, so node index 2 in another graph is never visited.
    assert 2 not in multi_source_distances({0:[1]},{0})
    roots=[("u",1,"USER"),("a",2,"AUTO")];shuffled=shuffle_root_types(roots,7);assert Counter(x[2] for x in shuffled)==Counter(x[2] for x in roots)
    pairs=[(1,5),(2,6)];shuffled_pairs=shuffle_root_host(pairs,7);assert Counter(x for x,_ in shuffled_pairs)==Counter(x for x,_ in pairs) and Counter(y for _,y in shuffled_pairs)==Counter(y for _,y in pairs)
    assert stable_id("Lx/H;->host()V")==stable_id(canonical("Lx/H;->host()V"))
    print("TRCBE_SELFTEST_PASS tests=signature,superclass,interface,user,autonomous,false_name,neutral,multisource,user_only,auto_only,both,neutral_only,unreachable,apk_isolation,host_alignment,root_type_shuffle,root_host_shuffle")
if __name__=="__main__":main()

