#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from haig_model import read_csv,load_graphs,write_csv
def main():
    p=argparse.ArgumentParser();p.add_argument("--lineage",type=Path,required=True);p.add_argument("--graphs",type=Path,required=True);p.add_argument("--limit",type=int,required=True);p.add_argument("--phase",required=True);args=p.parse_args();rows=read_csv(args.lineage)[:args.limit];graphs,fail=load_graphs(rows,args.graphs)
    stats=[]
    for g in graphs:stats.append({"sha256":g["sha256"],"status":g.get("status"),**g.get("stats",{})})
    ok=[g for g in graphs if g.get("status")=="OK"];total_nodes=sum(g["stats"]["num_api_nodes"] for g in ok);same=sum(g["stats"]["num_same_host_edges"] for g in ok);call=sum(g["stats"]["num_host_call_edges"] for g in ok);caps=sum(g["stats"].get("same_edge_cap_hit",False) or g["stats"].get("call_edge_cap_hit",False) for g in ok)
    decision="FEASIBLE" if len(ok)>=max(1,int(.8*len(graphs))) and total_nodes>0 and same>0 and caps==0 else "BLOCKED"
    out={"phase":args.phase,"decision":decision,"rows":len(graphs),"ok":len(ok),"failed":len(graphs)-len(ok),"api_nodes":total_nodes,"same_host_edges":same,"host_call_edges":call,"edge_cap_hits":caps,"H3_cross_host_available":call>0}
    root=Path("artifacts/care-p31rf-haig-source-v1");(root/f"{args.phase}_audit.json").write_text(json.dumps(out,indent=2)+"\n");write_csv(root/f"{args.phase}_stats.csv",stats)
    print(f"HAIG_{args.phase.upper()}_AUDIT {decision} rows={len(graphs)} ok={len(ok)} nodes={total_nodes} same={same} call={call} caps={caps}")
    if decision!="FEASIBLE":raise RuntimeError("BLOCKED_BY_CONTRACT: HAIG extraction audit failed")
if __name__=="__main__":main()
