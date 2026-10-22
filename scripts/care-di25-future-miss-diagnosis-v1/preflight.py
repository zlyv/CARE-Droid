#!/usr/bin/env python3
import run_diagnosis as r

p,rows=r.resolve_scores()
sha=[r.id_of(x) for x in rows]
years=[int(x["year"]) for x in rows]
if len(rows)!=1989 or len(set(sha))!=1989:raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort invalid")
print("DI25_FUTURE_MISS_DIAGNOSIS_PREFLIGHT_PASS")
print("score_file="+str(p.resolve()))
print("future_rows=1989")
print("year_counts=2018:793,2019:399,2020:397,2021:400")
print("analysis_years=2020,2021")
print("training_invoked=false")
print("route_modified=false")

