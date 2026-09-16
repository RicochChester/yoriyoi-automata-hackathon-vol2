"""Mechanism interventions across known-pair identities; original model reused."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, NormalDist
from .bridge_experiment import PROJECT_ROOT, _sha256, _write_csv, _write_json
from .duration_experiment import HORIZONS, extract
from .domain import canonical_pair
from .engine import run_ab
from .mechanism_experiment import Case, MechanismInitiator, MechanismResponder, compact_result, estimate, VERSION as PROVIDER_VERSION
from .metrics import compute_metrics
from .pair_identity_experiment import PAIRS, pair_config
from .rule_providers import RuleParticipantProvider

VERSION='yoriyoi.pair_mechanism.v1'
CASES=(Case(1,2,True,True),Case(0,2,True,True),Case(1,0,True,True),
       Case(1,2,False,True),Case(1,2,True,False),Case(0,0,False,False),
       Case(1,2,True,True,'A'),Case(1,2,True,True,'B'))
CASE_BY_ID={c.id:c for c in CASES}
BASE=CASES[0].id
METRICS=('bridge_edge_count','bridge_edge_familiar_count','known_conversations',
         'known_share','endpoint_outbound_bridge','bridge_unformed_turns','bridge_all_formed')

def run_intervention(pair,seed,case,horizon=48,donor=None):
    cfg=pair_config(pair,horizon)
    cfg=replace(cfg,rules=replace(cfg.rules,
        selection=replace(cfg.rules.selection,online_known_bonus=float(case.selection)),
        reaction=replace(cfg.rules.reaction,online_known_bonus=case.reaction)))
    schedule=()
    if case.clamp:
        if donor is None: raise ValueError('clamp requires a matching baseline donor')
        if donor['root_seed']!=seed or donor['runs']['B']['condition']['online_known_pairs']!=[list(canonical_pair(*pair))]:
            raise ValueError('donor seed or known pair differs')
        if len(donor['runs'][case.clamp]['turns'])<horizon: raise ValueError('donor is too short')
        schedule=tuple((t['actor'],t['target']) for t in donor['runs'][case.clamp]['turns'][:horizon])
    def factory():
        provider=RuleParticipantProvider(initiator=MechanismInitiator(case.topic,schedule),
                                         responder=MechanismResponder(case.caution))
        if case.clamp or not case.topic or not case.caution:
            provider.name='mechanism-rule-participant'; provider.version=PROVIDER_VERSION
        return provider
    full=run_ab(seed=seed,config=cfg,participant_provider_factory=factory)
    raw=compact_result(full)
    for c in ('A','B'): raw['runs'][c]['snapshots']=full['runs'][c]['snapshots']
    return raw

def checkpoint(raw,h):
    if h not in HORIZONS: raise ValueError('unsupported checkpoint')
    runs={}
    for c,run in raw['runs'].items():
        if len(run['turns'])<h: raise ValueError('checkpoint exceeds trace')
        snap=next(s for s in run['snapshots'] if s['phase']=='turn_end' and s['turn']==h)
        r=dict(run)
        r.update(rules={**run['rules'],'turns':h},turns=run['turns'][:h],
                 turn_order=run['turn_order'][:h],
                 snapshots=[s for s in run['snapshots'] if s['turn']<=h and s['phase']!='end']+
                           [dict(snap,phase='end')],final_relationships=snap['pairs'])
        r['metrics']=compute_metrics(r); runs[c]=r
    return dict(root_seed=raw['root_seed'],participants=raw['participants'],runs=runs)

def onsite_behavior(run,known,bridge=False):
    def keep(left,right):
        p=canonical_pair(left,right)
        return not bridge or (p!=known and bool(set(p)&set(known)))
    return dict(turns=[t for t in run['turns'] if keep(t['actor'],t['target'])],
        pairs=[{k:v for k,v in p.items() if k!='online_known'} for p in run['final_relationships']
               if keep(p['left_id'],p['right_id'])])

def check_invariants(results,pair):
    known=canonical_pair(*pair); base=results[BASE]
    for case in CASES:
        r=results[case.id]
        if not case.clamp:
            if onsite_behavior(r['runs']['A'],known)!=onsite_behavior(base['runs']['A'],known):
                raise ValueError('natural A behavior changed')
        else:
            donor=base['runs'][case.clamp]
            schedule=[(t['actor'],t['target']) for t in donor['turns']]
            for run in r['runs'].values():
                if [(t['actor'],t['target']) for t in run['turns']]!=schedule:
                    raise ValueError('clamp schedule differs')
                if onsite_behavior(run,known,True)!=onsite_behavior(donor,known,True):
                    raise ValueError('clamped bridge behavior differs from donor')
    off=results[CASES[5].id]
    if onsite_behavior(off['runs']['A'],known)!=onsite_behavior(off['runs']['B'],known):
        raise ValueError('all-off A/B behavior differs')

def one_seed(task):
    pair,seed=task
    base=run_intervention(pair,seed,CASES[0])
    results={BASE:base}
    for case in CASES[1:]:
        results[case.id]=run_intervention(pair,seed,case,donor=base)
    check_invariants(results,pair)
    return seed,results

def bounded_results(tasks,workers):
    if workers==1:
        for task in tasks: yield one_seed(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=deque()
        tasks=iter(tasks)
        for _ in range(workers*2):
            task=next(tasks,None)
            if task is not None: pending.append(pool.submit(one_seed,task))
        while pending:
            yield pending.popleft().result()
            task=next(tasks,None)
            if task is not None: pending.append(pool.submit(one_seed,task))

def ci(values,family):
    e=estimate(values); se=e['standard_error']; z=NormalDist().inv_cdf(1-.05/(2*family))
    return {**e,'simultaneous_low':e['mean']-z*se if se is not None else None,
            'simultaneous_high':e['mean']+z*se if se is not None else None}

def summarize(rows):
    keys={(r['known_pair'],r['seed'],r['turns'],r['case_id']):r for r in rows}
    seeds=sorted({r['seed'] for r in rows})
    expected={( '-'.join(p),s,h,c.id) for p in PAIRS for s in seeds for h in HORIZONS for c in CASES}
    if not seeds or len(keys)!=len(rows) or set(keys)!=expected: raise ValueError('incomplete or duplicate design')
    summary=[]; contrasts=[]
    for pair in PAIRS:
        key='-'.join(pair)
        for h in HORIZONS:
            for case in CASES:
                selected=[keys[(key,s,h,case.id)] for s in seeds]
                delta=[r['B_minus_A'] for r in selected]
                row=dict(known_pair=key,turns=h,case_id=case.id,**ci(delta,192),
                         B_greater=sum(d>0 for d in delta),equal=sum(d==0 for d in delta),B_less=sum(d<0 for d in delta))
                for m in METRICS:
                    for c in ('A','B'): row[f'{m}_{c}']=mean(r[f'{m}_{c}'] for r in selected)
                summary.append(row)
                if case.id!=BASE:
                    contrast=dict(known_pair=key,turns=h,case_id=case.id,
                        **ci([keys[(key,s,h,case.id)]['B_minus_A']-keys[(key,s,h,BASE)]['B_minus_A'] for s in seeds],168))
                    contrasts.append(contrast)
    return summary,contrasts

def run_study(output,seed_start=101,seed_end=1100,workers=2):
    if type(workers)!=int or workers not in (1,2): raise ValueError('workers must be 1 or 2')
    if not 101<=seed_start<=seed_end<=1100: raise ValueError('seed range must be within 101..1100')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    sources=sorted((PROJECT_ROOT/'poc').rglob('*.py'))+sorted((PROJECT_ROOT/'poc/config').glob('*.json'))
    sources += [PROJECT_ROOT/'docs/PAIR_MECHANISM_PLAN.md']
    hashes={p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in sources}
    from .domain import to_jsonable
    _write_json(output/'design.json',dict(version=VERSION,seed_start=seed_start,seed_end=seed_end,workers=workers,
        cases={c.id:asdict(c) for c in CASES},horizons=HORIZONS,
        baseline_configs={'-'.join(p):to_jsonable(pair_config(p,48)) for p in PAIRS},source_sha256=hashes))
    started=datetime.now(timezone.utc).isoformat()
    rows=[]
    for pair in PAIRS:
        key='-'.join(pair); folder=output/key; folder.mkdir()
        with gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as rawfile, \
             (folder/'pairs.csv').open('w',encoding='utf-8-sig',newline='') as pf, \
             (folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf:
            pw=sw=None
            for seed,results in bounded_results(((pair,s) for s in range(seed_start,seed_end+1)),workers):
                for case in CASES:
                    raw=results[case.id]
                    rawfile.write(json.dumps(dict(case_id=case.id,result=raw),ensure_ascii=False,separators=(',',':'))+'\n')
                    for h in HORIZONS:
                        pairs,row=extract(checkpoint(raw,h))
                        row.update(known_pair=key,case_id=case.id)
                        for p in pairs: p.update(known_pair=key,case_id=case.id)
                        if pw is None:
                            pw=csv.DictWriter(pf,fieldnames=list(pairs[0])); pw.writeheader()
                            sw=csv.DictWriter(sf,fieldnames=list(row)); sw.writeheader()
                        pw.writerows(pairs); sw.writerow(row); rows.append(row)
                if (seed-seed_start+1)%100==0: print(f'{key}: {seed-seed_start+1}/{seed_end-seed_start+1} seeds, all 8 interventions',flush=True)
        _write_json(folder/'complete.json',dict(seed_start=seed_start,seed_end=seed_end,
            output_sha256={p.name:_sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}))
    summary,contrasts=summarize(rows)
    for name,values in (('summary',summary),('contrasts',contrasts)):
        _write_json(output/(name+'.json'),values); _write_csv(output/(name+'.csv'),values)
    for name,digest in hashes.items():
        if _sha256(PROJECT_ROOT/name)!=digest: raise ValueError('source changed during execution')
    n=seed_end-seed_start+1
    manifest=dict(version=VERSION,status='complete',seed_start=seed_start,seed_end=seed_end,
        started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        actual_condition_runs=96*n,condition_checkpoints=384*n,pair_rows=2304*n,
        source_sha256=hashes,output_sha256={p.relative_to(output).as_posix():_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    _write_json(output/'manifest.json',manifest)
    print(f'Completed: {output.resolve()}',flush=True)
    return manifest

def main():
    parser=argparse.ArgumentParser(description='All-pair mechanism interventions, fixed rule model')
    parser.add_argument('--seed-start',type=int,default=101); parser.add_argument('--seed-end',type=int,default=1100)
    parser.add_argument('--workers',type=int,default=2); parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        run_study(args.output or PROJECT_ROOT/'results'/datetime.now().strftime('pair-mechanism-%Y%m%d-%H%M%S-%f'),
                  args.seed_start,args.seed_end,args.workers)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(1,f'Experiment failed: {exc}. No top-level manifest means incomplete.\n')
if __name__=='__main__': main()
