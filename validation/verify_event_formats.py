"""Audit event schedules, original baselines, full states and all output tables."""
import argparse
from collections import Counter
from dataclasses import replace
import csv
import gzip
from itertools import combinations
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.event_format_experiment import *
from poc.ab_poc.bridge_experiment import _sha256, _write_json
from poc.ab_poc.domain import Event, PairState, OnlineKnown, RelationshipThresholds
from poc.ab_poc.rule_providers import RuleRelationshipEvaluator

def strings(r): return {k:'' if v is None else str(v) for k,v in r.items()}

def clean_baseline(raw):
    r=compact_result(raw)
    for c in ('A','B'): r['runs'][c]['snapshots']=raw['runs'][c]['snapshots']
    return r

def verify_run(raw,case,design=DEFAULT_DESIGN):
    horizon=design.settings['turns']
    seed=raw['root_seed']; active=set(design.positions(case)); plan=contact_plan(seed,case,design.positions(case))
    ids=[p['id'] for p in raw['participants']]
    planned=ids[:]; random.Random(seed).shuffle(planned); planned*=horizon//4
    for run in raw['runs'].values():
        known={canonical_pair(*p) for p in run['condition']['online_known_pairs']}
        states={p:PairState(*p,online_known=OnlineKnown.DIRECT if p in known else OnlineKnown.NONE) for p in PAIRS}
        history={p:[] for p in PAIRS}; counts=Counter()
        evaluator=RuleRelationshipEvaluator(RelationshipThresholds(**run['rules']['relationship']))
        snapshots=run['snapshots']
        assert [(s['turn'],s['phase']) for s in snapshots]==[(0,'start')]+[(i,'turn_end') for i in range(1,horizon+1)]+[(horizon,'end')]
        def check(s):
            assert {canonical_pair(p['left_id'],p['right_id']):p for p in s['pairs']}=={p:to_jsonable(v) for p,v in states.items()}
        check(snapshots[0]); forced=Counter()
        assert run['turn_order']==[t['actor'] for t in run['turns']]
        assert len(run['event_log'])==(0 if case=='free' else horizon)
        for i,t in enumerate(run['turns'],1):
            assert t['turn']==i and t['actor'] in ids and t['target'] in ids and t['actor']!=t['target']
            pair=canonical_pair(t['actor'],t['target'])
            if case!='free':
                e=run['event_log'][i-1]
                assert e['turn']==i and e['active']==(i in active)
                assert e['planned_actor']==planned[i-1] and e['actual_actor']==t['actor'] and e['actual_target']==t['target']
                assert e['shared_override']==(case.startswith('topic_') and i in active)
                if i in plan:
                    assert (t['actor'],t['target'])==plan[i]
                    assert e['forced_pair']==list(pair); forced[pair]+=1
                    if case.startswith('task_'): assert e['task_role'] in t['approach']
                if case.startswith('host_') and i in active:
                    minimum=min(counts[p] for p in PAIRS)
                    tied=sorted(p for p in PAIRS if counts[p]==minimum)
                    introduced=rng_for(seed,f'host-tie:{i}').choice(tied)
                    actor=introduced[rng_for(seed,f'host-direction:{i}').randrange(2)]
                    assert e['introduced_pair']==list(introduced) and e['introduced_prior_count']==minimum
                    assert t['actor']==actor and e['host_bonus']==design.settings['host_selection_bonus']
            if i not in active or case.startswith('topic_'): assert t['actor']==planned[i-1]
            history[pair].append(Event(**t)); counts[pair]+=1
            states[pair]=replace(evaluator.evaluate(history[pair]).pair,online_known=states[pair].online_known)
            check(snapshots[i])
        check(snapshots[-1])
        if plan: assert forced==Counter({p:len(active)//6 for p in PAIRS})
    if plan:
        for turn in active:
            for c in ('A','B'): assert (raw['runs'][c]['turns'][turn-1]['actor'],raw['runs'][c]['turns'][turn-1]['target'])==plan[turn]

def audit(output,follow=False,compare_previous=True,write_report=True,previous_root=None):
    out=Path(output); deadline=time.monotonic()+1800
    def wait_file(p):
        while follow and not p.exists() and time.monotonic()<deadline: time.sleep(2)
        assert p.exists(),str(p)
    wait_file(out/'design.json')
    design=json.loads((out/'design.json').read_text(encoding='utf-8'))
    profile=EventDesign(design['settings']); horizon=profile.settings['turns']
    start,end=design['seed_start'],design['seed_end']
    rows=[]; previous_matches=pair_rows=task_matches=0
    for p in PAIRS:
        key='-'.join(p); folder=out/key
        wait_file(folder/'complete.json')
        done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        for name,h in done['output_sha256'].items(): assert _sha256(folder/name)==h
        previous=None
        if compare_previous:
            previous_dir=Path(previous_root) if previous_root is not None else ROOT/'results/pair-identity-101-1100'
            previous=gzip.open(previous_dir/key/'traces.jsonl.gz','rt',encoding='utf-8')
            for _ in range((start-101)*4): next(previous)
        try:
            with gzip.open(folder/'traces.jsonl.gz','rt',encoding='utf-8') as f, \
                 (folder/'pairs.csv').open(encoding='utf-8-sig',newline='') as pf, \
                 (folder/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as sf:
                pr,sr=csv.DictReader(pf),csv.DictReader(sf)
                for seed in range(start,end+1):
                    results={}
                    for case in profile.cases:
                        rec=json.loads(next(f)); raw=rec['result']
                        assert rec['case_id']==case and raw['root_seed']==seed
                        results[case]=raw
                        cfg=pair_config(p,horizon)
                        configured=to_jsonable(cfg.participants)
                        assert raw['participants']==[{k:p[k] for k in ('id','name','interests','stance')} for p in configured]
                        assert design['configs'][key]==to_jsonable(cfg)
                        for i,c in enumerate(('A','B')):
                            assert raw['runs'][c]['condition']==to_jsonable(cfg.conditions[i])
                            assert raw['runs'][c]['rules']==to_jsonable(cfg.rules)
                        verify_run(raw,case,profile)
                        if previous and case=='free':
                            old_runs=[json.loads(next(previous)) for _ in range(4)]
                            old=old_runs[(12,24,36,48).index(horizon)]
                            assert clean_baseline(raw)==old; previous_matches+=1
                        pairs,row=extract_event(raw,case,profile); rows.append(row)
                        assert next(sr)==strings(row)
                        for pair in pairs: assert next(pr)==strings(pair); pair_rows+=1
                    for strength in profile.strengths if 'task_blocks' not in profile.settings else ():
                        for c in ('A','B'):
                            a=results['shuffle_'+strength]['runs'][c]; b=results['task_'+strength]['runs'][c]
                            assert [{k:v for k,v in t.items() if k!='approach'} for t in a['turns']]==[{k:v for k,v in t.items() if k!='approach'} for t in b['turns']]
                            assert a['snapshots']==b['snapshots']
                            task_matches+=1
                    if (seed-start+1)%250==0: print(f'audit {key}: {seed-start+1}/{end-start+1}',flush=True)
                assert next(f,None) is None and next(pr,None) is None and next(sr,None) is None
        finally:
            if previous: previous.close()
    wait_file(out/'manifest.json')
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['source_sha256'].items(): assert _sha256(ROOT/name)==h
    for name,h in manifest['output_sha256'].items(): assert _sha256(out/name)==h
    summary,pooled=summarize_event(rows,profile)
    for name,data in (('summary',summary),('pooled',pooled)):
        assert data==json.loads((out/(name+'.json')).read_text(encoding='utf-8'))
        with (out/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f: assert list(csv.DictReader(f))==[strings(r) for r in data]
    assert pair_rows==manifest['pair_rows']
    report=dict(status='pass',condition_runs=manifest['condition_runs'],pair_rows=pair_rows,
        previous_free_AB_runs_matched=previous_matches,task_shuffle_condition_matches=task_matches,
        all_states_rebuilt=True,schedules_host_history_and_logs_verified=True,
        all_summary_and_pooled_metrics_recomputed=True,source_output_hashes_match=True)
    if write_report: _write_json(ROOT/('validation/event-format-48-audit.json' if horizon==48 else 'validation/event-format-audit.json'),report)
    print(json.dumps(report),flush=True); return report
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',nargs='?',default=str(ROOT/'results/event-formats-101-1100'))
    p.add_argument('--follow',action='store_true'); a=p.parse_args(); audit(a.output,follow=a.follow)
