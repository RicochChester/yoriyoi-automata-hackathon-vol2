"""Reconstruct full trajectories and verify every mechanism intervention."""
import argparse
import csv
from dataclasses import replace
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.pair_mechanism_experiment import *
from poc.ab_poc.bridge_experiment import _sha256, _write_json
from poc.ab_poc.domain import Event, OnlineKnown, PairState, RelationshipThresholds, Stance, to_jsonable
from poc.ab_poc.rule_providers import RuleRelationshipEvaluator, RuleTalkInitiator

def digest(r):
    return hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def strings(row):
    return {k:'' if v is None else str(v) for k,v in row.items()}

def rebuild(raw):
    for run in raw['runs'].values():
        ids=[p['id'] for p in run['participants']]
        known={canonical_pair(*p) for p in run['condition']['online_known_pairs']}
        states={p:PairState(*p,online_known=OnlineKnown.DIRECT if p in known else OnlineKnown.NONE)
                for p in (canonical_pair(*v) for v in combinations(ids,2))}
        history={p:[] for p in states}
        evaluator=RuleRelationshipEvaluator(RelationshipThresholds(**run['rules']['relationship']))
        stances={p['id']:Stance(p['stance']) for p in run['participants']}
        snapshots=run['snapshots']
        assert [(s['turn'],s['phase']) for s in snapshots]==[(0,'start')]+[(t,'turn_end') for t in range(1,49)]+[(48,'end')]
        def check(s):
            assert {canonical_pair(p['left_id'],p['right_id']):p for p in s['pairs']}=={p:to_jsonable(v) for p,v in states.items()}
        check(snapshots[0])
        for t in run['turns']:
            p=canonical_pair(t['actor'],t['target'])
            history[p].append(Event(**t,approach=RuleTalkInitiator._approach(stances[t['actor']])))
            states[p]=replace(evaluator.evaluate(history[p]).pair,online_known=states[p].online_known)
            check(snapshots[t['turn']])
        check(snapshots[-1])

def old_mechanisms(start,end):
    old={}
    prefixes=tuple('{"case_id":"'+c.id+'",' for c in CASES)
    path=ROOT/'results/mechanism-confirmed-001-1100/traces.jsonl.gz'
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if not line.startswith(prefixes): continue
            record=json.loads(line); r=record['result']
            if start<=r['root_seed']<=end:
                old[(record['case_id'],r['root_seed'])]=digest(r)
    assert len(old)==8*(end-start+1)
    return old

def audit(output,follow=False,compare_previous=True,write_report=True):
    output=Path(output)
    deadline=time.monotonic()+1800
    def wait_file(path):
        while follow and not path.exists() and time.monotonic()<deadline: time.sleep(2)
        assert path.exists(),f'missing completion file: {path}'
    wait_file(output/'design.json')
    design=json.loads((output/'design.json').read_text(encoding='utf-8'))
    start,end=design['seed_start'],design['seed_end']
    allrows=[]; baseline_matches=legacy_matches=pair_rows=0
    old=None
    for pair in PAIRS:
        key='-'.join(pair); folder=output/key
        wait_file(folder/'complete.json')
        completed=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        for name,h in completed['output_sha256'].items(): assert _sha256(folder/name)==h
        previous=None
        if compare_previous:
            previous=gzip.open(ROOT/'results/pair-identity-101-1100'/key/'traces.jsonl.gz','rt',encoding='utf-8')
            for _ in range((start-101)*4): next(previous)
            if pair==('akane','midori'): old=old_mechanisms(start,end)
        try:
            with gzip.open(folder/'traces.jsonl.gz','rt',encoding='utf-8') as f, \
                 (folder/'pairs.csv').open(encoding='utf-8-sig',newline='') as pf, \
                 (folder/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as sf:
                pr,sr=csv.DictReader(pf),csv.DictReader(sf)
                for seed in range(start,end+1):
                    results={}
                    for case in CASES:
                        record=json.loads(next(f))
                        raw=record['result']
                        assert record['case_id']==case.id and raw['root_seed']==seed
                        results[case.id]=raw
                        cfg=pair_config(pair,48)
                        for i,c in enumerate(('A','B')):
                            assert raw['runs'][c]['condition']==to_jsonable(cfg.conditions[i])
                            expected_rules=to_jsonable(replace(cfg.rules,
                                selection=replace(cfg.rules.selection,online_known_bonus=float(case.selection)),
                                reaction=replace(cfg.rules.reaction,online_known_bonus=case.reaction)))
                            assert raw['runs'][c]['rules']==expected_rules
                        rebuild(raw)
                        for h in HORIZONS:
                            cp=checkpoint(raw,h)
                            if previous and case.id==BASE:
                                assert cp==json.loads(next(previous))
                                baseline_matches+=1
                            if old is not None and pair==('akane','midori') and h==12:
                                compact=compact_result(cp)
                                assert digest(compact)==old[(case.id,seed)]
                                legacy_matches+=1
                            pairs,row=extract(cp); row.update(known_pair=key,case_id=case.id)
                            assert next(sr)==strings(row)
                            allrows.append(row)
                            for p in pairs:
                                p.update(known_pair=key,case_id=case.id)
                                assert next(pr)==strings(p); pair_rows+=1
                    check_invariants(results,pair)
                    if (seed-start+1)%250==0: print(f'Audit {key}: {seed-start+1}/{end-start+1}',flush=True)
                assert next(f,None) is None and next(pr,None) is None and next(sr,None) is None
        finally:
            if previous: previous.close()
        print(f'PASS {key}: full state reconstruction and intervention invariants',flush=True)
    wait_file(output/'manifest.json')
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['output_sha256'].items(): assert _sha256(output/name)==h,name
    for name,h in manifest['source_sha256'].items(): assert _sha256(ROOT/name)==h,name
    summary,contrasts=summarize(allrows)
    for name,records in (('summary',summary),('contrasts',contrasts)):
        assert records==json.loads((output/(name+'.json')).read_text(encoding='utf-8'))
        with (output/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:
            assert list(csv.DictReader(f))==[strings(r) for r in records]
    assert pair_rows==manifest['pair_rows']
    report=dict(status='pass',actual_condition_runs=manifest['actual_condition_runs'],
        condition_checkpoints=manifest['condition_checkpoints'],pair_rows=pair_rows,
        baseline_AB_checkpoints_matched=baseline_matches,legacy_12_turn_cases_matched=legacy_matches,
        all_48_turn_states_reconstructed=True,all_off_behavior_equal=True,clamp_bridge_behavior_matches_donor=True,
        all_192_estimates_168_contrasts_recomputed=True,source_and_output_hashes_match=True)
    if write_report: _write_json(ROOT/'validation/pair-mechanism-audit.json',report)
    print(json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',nargs='?',default=str(ROOT/'results/pair-mechanism-101-1100'))
    p.add_argument('--follow',action='store_true')
    a=p.parse_args(); audit(a.output,follow=a.follow)
