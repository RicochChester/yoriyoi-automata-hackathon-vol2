"""Audit duration traces, state transitions, censored times and saved statistics."""
import csv
from dataclasses import replace
import gzip
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poc.ab_poc.bridge_experiment import _sha256
from poc.ab_poc.domain import Event, OnlineKnown, PairState, RelationshipThresholds, Stance, canonical_pair, to_jsonable
from poc.ab_poc.rule_providers import RuleRelationshipEvaluator, RuleTalkInitiator
from poc.ab_poc.duration_experiment import HORIZONS, extract, summarize, verify_prefix


def _strings(row):
    return {k: '' if v is None else str(v) for k,v in row.items()}


def verify(output, *, write_report=True, compare_previous=True):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    design = json.loads((output/'design.json').read_text(encoding='utf-8'))
    for name,digest in manifest['source_sha256'].items(): assert _sha256(ROOT/name)==digest, name
    for name,digest in manifest['output_sha256'].items(): assert _sha256(output/name)==digest, name
    previous = {}
    if compare_previous:
        with gzip.open(ROOT/'results/mechanism-confirmed-001-1100/traces.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                if not line.startswith('{"case_id":"S1_R2_T1_C1",'): continue
                r=json.loads(line)['result']
                previous[r['root_seed']]=r
                if len(previous)==1100: break
    summaries=[]
    expected=[(s,h) for s in range(design['seed_start'],design['seed_end']+1) for h in HORIZONS]
    seen=[]
    reconstructed=prefix_checks=prior_checks=0
    short=None
    with gzip.open(output/'traces.jsonl.gz','rt',encoding='utf-8') as raw, \
         (output/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as sf, \
         (output/'pairs.csv').open(encoding='utf-8-sig',newline='') as pf:
        sr,pr=csv.DictReader(sf),csv.DictReader(pf)
        for line in raw:
            result=json.loads(line)
            horizon=result['runs']['A']['rules']['turns']
            seen.append((result['root_seed'],horizon))
            if horizon==12:
                if compare_previous:
                    old=previous[result['root_seed']]
                    for c in ('A','B'):
                        for key in ('turns','final_relationships','metrics','rules'):
                            assert result['runs'][c][key]==old['runs'][c][key], (result['root_seed'],key)
                    prior_checks+=1
            else:
                verify_prefix(short,result)
                prefix_checks+=2
            short=result
            pairs,summary=extract(result)
            assert next(sr)==_strings(summary)
            summaries.append(summary)
            for row in pairs: assert next(pr)==_strings(row)
            for c,run in result['runs'].items():
                ids=[p['id'] for p in run['participants']]
                known={canonical_pair(*p) for p in run['condition']['online_known_pairs']}
                states={p:PairState(*p,online_known=OnlineKnown.DIRECT if p in known else OnlineKnown.NONE)
                        for p in (canonical_pair(*v) for v in combinations(ids,2))}
                history={p:[] for p in states}
                first={p:[None,None] for p in states}
                unformed={p:0 for p in states}
                snapshots=run['snapshots']
                assert [(s['turn'],s['phase']) for s in snapshots]==[(0,'start')]+[(t,'turn_end') for t in range(1,horizon+1)]+[(horizon,'end')]
                def check(snapshot):
                    recorded={canonical_pair(p['left_id'],p['right_id']):p for p in snapshot['pairs']}
                    assert recorded=={p:to_jsonable(v) for p,v in states.items()}
                check(snapshots[0])
                evaluator=RuleRelationshipEvaluator(RelationshipThresholds(**run['rules']['relationship']))
                stances={p['id']:Stance(p['stance']) for p in run['participants']}
                for t in run['turns']:
                    pair=canonical_pair(t['actor'],t['target'])
                    event=Event(**t,approach=RuleTalkInitiator._approach(stances[t['actor']]))
                    history[pair].append(event)
                    rebuilt=evaluator.evaluate(history[pair]).pair
                    states[pair]=replace(rebuilt,online_known=states[pair].online_known)
                    check(snapshots[t['turn']])
                    for p,state in states.items():
                        stage=state.onsite_relationship.value
                        unformed[p]+=stage=='未形成'
                        if stage!='未形成' and first[p][0] is None: first[p][0]=t['turn']
                        if stage=='親しみがある' and first[p][1] is None: first[p][1]=t['turn']
                check(snapshots[-1])
                for row in (p for p in pairs if p['condition']==c):
                    p=canonical_pair(row['left_id'],row['right_id'])
                    assert [row['first_acquaintance_turn'],row['first_familiar_turn']]==first[p]
                    assert row['unformed_turns']==unformed[p]
                    reconstructed+=1
        assert next(sr,None) is None and next(pr,None) is None
    assert seen==expected
    summary=summarize(summaries)
    assert summary==json.loads((output/'summary.json').read_text(encoding='utf-8'))
    with (output/'summary.csv').open(encoding='utf-8-sig',newline='') as f: assert list(csv.DictReader(f))==[_strings(r) for r in summary]
    assert reconstructed==manifest['pair_rows'] and prefix_checks==manifest['prefix_checks']
    report={'status':'pass','paired_runs':len(seen),'pair_rows':reconstructed,'prefix_checks':prefix_checks,
            'previous_12_turn_results_matched':prior_checks,'all_snapshots_rebuilt_from_events':True,
            'formation_times_and_unformed_counts_match':True,'CSV_JSON_and_hashes_match':True}
    if write_report:
        (ROOT/'validation/duration-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))
    return report


if __name__=='__main__': verify(sys.argv[1] if len(sys.argv)>1 else ROOT/'results/duration-101-1100')
