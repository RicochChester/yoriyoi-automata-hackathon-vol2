"""Saved-trace audit: replay original choices/topics/reactions and pair states."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.early_recontact_experiment import CASES, DESIGN, PAIRS, permutation, metrics, summarize, verify_prefix
from poc.ab_poc.bridge_experiment import _sha256, _write_json
from poc.ab_poc.pair_identity_experiment import pair_config
from poc.ab_poc.domain import canonical_pair, PairState, OnlineKnown, OnsiteRelationship, Event, ResponseKind, to_jsonable
from poc.ab_poc.contracts import ParticipantContext
from poc.ab_poc.rule_providers import RuleTalkInitiator, RuleResponseProvider
from poc.ab_poc.engine import _stable_provider_seed
from poc.ab_poc.task_information_experiment import InformationState, clone_rng
from poc.ab_poc.task_need_matched_experiment import distribution
from validation.verify_event_formats import verify_run, strings

def verify(raw,case,novel=True):
    verify_run(raw,'task_weak',DESIGN)
    seed=raw['root_seed']; pair=canonical_pair(*raw['runs']['B']['condition']['online_known_pairs'][0])
    cfg=pair_config(pair,48); theta,permuted=CASES[case]
    assert raw['participants']==to_jsonable(cfg.participants)
    for index,(c,r) in enumerate(raw['runs'].items()):
        condition=cfg.conditions[index]
        assert r['participants']==to_jsonable(cfg.participants) and r['rules']==to_jsonable(cfg.rules)
        assert r['condition']==to_jsonable(condition)
        people={p.id:p for p in cfg.participants}; mapping=permutation(list(people),seed)
        assert r['information']['mapping']==mapping and r['information']['case_id']==case
        assert r['information']['theta']==theta and r['information']['permuted']==permuted
        info=InformationState(list(people),seed,novel); received=Counter(); events=[]; last={p:None for p in people}
        assert info.snapshot()==r['information']['initial']
        initiator=RuleTalkInitiator(); responder=RuleResponseProvider()
        for i,(t,e) in enumerate(zip(r['turns'],r['event_log']),1):
            actor,target=t['actor'],t['target']; rng=random.Random(_stable_provider_seed(seed,i))
            states={canonical_pair(p['left_id'],p['right_id']):PairState(**p) for p in r['snapshots'][i-1]['pairs']}
            memory=condition.online_experience.memory_for(actor) if condition.online_experience else ()
            context=ParticipantContext(turn=i,actor=people[actor],candidates=tuple(p for p in cfg.participants if p.id!=actor),pair_states=states,
                events=tuple(events),last_target=last[actor],rules=cfg.rules,rng=rng,online_memories=memory)
            draw=rng.random(); assert e['selection_draw']==draw
            assert info.snapshot()==e['information_before']
            token,available=info.query(actor,target) if i<=6 else (None,False)
            assert e['requested_token']==token and e['information_available']==available
            assert e['transferred_token']==(token if available else None)
            if i<=6: assert not e['selection']
            else:
                candidates=[p.id for p in context.candidates]; assert list(e['selection'])==candidates
                weights=[]
                for candidate in context.candidates:
                    base=initiator.selection_weight(context,candidate)
                    actual=received[actor,candidate.id]
                    assigned=received[actor,mapping[actor][candidate.id]] if permuted else actual
                    weight=base*2**(theta*assigned/3); weights.append(weight); v=e['selection'][candidate.id]
                    assert v['base']==base and v['actual_count']==actual and v['assigned_count']==assigned and v['weight']==weight
                if permuted: assert sorted(v['actual_count'] for v in e['selection'].values())==sorted(v['assigned_count'] for v in e['selection'].values())
                threshold=draw*sum(weights); cumulative=0; chosen=candidates[-1]
                for candidate,weight in zip(candidates,weights):
                    assert e['selection'][candidate]['probability']==weight/sum(weights)
                    cumulative+=weight
                    if cumulative>threshold and chosen==candidates[-1]:
                        chosen=candidate; break
                assert chosen==target
                for candidate,weight in zip(candidates,weights): assert e['selection'][candidate]['probability']==weight/sum(weights)
            history=tuple(x for x in events if canonical_pair(x.actor,x.target)==canonical_pair(actor,target))
            topic=initiator._choose_topic(context,people[target],history)
            assert topic==t['topic']==e['natural_topic']==e['chosen_topic']
            jitter=clone_rng(rng).choice(cfg.rules.reaction.jitter_values)
            points=responder.reaction_points(context,people[target],topic,history)
            assert e['jitter']==jitter and e['reaction_points']==points
            assert e['shared_interest']==bool(set(people[actor].interests)&set(people[target].interests))
            assert e['topic_match']==(topic in people[target].interests)
            assert e['response_probabilities']==distribution(points-jitter,r['rules']['reaction'])
            expected=ResponseKind.POSITIVE if points>=cfg.rules.reaction.positive_threshold else ResponseKind.NEUTRAL if points>=cfg.rules.reaction.neutral_threshold else ResponseKind.MISALIGNED
            assert t['response']==expected
            if available: info.receive(actor,target,token); received[actor,target]+=1
            assert e['information_after']==info.snapshot()
            events.append(Event(**t)); last[actor]=target
        assert r['information']['final']==info.snapshot()
        assert all(p['onsite_relationship']=='未形成' for p in r['snapshots'][6]['pairs'])

def check_bundle(records):
    rows=[]
    verify_prefix(records)
    for rec in records:
        verify(rec['result'],rec['case_id']); rows.extend(metrics(rec['result'],rec['case_id']))
    return rows

def audit(output,report):
    out=Path(output); design=json.loads((out/'design.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    for path,digest in manifest['source_sha256'].items(): assert _sha256(ROOT/path)==digest,path
    for path,digest in manifest['output_sha256'].items(): assert _sha256(out/path)==digest,path
    rows=[]; records=0
    for pair in PAIRS:
        key='-'.join(pair)
        with gzip.open(out/key/'traces.jsonl.gz','rt',encoding='utf-8') as tf,ProcessPoolExecutor(max_workers=2) as pool:
            def bundles():
                for seed in range(design['start'],design['end']+1):
                    bundle=[json.loads(next(tf)) for _ in CASES]
                    assert all(r['result']['root_seed']==seed for r in bundle)
                    assert all(canonical_pair(*r['result']['runs']['B']['condition']['online_known_pairs'][0])==pair for r in bundle)
                    yield bundle
                assert next(tf,None) is None
            tasks=iter(bundles()); pending=deque(); local=[]
            def submit():
                value=next(tasks,None)
                if value is not None: pending.append(pool.submit(check_bundle,value))
            for _ in range(4): submit()
            while pending:
                local.extend(pending.popleft().result()); submit()
                if len(local)%1500==0: print(f'audit {key}: {len(local)//15} seeds',flush=True)
            rows.extend(local); records+=len(local)//3
        with (out/key/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as f: assert list(csv.DictReader(f))==[strings(r) for r in local]
    summary,contrasts=summarize(rows)
    for name,data in [('summary',summary),('contrasts',contrasts)]:
        assert json.loads((out/(name+'.json')).read_text(encoding='utf-8'))==data
        with (out/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f: assert list(csv.DictReader(f))==[strings(r) for r in data]
    assert records*2==manifest['condition_runs']==(design['end']-design['start']+1)*6*5*2
    result=dict(status='pass',condition_runs=records*2,conversations=records*2*48,seed_metric_rows=len(rows),summary_rows=len(summary),contrast_rows=len(contrasts),manifest_sha256=_sha256(out/'manifest.json'))
    _write_json(Path(report),result); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--report',required=True)
    a=p.parse_args(); audit(a.output,a.report)
