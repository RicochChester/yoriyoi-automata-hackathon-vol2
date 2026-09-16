"""Negative control: information need changes, social inputs stay matched."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

from .task_information_experiment import InformationProvider, InformationState, clone_rng, PROJECT_ROOT
from .task_topic_pathway_experiment import inputs, PREVIOUS
from .bridge_experiment import _sha256, _write_json, _write_csv
from .domain import canonical_pair, ResponseKind, Stance
from .engine import run_ab
from .event_format_experiment import extract_event, MEASURES, estimate_ci
from .event_format_48_experiment import DESIGN
from .mechanism_experiment import compact_result
from .pair_identity_experiment import PAIRS, pair_config
from .rule_providers import RuleResponseProvider

VERSION='task-need-matched-v1'
CASES=('task_weak','task_medium','task_strong')
EXTRA=('confirmation_requests','information_transfers','information_coverage','information_complete')

def distribution(base,rules):
    def label(points):
        return ResponseKind.POSITIVE.value if points>=rules['positive_threshold'] else ResponseKind.NEUTRAL.value if points>=rules['neutral_threshold'] else ResponseKind.MISALIGNED.value
    outcomes=[label(base+j) for j in rules['jitter_values']]
    return {r.value:outcomes.count(r.value)/len(outcomes) for r in ResponseKind}

class ProbabilityResponder(RuleResponseProvider):
    def __init__(self,owner): self.owner=owner
    def reaction_points(self,context,target,topic,pair_events):
        jitter=clone_rng(context.rng).choice(context.rules.reaction.jitter_values)
        points=super().reaction_points(context,target,topic,pair_events)
        rules=self.owner.donor['rules']['reaction']
        self.owner.log[-1].update(jitter=jitter,reaction_points=points,
            shared_interest=bool(self.shared_interests(context.actor,target)),topic_match=topic in target.interests,
            response_probabilities=distribution(points-jitter,rules))
        return points

class MatchedProvider(InformationProvider):
    name='task-need-matched'; version=VERSION
    def __init__(self,donor,seed,case,novel):
        super().__init__(donor,seed,case,novel,topic_link=False)
        self.responder=ProbabilityResponder(self)

def simulate(donor,case,novel):
    pair=tuple(donor['runs']['B']['condition']['online_known_pairs'][0]); seed=donor['root_seed']
    providers=iter(MatchedProvider(donor['runs'][c],seed,case,novel) for c in ('A','B'))
    full=run_ab(seed=seed,config=pair_config(pair,48),participant_provider_factory=lambda:next(providers))
    raw=compact_result(full)
    for c,r in full['runs'].items():
        raw['runs'][c].update(snapshots=r['snapshots'],turns=[{k:t[k] for k in ('turn','actor','target','topic','approach','response')} for t in r['turns']],
            event_log=r['participant_audit']['turns'],information=r['participant_audit'])
    return raw

def verify(raw,donor,case,novel):
    from validation.verify_event_formats import verify_run
    verify_run(raw,case,DESIGN)
    for c,r in raw['runs'].items():
        d=donor['runs'][c]
        assert all(r[k]==d[k] for k in ('participants','condition','rules','turns','snapshots'))
        people={p['id']:p for p in r['participants']}
        state=InformationState(list(people),raw['root_seed'],novel)
        assert state.snapshot()==r['information']['initial']
        rules=r['rules']['reaction']; known={canonical_pair(*p) for p in r['condition']['online_known_pairs']}
        for i,(t,e) in enumerate(zip(r['turns'],r['event_log'])):
            assert e['information_before']==state.snapshot()
            token,available=state.query(t['actor'],t['target']) if t['turn'] in DESIGN.positions(case) else (None,False)
            assert e.get('requested_token')==token
            if t['turn'] in DESIGN.positions(case): assert e['information_available']==available
            assert e['transferred_token']==(token if available else None)
            if available: state.receive(t['actor'],t['target'],token)
            assert e['information_after']==state.snapshot()
            a,b=people[t['actor']],people[t['target']]; pair=canonical_pair(t['actor'],t['target'])
            shared=bool(set(a['interests'])&set(b['interests'])); online=pair in known
            positive=any(canonical_pair(x['actor'],x['target'])==pair and x['response']==ResponseKind.POSITIVE for x in r['turns'][:i])
            match=t['topic'] in b['interests']
            base=(rules['base']+rules['online_known_bonus']*online+rules['topic_match_bonus']*match+
                rules['prior_positive_bonus']*positive+rules['proactive_actor_bonus']*(a['stance']==Stance.PROACTIVE)+
                rules['cautious_actor_penalty']*(a['stance']==Stance.CAUTIOUS and not(online or shared))+
                rules['cautious_target_penalty']*(b['stance']==Stance.CAUTIOUS and not(online or shared or positive)))
            assert e['shared_interest']==shared and e['topic_match']==match and t['topic']==e['natural_topic']==e['chosen_topic']
            assert e['jitter'] in rules['jitter_values'] and e['reaction_points']==base+e['jitter']
            assert e['response_probabilities']==distribution(base,rules)
            points=base+e['jitter']
            expected=ResponseKind.POSITIVE if points>=rules['positive_threshold'] else ResponseKind.NEUTRAL if points>=rules['neutral_threshold'] else ResponseKind.MISALIGNED
            assert t['response']==expected
        assert state.snapshot()==r['information']['final']

def verify_matched(familiar,novel):
    for c in ('A','B'):
        a,b=familiar['runs'][c],novel['runs'][c]
        assert a['turns']==b['turns'] and a['snapshots']==b['snapshots']
        for x,y in zip(a['event_log'],b['event_log']):
            for k in ('jitter','reaction_points','shared_interest','topic_match','response_probabilities'):
                assert x[k]==y[k]

def metrics(raw,case,profile):
    _,row=extract_event(raw,case,DESIGN); row['profile']=profile
    for c,r in raw['runs'].items():
        log=r['event_log']; info=r['information']['final']
        row.update({f'confirmation_requests_{c}':sum(e.get('requested_token') is not None for e in log),
                    f'information_transfers_{c}':sum(e['transferred_token'] is not None for e in log),
                    f'information_coverage_{c}':info['task_progress'],f'information_complete_{c}':int(info['complete'])})
    return row

def one(item):
    case,donor,_=item; outputs=[]
    for novel in (False,True):
        raw=simulate(donor,case,novel); verify(raw,donor,case,novel)
        profile='novel' if novel else 'familiar'
        outputs.append((dict(case_id=case,profile=profile,result=raw),metrics(raw,case,profile)))
    verify_matched(outputs[0][0]['result'],outputs[1][0]['result'])
    return outputs

def summarize(rows):
    seeds=sorted({r['seed'] for r in rows}); keys=['-'.join(p) for p in PAIRS]
    by={(r['known_pair'],r['seed'],r['case_id'],r['profile']):r for r in rows}
    assert len(by)==len(rows)==len(seeds)*6*3*2
    out=[]
    for key in keys+['pooled']:
        selected=keys if key=='pooled' else [key]
        for case in CASES:
            for metric in MEASURES+EXTRA:
                r=dict(known_pair=key,case_id=case,metric=metric,seeds=len(seeds))
                for c in ('A','B'):
                    values={p:[mean(by[k,s,case,p][metric+'_'+c] for k in selected) for s in seeds] for p in ('familiar','novel')}
                    for profile,v in values.items(): r.update({f'{profile}_{c}_{k}':x for k,x in estimate_ci(v).items()})
                    r.update({f'novel_minus_familiar_{c}_{k}':x for k,x in estimate_ci([b-a for a,b in zip(values['familiar'],values['novel'])]).items()})
                out.append(r)
    return out

def run(output,start=101,end=1100,workers=2):
    if not 101<=start<=end<=1100 or workers not in (1,2): raise ValueError('range/workers')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    sources=list((PROJECT_ROOT/'poc').rglob('*.py'))+list((PROJECT_ROOT/'poc/config').glob('*.json'))+[PROJECT_ROOT/'docs/TASK_NEED_MATCHED_PLAN.md',PROJECT_ROOT/'validation/verify_task_need_matched.py',PROJECT_ROOT/'validation/verify_event_formats.py']
    hashes={p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in sources}
    prior=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    _write_json(out/'design.json',dict(version=VERSION,start=start,end=end,topic_link=False,event_settings=DESIGN.settings,
        source_sha256=hashes,previous_manifest_sha256=_sha256(PREVIOUS/'manifest.json')))
    rows=[]; started=datetime.now(timezone.utc).isoformat()
    for pair in PAIRS:
        key='-'.join(pair); folder=out/key; folder.mkdir()
        assert _sha256(PREVIOUS/key/'traces.jsonl.gz')==prior['output_sha256'][f'{key}/traces.jsonl.gz']
        with gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as tf,(folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf,ProcessPoolExecutor(max_workers=workers) as pool:
            tasks=iter(inputs(key,start,end)); pending=deque(); writer=None; done=0
            def submit():
                item=next(tasks,None)
                if item is not None: pending.append(pool.submit(one,item))
            for _ in range(workers*2): submit()
            while pending:
                result=pending.popleft().result(); submit()
                for rec,row in result:
                    if writer is None: writer=csv.DictWriter(sf,fieldnames=list(row)); writer.writeheader()
                    tf.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n'); writer.writerow(row); rows.append(row)
                done+=1
                if done%300==0: print(f'{key}: {done//3}/{end-start+1} seeds verified',flush=True)
        assert done==3*(end-start+1)
        _write_json(folder/'complete.json',dict(records=done*2,output_sha256={p.name:_sha256(p) for p in folder.iterdir()}))
    summary=summarize(rows); _write_json(out/'summary.json',summary); _write_csv(out/'summary.csv',summary)
    assert all(_sha256(PROJECT_ROOT/k)==v for k,v in hashes.items())
    _write_json(out/'manifest.json',dict(status='complete',started=started,ended=datetime.now(timezone.utc).isoformat(),condition_runs=len(rows)*2,
        matched_familiar_novel_comparisons=len(rows),source_sha256=hashes,
        output_sha256={p.relative_to(out).as_posix():_sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'complete: {len(rows)*2} conditions',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed-start',type=int,default=101); p.add_argument('--seed-end',type=int,default=1100); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); run(a.output,a.seed_start,a.seed_end,a.workers)
