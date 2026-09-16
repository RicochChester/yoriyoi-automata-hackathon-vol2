"""Hypothetical post-task partner-selection sensitivity; no social score bonus."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gzip
import json
import math
from pathlib import Path
import random
from statistics import mean

from .bridge_experiment import PROJECT_ROOT, _sha256, _write_json, _write_csv
from .task_topic_pathway_experiment import inputs, PREVIOUS
from .task_need_matched_experiment import MatchedProvider, verify as verify_task, distribution
from .task_information_experiment import clone_rng
from .event_format_experiment import estimate_ci
from .pair_identity_experiment import PAIRS, pair_config
from .mechanism_experiment import compact_result
from .domain import canonical_pair, Event, PairState, OnlineKnown, RelationshipThresholds, ResponseKind, Stance, to_jsonable
from .engine import run_ab, _stable_provider_seed
from .rule_providers import RuleTalkInitiator, RuleRelationshipEvaluator

SETTINGS=json.loads((PROJECT_ROOT/'poc/config/post_task_recontact_v1.json').read_text(encoding='utf-8'))
if (SETTINGS['task_turns']!=48 or SETTINGS['free_turns']!=24 or SETTINGS['task_case']!='task_strong'
        or SETTINGS['theta_values']!=[-1,0,1] or SETTINGS['received_count_normalizer']!=3
        or not math.isfinite(SETTINGS['weight_base']) or SETTINGS['weight_base']<=1):
    raise ValueError('unsupported v1 design')
TASK=SETTINGS['task_turns']; TOTAL=TASK+SETTINGS['free_turns']; CASE=SETTINGS['task_case']

class PostInitiator(RuleTalkInitiator):
    def __init__(self,owner,task_initiator): self.owner,self.task_initiator=owner,task_initiator
    def selection_weight(self,context,candidate):
        base=super().selection_weight(context,candidate)
        score=self.owner.received[context.actor.id,candidate.id]/SETTINGS['received_count_normalizer']
        weight=base*SETTINGS['weight_base']**(self.owner.theta*score)
        self.owner.log[-1]['selection'][candidate.id]=dict(base=base,score=score,weight=weight)
        return weight
    def initiate(self,context):
        if context.turn<=TASK: return self.task_initiator.initiate(context)
        self.owner.pending=None
        e=self.owner.log[-1]; e['selection_draw']=clone_rng(context.rng).random()
        initiation=super().initiate(context)
        total=sum(v['weight'] for v in e['selection'].values())
        for v in e['selection'].values(): v['probability']=v['weight']/total
        e.update(natural_topic=initiation.topic,chosen_topic=initiation.topic)
        return initiation

class PostProvider(MatchedProvider):
    name='post-task-recontact'; version=SETTINGS['version']
    def __init__(self,donor,seed,theta,novel=True):
        super().__init__(donor,seed,CASE,novel)
        self.theta=theta; self.received=Counter()
        self.initiator=PostInitiator(self,self.initiator)
    def select_event_actor(self,turn,planned_actor,participants,events):
        if turn<=TASK: return super().select_event_actor(turn,planned_actor,participants,events)
        self.log.append(dict(turn=turn,active=False,actual_actor=planned_actor,selection={},
            information_before=self.info.snapshot(),transferred_token=None))
        return planned_actor
    def propose(self,context):
        proposal=super().propose(context)
        if self.pending: self.received[self.pending[0],self.pending[1]]+=1
        self.log[-1]['actual_target']=proposal.target
        return proposal

def simulate(donor,theta,novel=True,ordinary_free=False):
    pair=tuple(donor['runs']['B']['condition']['online_known_pairs'][0]); seed=donor['root_seed']
    cfg=pair_config(pair,TASK); cfg=replace(cfg,rules=replace(cfg.rules,turns=TOTAL))
    class OrdinaryProvider(PostProvider):
        def propose(self,context):
            if context.turn>TASK:
                self.pending=None
                old=self.initiator; self.initiator=RuleTalkInitiator()
                try: return super().propose(context)
                finally: self.initiator=old
            return super().propose(context)
    cls=OrdinaryProvider if ordinary_free else PostProvider
    providers=iter(cls(donor['runs'][c],seed,theta,novel) for c in ('A','B'))
    full=run_ab(seed=seed,config=cfg,participant_provider_factory=lambda:next(providers)); raw=compact_result(full)
    for c,r in full['runs'].items():
        raw['runs'][c].update(snapshots=r['snapshots'],turns=[{k:t[k] for k in ('turn','actor','target','topic','approach','response')} for t in r['turns']],
            event_log=r['participant_audit']['turns'],information=r['participant_audit'])
    return raw

def verify(raw,donor,theta,novel=True):
    short=deepcopy(raw)
    for c,r in short['runs'].items():
        r['rules']['turns']=TASK; r['turns']=r['turns'][:TASK]; r['turn_order']=r['turn_order'][:TASK]; r['event_log']=r['event_log'][:TASK]
        r['snapshots']=r['snapshots'][:TASK+1]+[dict(r['snapshots'][TASK],phase='end')]
        r['information']['final']=r['event_log'][-1]['information_after']
    verify_task(short,donor,CASE,novel)
    for c,r in raw['runs'].items():
        assert r['rules']['turns']==TOTAL
        people={p['id']:p for p in r['participants']}; ids=list(people); order=ids[:]; random.Random(raw['root_seed']).shuffle(order)
        known={canonical_pair(*p) for p in r['condition']['online_known_pairs']}
        histories={p:[] for p in PAIRS}; states={p:PairState(*p,online_known=OnlineKnown.DIRECT if p in known else OnlineKnown.NONE) for p in PAIRS}
        evaluator=RuleRelationshipEvaluator(RelationshipThresholds(**r['rules']['relationship']))
        received=Counter(); last={p:None for p in ids}
        assert len(r['turns'])==len(r['event_log'])==TOTAL
        assert [(s['turn'],s['phase']) for s in r['snapshots']]==[(0,'start')]+[(t,'turn_end') for t in range(1,TOTAL+1)]+[(TOTAL,'end')]
        for i,(t,e) in enumerate(zip(r['turns'],r['event_log']),1):
            assert t['turn']==e['turn']==i
            actor,target=t['actor'],t['target']; pair=canonical_pair(actor,target)
            if i<=TASK:
                if e['transferred_token'] is not None: received[actor,target]+=1
            else:
                assert actor==order[(i-1)%4] and e['actual_actor']==actor and e['actual_target']==target
                assert e['transferred_token'] is None and e['information_before']==e['information_after']==r['event_log'][TASK-1]['information_after']
                candidates=[p for p in ids if p!=actor]; assert list(e['selection'])==candidates
                weights=[]; sw=r['rules']['selection']
                for candidate in candidates:
                    h=histories[canonical_pair(actor,candidate)]; shared=bool(set(people[actor]['interests'])&set(people[candidate]['interests']))
                    base=sw['base']+sw['online_known_bonus']*(canonical_pair(actor,candidate) in known)+sw['shared_interest_bonus']*shared+sw['prior_positive_bonus']*any(x.response==ResponseKind.POSITIVE for x in h)
                    if last[actor]==candidate: base*=sw['repeated_target_multiplier']
                    score=received[actor,candidate]/SETTINGS['received_count_normalizer']; weight=base*SETTINGS['weight_base']**(theta*score)
                    v=e['selection'][candidate]; assert v['base']==base and v['score']==score and v['weight']==weight and weight>0
                    weights.append(weight)
                cumulative=0; chosen=candidates[-1]; threshold=e['selection_draw']*sum(weights)
                assert e['selection_draw']==random.Random(_stable_provider_seed(raw['root_seed'],i)).random()
                for candidate,weight in zip(candidates,weights):
                    cumulative+=weight
                    if threshold<cumulative: chosen=candidate; break
                assert target==chosen
                for candidate,weight in zip(candidates,weights): assert e['selection'][candidate]['probability']==weight/sum(weights)
                # Recompute the unchanged reaction equation from actual histories.
                a,b=people[actor],people[target]; shared=bool(set(a['interests'])&set(b['interests'])); online=pair in known
                positive=any(x.response==ResponseKind.POSITIVE for x in histories[pair]); rr=r['rules']['reaction']; match=t['topic'] in b['interests']
                assert e['shared_interest']==shared and e['topic_match']==match and e['chosen_topic']==e['natural_topic']==t['topic']
                base=(rr['base']+rr['online_known_bonus']*online+rr['topic_match_bonus']*match+rr['prior_positive_bonus']*positive+
                    rr['proactive_actor_bonus']*(a['stance']==Stance.PROACTIVE)+rr['cautious_actor_penalty']*(a['stance']==Stance.CAUTIOUS and not(online or shared))+
                    rr['cautious_target_penalty']*(b['stance']==Stance.CAUTIOUS and not(online or shared or positive)))
                assert e['reaction_points']==base+e['jitter'] and e['jitter'] in rr['jitter_values'] and e['response_probabilities']==distribution(base,rr)
                expected=ResponseKind.POSITIVE if e['reaction_points']>=rr['positive_threshold'] else ResponseKind.NEUTRAL if e['reaction_points']>=rr['neutral_threshold'] else ResponseKind.MISALIGNED
                assert t['response']==expected
            histories[pair].append(Event(**t)); states[pair]=replace(evaluator.evaluate(histories[pair]).pair,online_known=states[pair].online_known)
            assert {canonical_pair(p['left_id'],p['right_id']):p for p in r['snapshots'][i]['pairs']}=={p:to_jsonable(v) for p,v in states.items()}
            last[actor]=target
        assert r['snapshots'][-1]['pairs']==r['snapshots'][-2]['pairs'] and r['turn_order']==[t['actor'] for t in r['turns']]

MEASURES=('bridge_before','bridge_after','bridge_rate','new_bridge','unformed_turns','all_formed','free_known_share','free_bridge_share','source_contact_share','source_exposure_mean','immediate_expected_exposure_shift')
def metrics(raw,theta):
    known=canonical_pair(*raw['runs']['B']['condition']['online_known_pairs'][0]); bridge={p for p in PAIRS if len(set(p)&set(known))==1}
    row=dict(seed=raw['root_seed'],known_pair='-'.join(known),theta=theta)
    for c,r in raw['runs'].items():
        first={}
        for snap in r['snapshots']:
            if snap['phase']=='turn_end':
                for p in snap['pairs']:
                    key=canonical_pair(p['left_id'],p['right_id'])
                    if p['onsite_relationship']!='未形成': first.setdefault(key,snap['turn'])
        before=sum(first.get(p,TOTAL+1)<=TASK for p in bridge); after=sum(p in first for p in bridge)
        turns=r['turns'][TASK:]; logs=r['event_log'][TASK:]; n=len(turns)
        shifts=[]
        for e in logs:
            vs=list(e['selection'].values()); baseline=sum(v['base'] for v in vs)
            shift=sum(v['probability']*v['score']-v['base']/baseline*v['score'] for v in vs)
            assert theta*shift>=-1e-12
            shifts.append(shift)
        values=dict(bridge_before=before,bridge_after=after,bridge_rate=after/4,new_bridge=after-before,
            unformed_turns=sum(first.get(p,TOTAL+1)-1 for p in bridge)/4,all_formed=int(after==4),
            free_known_share=sum(canonical_pair(t['actor'],t['target'])==known for t in turns)/n,
            free_bridge_share=sum(canonical_pair(t['actor'],t['target']) in bridge for t in turns)/n,
            source_contact_share=sum(e['selection'][t['target']]['score']>0 for t,e in zip(turns,logs))/n,
            source_exposure_mean=mean(e['selection'][t['target']]['score'] for t,e in zip(turns,logs)),
            immediate_expected_exposure_shift=mean(shifts))
        row.update({f'{k}_{c}':v for k,v in values.items()})
    return row

def one(item):
    donor=item; results=[]
    for theta in SETTINGS['theta_values']:
        raw=simulate(donor,theta); verify(raw,donor,theta)
        results.append((dict(theta=theta,result=raw),metrics(raw,theta)))
    return results

def summarize(rows):
    seeds=sorted({r['seed'] for r in rows}); keys=['-'.join(p) for p in PAIRS]
    by={(r['known_pair'],r['seed'],r['theta']):r for r in rows}
    assert len(by)==len(rows)==len(seeds)*6*3
    output=[]
    for key in keys+['pooled']:
        selected=keys if key=='pooled' else [key]
        for theta in SETTINGS['theta_values']:
            for metric in MEASURES:
                r=dict(known_pair=key,theta=theta,metric=metric,seeds=len(seeds))
                for c in ('A','B'):
                    values=[mean(by[k,s,theta][metric+'_'+c] for k in selected) for s in seeds]
                    diffs=[mean(by[k,s,theta][metric+'_'+c]-by[k,s,0][metric+'_'+c] for k in selected) for s in seeds]
                    r.update({f'{c}_{k}':v for k,v in estimate_ci(values).items()}); r.update({f'difference_{c}_{k}':v for k,v in estimate_ci(diffs,24).items()})
                gaps=[mean(by[k,s,theta][metric+'_B']-by[k,s,theta][metric+'_A'] for k in selected) for s in seeds]
                changes=[mean((by[k,s,theta][metric+'_B']-by[k,s,theta][metric+'_A'])-(by[k,s,0][metric+'_B']-by[k,s,0][metric+'_A']) for k in selected) for s in seeds]
                r.update({f'B_minus_A_{k}':v for k,v in estimate_ci(gaps).items()}); r.update({f'gap_change_{k}':v for k,v in estimate_ci(changes,12).items()}); output.append(r)
    return output

def run(output,start=101,end=1100,workers=2):
    if not 101<=start<=end<=1100 or workers not in (1,2): raise ValueError('range/workers')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    sources=list((PROJECT_ROOT/'poc').rglob('*.py'))+list((PROJECT_ROOT/'poc/config').glob('*.json'))+[PROJECT_ROOT/'docs/POST_TASK_RECONTACT_PLAN.md',PROJECT_ROOT/'validation/verify_post_task_recontact.py',PROJECT_ROOT/'validation/verify_event_formats.py']
    hashes={p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in sources}
    prior=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    _write_json(out/'design.json',dict(settings=SETTINGS,start=start,end=end,source_sha256=hashes,previous_manifest_sha256=_sha256(PREVIOUS/'manifest.json')))
    rows=[]; started=datetime.now(timezone.utc).isoformat()
    for pair in PAIRS:
        key='-'.join(pair); folder=out/key; folder.mkdir()
        assert _sha256(PREVIOUS/key/'traces.jsonl.gz')==prior['output_sha256'][f'{key}/traces.jsonl.gz']
        with gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as tf,(folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf,ProcessPoolExecutor(max_workers=workers) as pool:
            tasks=(d for case,d,_ in inputs(key,start,end) if case==CASE); pending=deque(); writer=None; done=0
            def submit():
                item=next(tasks,None)
                if item is not None: pending.append(pool.submit(one,item))
            for _ in range(workers*2): submit()
            while pending:
                outputs=pending.popleft().result(); submit()
                for rec,row in outputs:
                    if writer is None: writer=csv.DictWriter(sf,fieldnames=list(row)); writer.writeheader()
                    tf.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n'); writer.writerow(row); rows.append(row)
                done+=1
                if done%100==0: print(f'{key}: {done}/{end-start+1} seeds verified',flush=True)
        assert done==end-start+1
        _write_json(folder/'complete.json',dict(records=done*3,output_sha256={p.name:_sha256(p) for p in folder.iterdir()}))
    s=summarize(rows); _write_json(out/'summary.json',s); _write_csv(out/'summary.csv',s)
    assert all(_sha256(PROJECT_ROOT/k)==v for k,v in hashes.items())
    _write_json(out/'manifest.json',dict(status='complete',started=started,ended=datetime.now(timezone.utc).isoformat(),condition_runs=len(rows)*2,source_sha256=hashes,
        output_sha256={p.relative_to(out).as_posix():_sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'complete: {len(rows)*2} conditions',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed-start',type=int,default=SETTINGS['seed_start']); p.add_argument('--seed-end',type=int,default=SETTINGS['seed_end']); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); run(a.output,a.seed_start,a.seed_end,a.workers)
