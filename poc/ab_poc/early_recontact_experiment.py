"""Early task then frozen information-source selection; original social rules."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

from .bridge_experiment import PROJECT_ROOT, _sha256, _write_json, _write_csv
from .event_format_experiment import EventDesign, EventProvider, EventInitiator, SETTINGS as EVENT_SETTINGS, rng_for, estimate_ci
from .task_information_experiment import InformationState, clone_rng
from .task_need_matched_experiment import ProbabilityResponder
from .pair_identity_experiment import PAIRS, pair_config
from .mechanism_experiment import compact_result
from .domain import canonical_pair, to_jsonable
from .engine import run_ab

VERSION='early-recontact-v1'
CASES={'neutral':(0,False),'actual_promote':(1,False),'actual_suppress':(-1,False),
       'permuted_promote':(1,True),'permuted_suppress':(-1,True)}
HORIZONS=(12,24,48)
DESIGN=EventDesign(dict(EVENT_SETTINGS,version=VERSION,turns=48,weak_turns=list(range(1,7))))

def permutation(ids,seed):
    out={}
    for actor in ids:
        candidates=[p for p in ids if p!=actor]; shuffled=candidates[:]
        rng_for(seed,'source-label:'+actor,VERSION).shuffle(shuffled)
        out[actor]=dict(zip(candidates,shuffled))
    return out

class EarlyInitiator(EventInitiator):
    def selection_weight(self,context,candidate):
        o=self.owner; base=super().selection_weight(context,candidate)
        actual=o.received[context.actor.id,candidate.id]
        source=o.mapping[context.actor.id][candidate.id] if o.permuted else candidate.id
        count=o.received[context.actor.id,source]; weight=base*2**(o.theta*count/3)
        o.log[-1]['selection'][candidate.id]=dict(base=base,actual_count=actual,assigned_count=count,weight=weight)
        return weight
    def initiate(self,context):
        e=self.owner.log[-1]; e['selection_draw']=clone_rng(context.rng).random()
        result=super().initiate(context)
        total=sum(v['weight'] for v in e['selection'].values())
        for v in e['selection'].values(): v['probability']=v['weight']/total
        e.update(natural_topic=result.topic,chosen_topic=result.topic)
        return result

class EarlyProvider(EventProvider):
    name='early-recontact'
    def __init__(self,cfg,seed,case,novel=True):
        super().__init__(seed,'task_weak',cfg.rules.common_topic,DESIGN)
        self.theta,self.permuted=CASES[case]; self.experiment_case=case
        self.info=InformationState([p.id for p in cfg.participants],seed,novel)
        self.initial=self.info.snapshot(); self.received=Counter()
        self.mapping=permutation(list(self.info.known),seed)
        self.donor={'rules':to_jsonable(cfg.rules)}
        self.initiator=EarlyInitiator(self); self.responder=ProbabilityResponder(self)
    def select_event_actor(self,turn,planned_actor,participants,events):
        actor=super().select_event_actor(turn,planned_actor,participants,events)
        self.log[-1].update(selection={},information_before=self.info.snapshot(),transferred_token=None)
        return actor
    def propose(self,context):
        result=super().propose(context); e=self.log[-1]
        token,available=self.info.query(context.actor.id,result.target) if context.turn<=6 else (None,False)
        e.update(requested_token=token,information_available=available)
        if available:
            self.info.receive(context.actor.id,result.target,token)
            self.received[context.actor.id,result.target]+=1; e['transferred_token']=token
        e['information_after']=self.info.snapshot()
        return result
    def audit_snapshot(self):
        return dict(turns=self.log,initial=self.initial,final=self.info.snapshot(),mapping=self.mapping,
                    case_id=self.experiment_case,theta=self.theta,permuted=self.permuted)

def simulate(pair,seed,case,novel=True):
    cfg=pair_config(pair,48)
    full=run_ab(seed=seed,config=cfg,participant_provider_factory=lambda:EarlyProvider(cfg,seed,case,novel))
    raw=compact_result(full)
    raw["participants"]=to_jsonable(cfg.participants)
    for c,r in full['runs'].items():
        raw['runs'][c].update(participants=to_jsonable(cfg.participants),snapshots=r['snapshots'],turns=[{k:t[k] for k in ('turn','actor','target','topic','approach','response')} for t in r['turns']],
            event_log=r['participant_audit']['turns'],information=r['participant_audit'])
    return raw

def verify_prefix(records):
    assert [r['case_id'] for r in records]==list(CASES)
    first=records[0]['result']
    for record in records:
        raw=record['result']
        assert raw['root_seed']==first['root_seed']
        for c,r in raw['runs'].items():
            b=first['runs'][c]
            assert all(r[k]==b[k] for k in ('participants','condition','rules'))
            assert r['turns'][:6]==b['turns'][:6] and r['snapshots'][:7]==b['snapshots'][:7]
            assert all(p['onsite_relationship']=='未形成' for p in r['snapshots'][6]['pairs'])
            assert r['information']['mapping']==b['information']['mapping']
        assert raw['runs']['A']['information']['final']==raw['runs']['B']['information']['final']

MEASURES=('bridge_count','bridge_rate','unformed_turns','all_formed','free_known_share','free_bridge_share',
          'source_contact_share','unformed_bridge_contacts','selection_total_variation','expected_source_shift',
          'permutation_unchanged_fraction','information_transfers')

def metrics(raw,case):
    known=canonical_pair(*raw['runs']['B']['condition']['online_known_pairs'][0])
    bridge={p for p in PAIRS if len(set(p)&set(known))==1}; output=[]
    for horizon in HORIZONS:
        row=dict(seed=raw['root_seed'],known_pair='-'.join(known),case_id=case,horizon=horizon)
        for c,r in raw['runs'].items():
            first={}
            for snap in r['snapshots'][1:horizon+1]:
                for p in snap['pairs']:
                    if p['onsite_relationship']!='未形成': first.setdefault(canonical_pair(p['left_id'],p['right_id']),snap['turn'])
            turns=r['turns'][6:horizon]; logs=r['event_log'][6:horizon]; n=len(turns)
            after=sum(p in first for p in bridge); tv=[]; shifts=[]; unchanged=[]
            counts=Counter((e['actual_actor'],e['actual_target']) for e in r['event_log'][:6] if e['transferred_token'] is not None)
            for e in logs:
                vs=list(e['selection'].values()); total=sum(v['base'] for v in vs)
                tv.append(.5*sum(abs(v['probability']-v['base']/total) for v in vs))
                shifts.append(sum((v['probability']-v['base']/total)*v['actual_count']/3 for v in vs))
            for actor,mapping in r['information']['mapping'].items():
                unchanged.append(all(counts[actor,p]==counts[actor,q] for p,q in mapping.items()))
            values=dict(bridge_count=after,bridge_rate=after/4,unformed_turns=sum(first.get(p,horizon+1)-1 for p in bridge)/4,
                all_formed=int(after==4),free_known_share=sum(canonical_pair(t['actor'],t['target'])==known for t in turns)/n,
                free_bridge_share=sum(canonical_pair(t['actor'],t['target']) in bridge for t in turns)/n,
                source_contact_share=sum(e['selection'][t['target']]['actual_count']>0 for t,e in zip(turns,logs))/n,
                unformed_bridge_contacts=sum(canonical_pair(t['actor'],t['target']) in bridge and first.get(canonical_pair(t['actor'],t['target']),horizon+1)>=t['turn'] for t in turns),
                selection_total_variation=mean(tv),expected_source_shift=mean(shifts),permutation_unchanged_fraction=mean(unchanged),
                information_transfers=sum(counts.values()))
            row.update({f'{k}_{c}':v for k,v in values.items()})
        output.append(row)
    return output

def one(item):
    from validation.verify_early_recontact import verify
    pair,seed=item; records=[]; rows=[]
    for case in CASES:
        raw=simulate(pair,seed,case); verify(raw,case)
        records.append(dict(case_id=case,result=raw)); rows.extend(metrics(raw,case))
    verify_prefix(records)
    return records,rows

def summarize(rows):
    seeds=sorted({r['seed'] for r in rows}); keys=['-'.join(p) for p in PAIRS]
    by={(r['known_pair'],r['seed'],r['case_id'],r['horizon']):r for r in rows}
    assert len(by)==len(rows)==len(seeds)*6*5*3
    output=[]
    for key in keys+['pooled']:
        selected=keys if key=='pooled' else [key]
        for h in HORIZONS:
            for case in CASES:
                for measure in MEASURES:
                    r=dict(known_pair=key,horizon=h,case_id=case,metric=measure,seeds=len(seeds))
                    values={}
                    for c in ('A','B'):
                        values[c]=[mean(by[k,s,case,h][measure+'_'+c] for k in selected) for s in seeds]
                        base=[mean(by[k,s,'neutral',h][measure+'_'+c] for k in selected) for s in seeds]
                        r.update({f'{c}_{k}':v for k,v in estimate_ci(values[c]).items()})
                        r.update({f'difference_{c}_{k}':v for k,v in estimate_ci([a-b for a,b in zip(values[c],base)],36).items()})
                    gap=[b-a for a,b in zip(values['A'],values['B'])]
                    basegap=[mean(by[k,s,'neutral',h][measure+'_B']-by[k,s,'neutral',h][measure+'_A'] for k in selected) for s in seeds]
                    r.update({f'B_minus_A_{k}':v for k,v in estimate_ci(gap).items()})
                    r.update({f'gap_change_{k}':v for k,v in estimate_ci([a-b for a,b in zip(gap,basegap)]).items()})
                    output.append(r)
    contrasts=[]
    for key in keys+['pooled']:
        selected=keys if key=='pooled' else [key]
        for direction in ('promote','suppress'):
            for h in HORIZONS:
                for c in ('A','B'):
                    values=[mean(by[k,s,'actual_'+direction,h]['bridge_count_'+c]-by[k,s,'permuted_'+direction,h]['bridge_count_'+c] for k in selected) for s in seeds]
                    contrasts.append(dict(known_pair=key,direction=direction,horizon=h,condition=c,**estimate_ci(values,36)))
    return output,contrasts

def run(output,start=1101,end=2100,workers=2):
    if not (start>=1 and start<=end and workers in (1,2)): raise ValueError('range/workers')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    sources=list((PROJECT_ROOT/'poc').rglob('*.py'))+list((PROJECT_ROOT/'poc/config').glob('*.json'))+[PROJECT_ROOT/'docs/NEXT_MODEL_ONLY_EXPERIMENTS.md',PROJECT_ROOT/'validation/verify_early_recontact.py',PROJECT_ROOT/'validation/verify_event_formats.py',PROJECT_ROOT/'validation/early-recontact-seed-inventory.json']
    hashes={p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in sources}
    _write_json(out/'design.json',dict(version=VERSION,start=start,end=end,cases=CASES,horizons=HORIZONS,primary_horizon=24,
        weight_base=2,normalizer=3,topic_link=False,event_settings=DESIGN.settings,source_sha256=hashes))
    rows=[]; started=datetime.now(timezone.utc).isoformat()
    for pair in PAIRS:
        key='-'.join(pair); folder=out/key; folder.mkdir()
        with gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as tf,(folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf,ProcessPoolExecutor(max_workers=workers) as pool:
            tasks=iter((pair,s) for s in range(start,end+1)); pending=deque(); writer=None; done=0
            def submit():
                item=next(tasks,None)
                if item is not None: pending.append(pool.submit(one,item))
            for _ in range(workers*2): submit()
            while pending:
                records,newrows=pending.popleft().result(); submit()
                for record in records: tf.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
                for row in newrows:
                    if writer is None: writer=csv.DictWriter(sf,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); rows.append(row)
                done+=1
                if done%50==0: print(f'{key}: {done}/{end-start+1} seeds verified',flush=True)
        assert done==end-start+1
        _write_json(folder/'complete.json',dict(records=done*5,output_sha256={p.name:_sha256(p) for p in folder.iterdir()}))
    summary,contrasts=summarize(rows)
    for name,data in [('summary',summary),('contrasts',contrasts)]:
        _write_json(out/(name+'.json'),data); _write_csv(out/(name+'.csv'),data)
    assert all(_sha256(PROJECT_ROOT/k)==v for k,v in hashes.items())
    _write_json(out/'manifest.json',dict(status='complete',started=started,ended=datetime.now(timezone.utc).isoformat(),condition_runs=len(rows)*2//3,
        source_sha256=hashes,output_sha256={p.relative_to(out).as_posix():_sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(f'complete: {len(rows)*2//3} conditions',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed-start',type=int,default=1101); p.add_argument('--seed-end',type=int,default=2100); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); run(a.output,a.seed_start,a.seed_end,a.workers)
