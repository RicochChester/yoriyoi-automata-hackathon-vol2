"""Information need changes topics, never relationship rules or contact budgets."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gzip
import json
import random
from pathlib import Path
from statistics import mean
from .bridge_experiment import PROJECT_ROOT, _sha256, _write_json, _write_csv
from .contracts import Initiation
from .domain import Event, canonical_pair, to_jsonable
from .engine import run_ab
from .event_format_experiment import EventProvider, extract_event, rng_for, MEASURES, estimate_ci
from .event_format_48_experiment import DESIGN
from .mechanism_experiment import compact_result
from .pair_identity_experiment import PAIRS, pair_config
from .rule_providers import RuleTalkInitiator, RuleParticipantProvider

SETTINGS=json.loads((PROJECT_ROOT/'poc/config/task_information_v1.json').read_text(encoding='utf-8'))
VERSION=SETTINGS['version']
BASELINE=PROJECT_ROOT/'results/event-formats-48-101-1100'

def clone_rng(rng):
    """Copy the complete PRNG state without recursive deepcopy overhead."""
    result=random.Random(0)
    result.setstate(rng.getstate())
    return result

class InformationState:
    def __init__(self,ids,seed,novel):
        owners=list(ids); rng_for(seed,'information-owners',VERSION).shuffle(owners)
        self.owners=dict(enumerate(owners))
        self.known={p:({i for i,o in self.owners.items() if o==p} if novel else set(range(4))) for p in ids}
    def snapshot(self):
        return dict(known={p:sorted(v) for p,v in self.known.items()},
            task_familiarity={p:(len(v)-1)/3 for p,v in self.known.items()},
            task_progress=sum(len(v)-1 for v in self.known.values())/12,
            complete=all(len(v)==4 for v in self.known.values()))
    def query(self,actor,target):
        missing=set(range(4))-self.known[actor]
        available=missing & self.known[target]
        token=min(available or missing) if missing else None
        return token, token is not None and token in available
    def receive(self,actor,target,token):
        if token not in self.known[target] or token in self.known[actor]: raise ValueError('invalid information transfer')
        self.known[actor].add(token)

class InformationInitiator(RuleTalkInitiator):
    def __init__(self,owner): self.owner=owner
    def initiate(self,context):
        o=self.owner; donor=o.donor['turns'][context.turn-1]
        if context.actor.id!=donor['actor']: raise ValueError('contact actor differs')
        target=next(p for p in context.candidates if p.id==donor['target'])
        pair=canonical_pair(context.actor.id,target.id)
        events=tuple(e for e in context.events if canonical_pair(e.actor,e.target)==pair)
        if getattr(o, 'freeze_topic_history', False):
            events=tuple(e for e in o.history[:context.turn-1] if canonical_pair(e.actor,e.target)==pair)
        context.rng.random()  # Same selection draw, despite replaying contacts.
        # Topic selection may change its candidate list after earlier questions.
        # Use a clone for that choice; retain the donor topic draw on the main
        # stream so response jitter is not changed by candidate-list lengths.
        topic=self._choose_topic(replace(context,rng=clone_rng(context.rng)),target,events)
        donor_events=tuple(e for e in o.history[:context.turn-1] if canonical_pair(e.actor,e.target)==pair)
        self._choose_topic(context,target,donor_events)
        entry=o.log[-1]; entry['natural_topic']=topic
        approach=donor['approach']; o.pending=None
        if context.turn in o.active_turns:
            token,available=o.info.query(context.actor.id,target.id)
            entry.update(requested_token=token,information_available=available)
            if token is not None:
                if o.topic_link:
                    topic=getattr(o, 'topics', SETTINGS['topics'])[token]
                    approach+=f' 課題の「{topic}」の手順を教えてください。'
                if available: o.pending=(context.actor.id,target.id,token)
        entry['chosen_topic']=topic
        return Initiation(target.id,topic,approach)

class InformationProvider(RuleParticipantProvider):
    name='task-information-participant'
    version=VERSION
    def __init__(self,donor,seed,case,novel,topic_link=True):
        self.donor=donor; self.active_turns=DESIGN.positions(case)
        self.info=InformationState([p['id'] for p in donor['participants']],seed,novel)
        self.initial=self.info.snapshot(); self.log=[]; self.pending=None; self.topic_link=topic_link
        self.history=[Event(**{k:t[k] for k in ('turn','actor','target','topic','approach','response')}) for t in donor['turns']]
        super().__init__(initiator=InformationInitiator(self))
    def select_event_actor(self,turn,planned_actor,participants,events):
        entry=deepcopy(self.donor['event_log'][turn-1])
        entry.update(information_before=self.info.snapshot(),transferred_token=None)
        self.log.append(entry)
        return self.donor['turns'][turn-1]['actor']
    def propose(self,context):
        proposal=super().propose(context)
        if self.pending:
            self.info.receive(*self.pending)
            self.log[-1]['transferred_token']=self.pending[2]
        self.log[-1]['information_after']=self.info.snapshot()
        return proposal
    def audit_snapshot(self): return dict(turns=self.log,initial=self.initial,final=self.info.snapshot(),owners=self.info.owners)

def simulate(raw,case,novel,topic_link=True):
    known=tuple(raw['runs']['B']['condition']['online_known_pairs'][0]); seed=raw['root_seed']
    providers=iter(InformationProvider(raw['runs'][c],seed,case,novel,topic_link) for c in ('A','B'))
    full=run_ab(seed=seed,config=pair_config(known,48),participant_provider_factory=lambda:next(providers))
    result=compact_result(full)
    for c in ('A','B'):
        run=full['runs'][c]; result['runs'][c].update(snapshots=run['snapshots'],
            turns=[{k:t[k] for k in ('turn','actor','target','topic','approach','response')} for t in run['turns']],
            event_log=run['participant_audit']['turns'],information=run['participant_audit'])
    return result

def verify(raw,donor,case,novel):
    from validation.verify_event_formats import verify_run
    verify_run(raw,case,DESIGN)  # Independently rebuild all relationship states.
    for c,run in raw['runs'].items():
        original=donor['runs'][c]
        assert run['rules']==original['rules'] and run['participants']==original['participants'] and run['condition']==original['condition']
        assert [(t['actor'],t['target']) for t in run['turns']]==[(t['actor'],t['target']) for t in original['turns']]
        if not novel:
            assert run['turns']==original['turns'] and run['snapshots']==original['snapshots']
        state=InformationState([p['id'] for p in run['participants']],raw['root_seed'],novel)
        assert state.snapshot()==run['information']['initial']
        for t,e in zip(run['turns'],run['event_log']):
            assert state.snapshot()==e['information_before']
            before={p:set(v) for p,v in state.known.items()}
            token=e['transferred_token']
            if t['turn'] in DESIGN.positions(case):
                requested,available=state.query(t['actor'],t['target'])
                assert e['requested_token']==requested and e['information_available']==available
                assert token==(requested if available else None)
                if requested is not None: assert t['topic']==SETTINGS['topics'][requested]
            else: assert token is None and t['topic']==e['natural_topic']
            if token is not None:
                assert token in before[t['target']] and token not in before[t['actor']]
                state.receive(t['actor'],t['target'],token)
            assert state.snapshot()==e['information_after']
        assert state.snapshot()==run['information']['final']

def one_record(item):
    case,donor=item; outputs=[]
    for novel in (False,True):
        raw=simulate(donor,case,novel); verify(raw,donor,case,novel)
        pairs,row=extract_event(raw,case,DESIGN)
        profile='novel' if novel else 'familiar'; row['profile']=profile
        for p in pairs: p['profile']=profile
        for c,run in raw['runs'].items():
            info=run['information']; logs=run['event_log']
            row.update({f'task_progress_{c}':info['final']['task_progress'],f'task_complete_{c}':int(info['final']['complete']),
                f'information_transfers_{c}':sum(e['transferred_token'] is not None for e in logs),
                f'questions_{c}':sum(e.get('requested_token') is not None for e in logs),
                f'changed_topics_{c}':sum(a['topic']!=b['topic'] for a,b in zip(run['turns'],donor['runs'][c]['turns']))})
        outputs.append((dict(case_id=case,profile=profile,result=raw),pairs,row))
    return outputs

def summarize(rows):
    by={(r['known_pair'],r['seed'],r['case_id'],r['profile']):r for r in rows}
    seeds=sorted({r['seed'] for r in rows}); cases=['task_'+s for s in SETTINGS['strengths']]
    keys=['-'.join(p) for p in PAIRS]
    assert len(by)==len(rows)==len(seeds)*len(cases)*len(keys)*2
    for s in seeds:
        for case in cases:
            for profile in ('familiar','novel'):
                six=[by[k,s,case,profile] for k in keys]
                by['pooled',s,case,profile]={k:mean(r[k] for r in six) for k in six[0] if isinstance(six[0][k],(float,int))}
    output=[]
    metrics=MEASURES+('task_progress','task_complete','information_transfers','questions','changed_topics')
    for key in keys+['pooled']:
        for case in cases:
            for metric in metrics:
                r=dict(known_pair=key,case_id=case,metric=metric,seeds=len(seeds))
                for c in ('A','B'):
                    for profile in ('familiar','novel'):
                        values=[by[key,s,case,profile][metric+'_'+c] for s in seeds]
                        r.update({f'{profile}_{c}_{k}':v for k,v in estimate_ci(values).items()})
                    diff=[by[key,s,case,'novel'][metric+'_'+c]-by[key,s,case,'familiar'][metric+'_'+c] for s in seeds]
                    r.update({f'novel_minus_familiar_{c}_{k}':v for k,v in estimate_ci(diff,18).items()})
                for profile in ('familiar','novel'):
                    values=[by[key,s,case,profile][metric+'_B']-by[key,s,case,profile][metric+'_A'] for s in seeds]
                    r.update({f'{profile}_B_minus_A_{k}':v for k,v in estimate_ci(values,36).items()})
                values=[(by[key,s,case,'novel'][metric+'_B']-by[key,s,case,'novel'][metric+'_A'])-(by[key,s,case,'familiar'][metric+'_B']-by[key,s,case,'familiar'][metric+'_A']) for s in seeds]
                r.update({f'island_change_{k}':v for k,v in estimate_ci(values,18).items()})
                output.append(r)
    return output

def run_study(output,start=101,end=1100,workers=2):
    if not 101<=start<=end<=1100 or workers not in (1,2): raise ValueError('invalid range/workers')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    source_files=sorted((PROJECT_ROOT/'poc').rglob('*.py'))+sorted((PROJECT_ROOT/'poc/config').glob('*.json'))+[PROJECT_ROOT/'docs/TASK_INFORMATION_PLAN.md',PROJECT_ROOT/'validation/verify_event_formats.py',PROJECT_ROOT/'validation/verify_task_information.py']
    hashes={p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in source_files}
    baseline_manifest=json.loads((BASELINE/'manifest.json').read_text(encoding='utf-8'))
    _write_json(out/'design.json',dict(settings=SETTINGS,event_settings=DESIGN.settings,start=start,end=end,
        source_sha256=hashes,baseline_manifest_sha256=_sha256(BASELINE/'manifest.json'),
        configs={'-'.join(p):to_jsonable(pair_config(p,48)) for p in PAIRS}))
    rows=[]; begun=datetime.now(timezone.utc).isoformat()
    for pair in PAIRS:
        key='-'.join(pair); folder=out/key; folder.mkdir()
        path=BASELINE/key/'traces.jsonl.gz'
        assert _sha256(path)==baseline_manifest['output_sha256'][f'{key}/traces.jsonl.gz']
        with gzip.open(path,'rt',encoding='utf-8') as source, gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as trace, (folder/'pairs.csv').open('w',encoding='utf-8-sig',newline='') as pf, (folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf, ProcessPoolExecutor(max_workers=workers) as pool:
            def inputs():
                for line in source:
                    record=json.loads(line)
                    if record['result']['root_seed']>end: break
                    if record['case_id'].startswith('task_') and start<=record['result']['root_seed']<=end:
                        yield record['case_id'],record['result']
            tasks=iter(inputs()); pending=deque(); pw=sw=None; done=0
            for _ in range(workers*2):
                task=next(tasks,None)
                if task is not None: pending.append(pool.submit(one_record,task))
            while pending:
                result=pending.popleft().result(); task=next(tasks,None)
                if task is not None: pending.append(pool.submit(one_record,task))
                for record,pairs,row in result:
                    if pw is None:
                        pw=csv.DictWriter(pf,fieldnames=list(pairs[0])); pw.writeheader()
                        sw=csv.DictWriter(sf,fieldnames=list(row)); sw.writeheader()
                    trace.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
                    pw.writerows(pairs); sw.writerow(row); rows.append(row)
                done+=1
                if done%300==0: print(f'{key}: {done//3}/{end-start+1} seeds, verified',flush=True)
        assert done==3*(end-start+1)
        _write_json(folder/'complete.json',dict(records=done*2,all_contacts_information_and_relationship_states_verified=True,output_sha256={p.name:_sha256(p) for p in folder.iterdir()}))
    summary=summarize(rows); _write_json(out/'summary.json',summary); _write_csv(out/'summary.csv',summary)
    assert all(_sha256(PROJECT_ROOT/name)==h for name,h in hashes.items())
    manifest=dict(status='complete',started=begun,ended=datetime.now(timezone.utc).isoformat(),condition_runs=len(rows)*2,pair_rows=len(rows)*12,
        familiar_condition_runs_exactly_match_baseline=len(rows),source_sha256=hashes,
        output_sha256={p.relative_to(out).as_posix():_sha256(p) for p in out.rglob('*') if p.is_file()})
    _write_json(out/'manifest.json',manifest); print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('sha256')}),flush=True)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path); p.add_argument('--seed-start',type=int,default=101); p.add_argument('--seed-end',type=int,default=1100); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    run_study(a.output or PROJECT_ROOT/'results'/datetime.now().strftime('task-information-%Y%m%d-%H%M%S'),a.seed_start,a.seed_end,a.workers)
