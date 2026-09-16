"""Fixed-budget event interventions: no direct relationship bonuses."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, NormalDist
from .bridge_experiment import PROJECT_ROOT, analyze_ab, _sha256, _write_csv, _write_json
from .contracts import Initiation
from .domain import canonical_pair, to_jsonable
from .engine import run_ab
from .mechanism_experiment import compact_result, estimate
from .pair_identity_experiment import PAIRS, pair_config
from .rule_providers import RuleParticipantProvider, RuleTalkInitiator, RuleResponseProvider, _shared_interests

SETTINGS=json.loads((PROJECT_ROOT/'poc/config/event_formats_v1.json').read_text(encoding='utf-8'))
CASES=('free',)+tuple(f'{kind}_{strength}' for kind in ('shuffle','task','host','topic') for strength in ('weak','strong'))
VERSION=SETTINGS['version']
MEASURES=('bridge_count','bridge_rate','unformed_turns','known_share','bridge_share','all_formed')

def rng_for(seed,label,version=VERSION):
    key=hashlib.sha256(f'{version}:{seed}:{label}'.encode()).digest()
    return random.Random(int.from_bytes(key,'big'))

def positions(case):
    if case not in CASES: raise ValueError('unknown event case')
    return () if case=='free' else tuple(SETTINGS[case.rsplit('_',1)[1]+'_turns'])

def contact_plan(seed,case,active_turns=None):
    if not case.startswith(('shuffle_','task_')): return {}
    pairs=list(PAIRS); rng_for(seed,'balanced-pairs').shuffle(pairs)
    directions=[rng_for(seed,f'direction:{i}').randrange(2) for i in range(6)]
    result={}
    for i,t in enumerate(positions(case) if active_turns is None else active_turns):
        p=pairs[i%6]; d=directions[i%6]^((i//6)%2)
        result[t]=(p[d],p[1-d])
    return result

class EventDesign:
    """Serializable experiment settings; v1 remains the default."""
    def __init__(self,settings):
        self.settings=settings
        if settings.get('rng_version',VERSION)!=VERSION:
            raise ValueError('unsupported intervention random stream version')
        self.strengths=tuple(s for s in ('weak','medium','strong') if s+'_turns' in settings)
        self.cases=('free',)+tuple(f'{k}_{s}' for k in ('shuffle','task','host','topic') for s in self.strengths)
        for case in self.cases:
            active=self.positions(case)
            if tuple(sorted(set(active)))!=active or any(t<1 or t>settings['turns'] for t in active):
                raise ValueError('invalid intervention schedule')
            if case!='free' and len(active)%6: raise ValueError('interventions must cover all six pairs')
        for strength,blocks in settings.get('task_blocks',{}).items():
            if strength not in self.strengths or any(end-start!=5 for start,end in blocks):
                raise ValueError('each task block must contain six consecutive turns')
            if len(self.positions('task_'+strength))!=len(settings[strength+'_turns']):
                raise ValueError('task and shuffle contact budgets differ')
    def positions(self,case):
        if case not in self.cases: raise ValueError('unknown event case')
        if case=='free': return ()
        strength=case.rsplit('_',1)[1]
        if case.startswith('task_') and 'task_blocks' in self.settings:
            return tuple(t for a,b in self.settings['task_blocks'][strength] for t in range(a,b+1))
        return tuple(self.settings[strength+'_turns'])

DEFAULT_DESIGN=EventDesign(SETTINGS)

class EventInitiator(RuleTalkInitiator):
    def __init__(self,owner): self.owner=owner
    def shared_interests(self,actor,target):
        shared=super().shared_interests(actor,target)
        return shared or ((self.owner.common_topic,) if self.owner.topic_active else ())
    def selection_weight(self,context,candidate):
        w=super().selection_weight(context,candidate)
        if self.owner.introduced and canonical_pair(context.actor.id,candidate.id)==self.owner.introduced:
            multiplier=context.rules.selection.repeated_target_multiplier if context.last_target==candidate.id else 1
            w+=self.owner.settings['host_selection_bonus']*multiplier
        return w
    def initiate(self,context):
        if context.turn not in self.owner.plan: return super().initiate(context)
        actor,target_id=self.owner.plan[context.turn]
        if actor!=context.actor.id: raise ValueError('forced actor mismatch')
        context.rng.random()
        target=next(p for p in context.candidates if p.id==target_id)
        pair=canonical_pair(actor,target_id)
        events=tuple(e for e in context.events if canonical_pair(e.actor,e.target)==pair)
        approach=self._approach(context.actor.stance)
        if self.owner.case.startswith('task_'):
            role=self.owner.settings['task_roles'][list(self.owner.plan).index(context.turn)%len(self.owner.settings['task_roles'])]
            approach=f'共同課題を進めるために、あなたと{role}。'
        return Initiation(target_id,self._choose_topic(context,target,events),approach)

class EventResponder(RuleResponseProvider):
    def __init__(self,owner): self.owner=owner
    def shared_interests(self,actor,target):
        return super().shared_interests(actor,target) or ((self.owner.common_topic,) if self.owner.topic_active else ())

class EventProvider(RuleParticipantProvider):
    name='event-rule-participant'
    version=VERSION
    def __init__(self,seed,case,common_topic,design=DEFAULT_DESIGN):
        self.seed,self.case,self.common_topic=seed,case,common_topic
        self.settings=design.settings; self.version=self.settings['version']
        self.active_turns=design.positions(case); self.plan=contact_plan(seed,case,self.active_turns)
        self.topic_active=False; self.introduced=None; self.log=[]
        super().__init__(initiator=EventInitiator(self),responder=EventResponder(self))
    def select_event_actor(self,turn,planned_actor,participants,events):
        self.introduced=None
        actor=planned_actor; entry=dict(turn=turn,active=turn in self.active_turns,planned_actor=planned_actor)
        if turn in self.plan:
            actor,target=self.plan[turn]
            entry.update(forced_pair=list(canonical_pair(actor,target)),forced_target=target)
            if self.case.startswith('task_'):
                entry['task_role']=self.settings['task_roles'][list(self.plan).index(turn)%len(self.settings['task_roles'])]
        elif self.case.startswith('host_') and turn in self.active_turns:
            counts=Counter(canonical_pair(e.actor,e.target) for e in events)
            low=min(counts[p] for p in PAIRS)
            tied=sorted(p for p in PAIRS if counts[p]==low)
            pair=rng_for(self.seed,f'host-tie:{turn}').choice(tied)
            actor=pair[rng_for(self.seed,f'host-direction:{turn}').randrange(2)]
            self.introduced=pair
            entry.update(introduced_pair=list(pair),introduced_prior_count=low,host_bonus=self.settings['host_selection_bonus'])
        entry['actual_actor']=actor
        self.log.append(entry)
        return actor
    def propose(self,context):
        self.topic_active=self.case.startswith('topic_') and context.turn in self.active_turns
        try:
            proposal=super().propose(context)
            self.log[-1].update(actual_target=proposal.target,shared_override=self.topic_active)
            return proposal
        finally: self.topic_active=False
    def audit_snapshot(self): return dict(turns=self.log)

def run_event(pair,seed,case,design=DEFAULT_DESIGN):
    design.positions(case)
    cfg=pair_config(pair,design.settings['turns'])
    if case=='free': full=run_ab(seed=seed,config=cfg)
    else:
        full=run_ab(seed=seed,config=cfg,participant_provider_factory=lambda:EventProvider(seed,case,cfg.rules.common_topic,design))
    raw=compact_result(full)
    for c in ('A','B'):
        run=full['runs'][c]
        raw['runs'][c]['snapshots']=run['snapshots']
        raw['runs'][c]['turns']=[{k:t[k] for k in ('turn','actor','target','topic','approach','response')} for t in run['turns']]
        raw['runs'][c]['event_log']=run.get('participant_audit',{}).get('turns',[])
    return raw

def extract_event(raw,case,design=DEFAULT_DESIGN):
    pairs,old=analyze_ab(raw,require_same_actor_order=False)
    known=canonical_pair(*raw['runs']['B']['condition']['online_known_pairs'][0])
    row=dict(seed=raw['root_seed'],known_pair='-'.join(known),case_id=case)
    active=set(design.positions(case)); horizon=design.settings['turns']
    for c,run in raw['runs'].items():
        if len(run['turns'])!=horizon: raise ValueError('unexpected horizon')
        first={}
        for s in run['snapshots']:
            if s['phase']!='turn_end': continue
            for p in s['pairs']:
                key=canonical_pair(p['left_id'],p['right_id'])
                if p['onsite_relationship']!='未形成' and key not in first: first[key]=s['turn']
        bridge=[p for p in pairs if p['condition']==c and p['pair_type']=='bridge_edge']
        bridge_keys={canonical_pair(p['left_id'],p['right_id']) for p in bridge}
        for p in pairs:
            if p['condition']!=c: continue
            key=canonical_pair(p['left_id'],p['right_id'])
            p.update(first_acquaintance_turn=first.get(key),acquaintance_censored=key not in first,
                     unformed_turns=first.get(key,horizon+1)-1,case_id=case,known_pair='-'.join(known))
        counts=Counter(canonical_pair(t['actor'],t['target']) for t in run['turns'])
        row.update({f'bridge_count_{c}':old[f'bridge_edge_count_{c}'],
                    f'bridge_rate_{c}':old[f'bridge_edge_rate_{c}'],
                    f'unformed_turns_{c}':sum(first.get(p,horizon+1)-1 for p in bridge_keys)/4,
                    f'known_share_{c}':counts[known]/horizon,
                    f'bridge_share_{c}':sum(counts[p] for p in bridge_keys)/horizon,
                    f'all_formed_{c}':int(all(p in first for p in bridge_keys))})
        before=Counter(); low=forced=intro=realized=new_talk=0
        log={e['turn']:e for e in run['event_log']}
        for t in run['turns']:
            p=canonical_pair(t['actor'],t['target']); e=log.get(t['turn'],{})
            if 'forced_pair' in e:
                forced+=1; low+=before[p]<run['rules']['relationship']['acquaintance_conversations']
            if 'introduced_pair' in e:
                intro+=1; realized+=p==tuple(e['introduced_pair'])
            if t['turn'] in active and p in bridge_keys and before[p]==0: new_talk+=1
            before[p]+=1
        for person in raw['participants']:
            row[f'actor_turns_{person["id"]}_{c}']=sum(t['actor']==person['id'] for t in run['turns'])
        formed_active=sum(t in active for p,t in first.items() if p in bridge_keys)
        row.update({f'forced_contacts_{c}':forced,f'forced_undercontacted_fraction_{c}':low/forced if forced else 0,
            f'introductions_{c}':intro,f'introduction_realization_{c}':realized/intro if intro else 0,
            f'intervention_new_bridge_contacts_{c}':new_talk,f'intervention_bridge_formations_{c}':formed_active,
            f'nonintervention_bridge_formations_{c}':sum(p in bridge_keys and t not in active for p,t in first.items())})
        if 'task_blocks' in design.settings:
            stages={'before':0,'during':0,'between':0,'after':0}
            if case.startswith('task_'):
                for p,t in first.items():
                    if p in bridge_keys:
                        stages['during' if t in active else 'before' if t<min(active) else 'after' if t>max(active) else 'between']+=1
            row.update({f'task_formed_{k}_{c}':v for k,v in stages.items()})
    row['B_minus_A']=row['bridge_count_B']-row['bridge_count_A']
    return pairs,row

def estimate_ci(values,family=1):
    e=estimate(values); se=e['standard_error']; z=NormalDist().inv_cdf(1-.05/(2*family))
    return dict(mean=e['mean'],ci_low=e['ci_low'],ci_high=e['ci_high'],
                simultaneous_low=e['mean']-z*se if se is not None else None,
                simultaneous_high=e['mean']+z*se if se is not None else None)

def summarize_event(rows,design=DEFAULT_DESIGN):
    cases=design.cases
    by={(r['known_pair'],r['seed'],r['case_id']):r for r in rows}
    seeds=sorted({r['seed'] for r in rows})
    expected={('-'.join(p),s,c) for p in PAIRS for s in seeds for c in cases}
    if not seeds or len(by)!=len(rows) or set(by)!=expected: raise ValueError('missing or duplicate event cases')
    summary=[]; pooled=[]
    def make(key,case,selected,baseline):
        r=dict(known_pair=key,case_id=case,seeds=len(seeds))
        for m in MEASURES:
            for c in ('A','B'):
                for k,v in estimate_ci([x[f'{m}_{c}'] for x in selected]).items():
                    if not k.startswith('simultaneous'): r[f'{m}_{c}_{k}']=v
        contrasts=dict(B_minus_A=[x['B_minus_A'] for x in selected],
            B_vs_free=[x['bridge_count_B']-b['bridge_count_B'] for x,b in zip(selected,baseline)],
            A_vs_free=[x['bridge_count_A']-b['bridge_count_A'] for x,b in zip(selected,baseline)],
            island_change=[x['B_minus_A']-b['B_minus_A'] for x,b in zip(selected,baseline)])
        for name,values in contrasts.items():
            for k,v in estimate_ci(values,6*len(cases) if name=='B_minus_A' else 6*(len(cases)-1)).items(): r[f'{name}_{k}']=v
        for col in selected[0]:
            if col.startswith(('actor_turns_','forced_','introductions_','introduction_realization_','intervention_','nonintervention_','task_formed_')):
                r[col]=mean(x[col] for x in selected)
        return r
    for p in PAIRS:
        key='-'.join(p)
        for case in cases:
            summary.append(make(key,case,[by[(key,s,case)] for s in seeds],[by[(key,s,'free')] for s in seeds]))
    for case in cases:
        selected=[]; baseline=[]
        for s in seeds:
            for name,dest in ((case,selected),('free',baseline)):
                six=[by[('-'.join(p),s,name)] for p in PAIRS]
                dest.append({k:mean(x[k] for x in six) for k in six[0] if isinstance(six[0][k],(int,float))})
        pooled.append(make('six_pair_equal_weight',case,selected,baseline))
    return summary,pooled

def one_seed(task):
    p,s,design=task
    return s,{c:run_event(p,s,c,design) for c in design.cases}

def run_study(output,seed_start=101,seed_end=1100,workers=2,design=DEFAULT_DESIGN):
    if not 101<=seed_start<=seed_end<=1100 or workers not in (1,2): raise ValueError('invalid seed range or workers')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    sources=sorted((PROJECT_ROOT/'poc').rglob('*.py'))+sorted((PROJECT_ROOT/'poc/config').glob('*.json'))+[PROJECT_ROOT/'docs/EVENT_FORMAT_PLAN.md']
    if 'task_blocks' in design.settings: sources.append(PROJECT_ROOT/'docs/EVENT_FORMAT_48_PLAN.md')
    hashes={p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in sources}
    started=datetime.now(timezone.utc).isoformat()
    _write_json(out/'design.json',dict(settings=design.settings,cases=design.cases,seed_start=seed_start,seed_end=seed_end,
        configs={'-'.join(p):to_jsonable(pair_config(p,design.settings['turns'])) for p in PAIRS},source_sha256=hashes))
    rows=[]
    for p in PAIRS:
        folder=out/'-'.join(p); folder.mkdir()
        with ProcessPoolExecutor(max_workers=workers) as pool, gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as f, \
             (folder/'pairs.csv').open('w',encoding='utf-8-sig',newline='') as pf, \
             (folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf:
            tasks=iter((p,s,design) for s in range(seed_start,seed_end+1)); pending=deque()
            for _ in range(2*workers):
                task=next(tasks,None)
                if task is not None: pending.append(pool.submit(one_seed,task))
            pw=sw=None
            while pending:
                seed,results=pending.popleft().result()
                task=next(tasks,None)
                if task is not None: pending.append(pool.submit(one_seed,task))
                for case,raw in results.items():
                    f.write(json.dumps(dict(case_id=case,result=raw),ensure_ascii=False,separators=(',',':'))+'\n')
                    pairs,row=extract_event(raw,case,design); rows.append(row)
                    if pw is None:
                        pw=csv.DictWriter(pf,fieldnames=list(pairs[0])); pw.writeheader()
                        sw=csv.DictWriter(sf,fieldnames=list(row)); sw.writeheader()
                    pw.writerows(pairs); sw.writerow(row)
                if (seed-seed_start+1)%100==0: print(f'{"-".join(p)} {seed-seed_start+1}/{seed_end-seed_start+1}',flush=True)
        _write_json(folder/'complete.json',dict(output_sha256={p.name:_sha256(p) for p in folder.iterdir() if p.is_file()}))
    summary,pooled=summarize_event(rows,design)
    for name,data in (('summary',summary),('pooled',pooled)):
        _write_json(out/(name+'.json'),data); _write_csv(out/(name+'.csv'),data)
    for name,h in hashes.items():
        if _sha256(PROJECT_ROOT/name)!=h: raise ValueError('source changed')
    manifest=dict(status='complete',version=design.settings['version'],condition_runs=12*len(design.cases)*(seed_end-seed_start+1),pair_rows=72*len(design.cases)*(seed_end-seed_start+1),
        started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        output_sha256={p.relative_to(out).as_posix():_sha256(p) for p in out.rglob('*') if p.is_file()})
    _write_json(out/'manifest.json',manifest); print(f'Completed: {out.resolve()}',flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed-start',type=int,default=101); p.add_argument('--seed-end',type=int,default=1100)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--output',type=Path)
    a=p.parse_args()
    try: run_study(a.output or PROJECT_ROOT/'results'/datetime.now().strftime('event-formats-%Y%m%d-%H%M%S-%f'),a.seed_start,a.seed_end,a.workers)
    except (ValueError,OSError,KeyError,TypeError) as exc: p.exit(1,f'Experiment failed: {exc}; missing manifest means incomplete.\n')
if __name__=='__main__': main()
