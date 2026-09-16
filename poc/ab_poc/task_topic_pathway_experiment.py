"""Controlled topic and history interventions; no new relationship scoring."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

from .task_information_experiment import InformationProvider, PROJECT_ROOT, clone_rng
from .bridge_experiment import _sha256, _write_json, _write_csv
from .domain import canonical_pair, ResponseKind, Stance
from .engine import run_ab
from .event_format_experiment import extract_event, estimate_ci, MEASURES
from .event_format_48_experiment import DESIGN
from .mechanism_experiment import compact_result
from .pair_identity_experiment import PAIRS, pair_config
from .rule_providers import RuleResponseProvider

PREVIOUS=PROJECT_ROOT/'results/task-information-101-1100'
TOPICS={'original':['地域','展示','ものづくり','食'],
        'rotated':['ものづくり','食','地域','展示'],
        'outside':['天文','古典','地質','法律']}
PROFILES={'original':('original',False,False), 'rotated':('rotated',False,False),
          'outside':('outside',False,False), 'freeze_T':('original',True,False),
          'freeze_R':('original',False,True), 'freeze_TR':('original',True,True)}

def profiles(case):
    return list(PROFILES) if case=='task_strong' else ['original','rotated','outside']

def reaction_label(points,rules):
    return ResponseKind.POSITIVE.value if points>=rules.positive_threshold else ResponseKind.NEUTRAL.value if points>=rules.neutral_threshold else ResponseKind.MISALIGNED.value

class DiagnosticResponder(RuleResponseProvider):
    def __init__(self,owner): self.owner=owner
    def reaction_points(self,context,target,topic,pair_events):
        o=self.owner; pair=canonical_pair(context.actor.id,target.id)
        history=tuple(e for e in o.history[:context.turn-1] if canonical_pair(e.actor,e.target)==pair) if o.freeze_response_history else pair_events
        entry=o.log[-1]
        clone=replace(context,rng=clone_rng(context.rng))
        entry['jitter']=clone_rng(context.rng).choice(context.rules.reaction.jitter_values)
        natural=super().reaction_points(clone,target,entry['natural_topic'],history)
        points=super().reaction_points(context,target,topic,history)
        assert clone.rng.getstate()==context.rng.getstate()
        entry.update(reaction_points=points,natural_reaction_points=natural,
                     natural_response=reaction_label(natural,context.rules.reaction),
                     reaction_history_prior_positive=any(e.response is ResponseKind.POSITIVE for e in history))
        return points

class TopicProvider(InformationProvider):
    name='task-topic-pathway'; version='task-topic-pathway-v1'
    def __init__(self,donor,seed,case,profile):
        topic,self.freeze_topic_history,self.freeze_response_history=PROFILES[profile]
        self.topics=TOPICS[topic]
        super().__init__(donor,seed,case,True)
        self.responder=DiagnosticResponder(self)

def simulate(donor,case,profile):
    pair=tuple(donor['runs']['B']['condition']['online_known_pairs'][0]); seed=donor['root_seed']
    providers=iter(TopicProvider(donor['runs'][c],seed,case,profile) for c in ('A','B'))
    full=run_ab(seed=seed,config=pair_config(pair,48),participant_provider_factory=lambda:next(providers))
    raw=compact_result(full)
    for c,r in full['runs'].items():
        raw['runs'][c].update(snapshots=r['snapshots'],turns=[{k:t[k] for k in ('turn','actor','target','topic','approach','response')} for t in r['turns']],
            event_log=r['participant_audit']['turns'],information=r['participant_audit'])
    return raw

def verify(raw,donor,previous,case,profile):
    from validation.verify_event_formats import verify_run
    verify_run(raw,case,DESIGN)
    topic,freeze_t,freeze_r=PROFILES[profile]
    for c,r in raw['runs'].items():
        d=donor['runs'][c]; old=previous['runs'][c]
        assert all(r[k]==d[k] for k in ('participants','rules','condition'))
        assert [(t['actor'],t['target']) for t in r['turns']]==[(t['actor'],t['target']) for t in d['turns']]
        assert r['information']['initial']==old['information']['initial'] and r['information']['final']==old['information']['final']
        people={p['id']:p for p in r['participants']}; rules=r['rules']['reaction']
        known={canonical_pair(*p) for p in r['condition']['online_known_pairs']}
        for i,(t,e,oe) in enumerate(zip(r['turns'],r['event_log'],old['event_log'])):
            for k in ('information_before','information_after','transferred_token','requested_token','information_available'):
                assert e.get(k)==oe.get(k)
            requested=e.get('requested_token')
            assert t['topic']==(TOPICS[topic][requested] if requested is not None else e['natural_topic'])
            if freeze_t: assert e['natural_topic']==d['turns'][i]['topic']
            pair=canonical_pair(t['actor'],t['target']); history=(d if freeze_r else r)['turns'][:i]
            positive=any(canonical_pair(x['actor'],x['target'])==pair and x['response']==ResponseKind.POSITIVE for x in history)
            assert positive==e['reaction_history_prior_positive']
            a,b=people[t['actor']],people[t['target']]; shared=bool(set(a['interests'])&set(b['interests'])); online=pair in known
            # Independent arithmetic checks the unchanged reaction equation.
            def score(subject):
                return (rules['base']+rules['online_known_bonus']*online+rules['topic_match_bonus']*(subject in b['interests'])+
                    rules['prior_positive_bonus']*positive+rules['proactive_actor_bonus']*(a['stance']==Stance.PROACTIVE)+
                    rules['cautious_actor_penalty']*(a['stance']==Stance.CAUTIOUS and not(online or shared))+
                    rules['cautious_target_penalty']*(b['stance']==Stance.CAUTIOUS and not(online or shared or positive))+e['jitter'])
            def label(p): return ResponseKind.POSITIVE.value if p>=rules['positive_threshold'] else ResponseKind.NEUTRAL.value if p>=rules['neutral_threshold'] else ResponseKind.MISALIGNED.value
            assert e['jitter'] in rules['jitter_values']
            assert score(t['topic'])==e['reaction_points'] and label(score(t['topic']))==t['response']
            assert score(e['natural_topic'])==e['natural_reaction_points'] and label(score(e['natural_topic']))==e['natural_response']
            if freeze_t and freeze_r and requested is None:
                assert (t['topic'],t['response'])==(d['turns'][i]['topic'],d['turns'][i]['response'])
        if profile=='original': assert r['turns']==old['turns'] and r['snapshots']==old['snapshots']

EXTRA=('questions','question_positive_difference','other_positive_difference','immediate_positive_difference',
       'immediate_positive_losses','immediate_positive_gains','changed_topics','task_progress')

def metrics(raw,donor,case,profile):
    pairs,row=extract_event(raw,case,DESIGN); row['profile']=profile
    for p in pairs: p['profile']=profile
    for c,r in raw['runs'].items():
        logs=r['event_log']; turns=r['turns']; dt=donor['runs'][c]['turns']
        question=[e.get('requested_token') is not None for e in logs]
        diff=[int(t['response']==ResponseKind.POSITIVE)-int(d['response']==ResponseKind.POSITIVE) for t,d in zip(turns,dt)]
        immediate=[int(t['response']==ResponseKind.POSITIVE)-int(e['natural_response']==ResponseKind.POSITIVE) for t,e,q in zip(turns,logs,question) if q]
        values=dict(questions=sum(question),question_positive_difference=sum(v for v,q in zip(diff,question) if q),
            other_positive_difference=sum(v for v,q in zip(diff,question) if not q),
            immediate_positive_difference=sum(immediate),immediate_positive_losses=sum(v<0 for v in immediate),
            immediate_positive_gains=sum(v>0 for v in immediate),changed_topics=sum(t['topic']!=d['topic'] for t,d in zip(turns,dt)),
            task_progress=r['information']['final']['task_progress'])
        row.update({f'{k}_{c}':v for k,v in values.items()})
    return pairs,row

def one(item):
    case,donor,previous=item; result=[]
    for profile in profiles(case):
        raw=simulate(donor,case,profile); verify(raw,donor,previous,case,profile)
        pairs,row=metrics(raw,donor,case,profile)
        result.append((dict(case_id=case,profile=profile,result=raw),pairs,row))
    return result

def inputs(key,start,end):
    # Familiar and novel saved adjacently; familiar is the exact event donor.
    with gzip.open(PREVIOUS/key/'traces.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            familiar=json.loads(line); novel=json.loads(next(f))
            seed=familiar['result']['root_seed']; case=familiar['case_id']
            assert familiar['profile']=='familiar' and novel['profile']=='novel'
            assert novel['case_id']==case and novel['result']['root_seed']==seed
            if seed>end: break
            if seed>=start: yield case,familiar['result'],novel['result']

def summary(rows,donors):
    seeds=sorted({r['seed'] for r in rows}); keys=['-'.join(p) for p in PAIRS]
    by={(r['known_pair'],r['seed'],r['case_id'],r['profile']):r for r in rows}
    base={(r['known_pair'],r['seed'],r['case_id']):r for r in donors}
    cells=[(c,p) for c in ('task_weak','task_medium','task_strong') for p in profiles(c)]
    assert len(by)==len(rows)==len(seeds)*6*len(cells)
    output=[]
    for key in keys+['pooled']:
        selected=keys if key=='pooled' else [key]
        for case,profile in cells:
            for metric in MEASURES+EXTRA:
                r=dict(known_pair=key,case_id=case,profile=profile,metric=metric,seeds=len(seeds))
                for c in ('A','B'):
                    values=[mean(by[k,s,case,profile][metric+'_'+c] for k in selected) for s in seeds]
                    r.update({f'{c}_{k}':v for k,v in estimate_ci(values,144).items()})
                    if metric in MEASURES:
                        diffs=[mean(by[k,s,case,profile][metric+'_'+c]-base[k,s,case][metric+'_'+c] for k in selected) for s in seeds]
                        r.update({f'difference_{c}_{k}':v for k,v in estimate_ci(diffs,144).items()})
                gaps=[mean(by[k,s,case,profile][metric+'_B']-by[k,s,case,profile][metric+'_A'] for k in selected) for s in seeds]
                r.update({f'B_minus_A_{k}':v for k,v in estimate_ci(gaps,144).items()}); output.append(r)
    return output

def run(output,start=101,end=1100,workers=2):
    if not 101<=start<=end<=1100 or workers not in (1,2): raise ValueError('range/workers')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    sources=list((PROJECT_ROOT/'poc').rglob('*.py'))+list((PROJECT_ROOT/'poc/config').glob('*.json'))+[PROJECT_ROOT/'docs/TASK_TOPIC_PATHWAY_PLAN.md',PROJECT_ROOT/'validation/verify_event_formats.py',PROJECT_ROOT/'validation/verify_task_topic_pathway.py']
    hashes={p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in sources}
    previous_manifest=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    _write_json(out/'design.json',dict(start=start,end=end,topics=TOPICS,profiles=PROFILES,source_sha256=hashes,
        previous_manifest_sha256=_sha256(PREVIOUS/'manifest.json'),event_settings=DESIGN.settings))
    rows=[]; donors=[]; begun=datetime.now(timezone.utc).isoformat()
    for pair in PAIRS:
        key='-'.join(pair); folder=out/key; folder.mkdir()
        assert _sha256(PREVIOUS/key/'traces.jsonl.gz')==previous_manifest['output_sha256'][f'{key}/traces.jsonl.gz']
        with gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as trace, (folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf, ProcessPoolExecutor(max_workers=workers) as pool:
            tasks=iter(inputs(key,start,end)); pending=deque(); writer=None; done=0
            def submit():
                item=next(tasks,None)
                if item is None: return
                donors.append(extract_event(item[1],item[0],DESIGN)[1])
                pending.append(pool.submit(one,item))
            for _ in range(workers*2): submit()
            while pending:
                outputs=pending.popleft().result(); submit()
                for record,pairs,row in outputs:
                    if writer is None: writer=csv.DictWriter(sf,fieldnames=list(row)); writer.writeheader()
                    trace.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
                    writer.writerow(row); rows.append(row)
                done+=1
                if done%150==0: print(f'{key}: {done//3}/{end-start+1} seeds audited',flush=True)
        assert done==3*(end-start+1)
        _write_json(folder/'complete.json',dict(records=12*(end-start+1),output_sha256={p.name:_sha256(p) for p in folder.iterdir()}))
    _write_csv(out/'baseline_metrics.csv',donors)
    summaries=summary(rows,donors); _write_json(out/'summary.json',summaries)
    # Diagnostic rows have no baseline difference fields; use a union schema.
    fields=list(dict.fromkeys(k for r in summaries for k in r))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summaries)
    assert all(_sha256(PROJECT_ROOT/k)==v for k,v in hashes.items())
    manifest=dict(status='complete',started=begun,ended=datetime.now(timezone.utc).isoformat(),condition_runs=len(rows)*2,
        original_condition_runs_reproduced=36*(end-start+1),source_sha256=hashes,
        output_sha256={p.relative_to(out).as_posix():_sha256(p) for p in out.rglob('*') if p.is_file()})
    _write_json(out/'manifest.json',manifest)
    print(f'complete: {len(rows)*2} condition runs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed-start',type=int,default=101); p.add_argument('--seed-end',type=int,default=1100); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); run(a.output,a.seed_start,a.seed_end,a.workers)
