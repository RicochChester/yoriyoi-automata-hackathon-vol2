"""Verify persisted information traces, CSVs, and paired estimates."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from poc.ab_poc.task_information_experiment import PAIRS, BASELINE, summarize, verify
from poc.ab_poc.event_format_experiment import extract_event
from poc.ab_poc.event_format_48_experiment import DESIGN
from poc.ab_poc.bridge_experiment import _sha256, _write_json

def strings(row): return {k:'' if v is None else str(v) for k,v in row.items()}

def audit(out,follow=False):
    out=Path(out); deadline=time.monotonic()+3600
    def wait(p):
        while follow and not p.exists() and time.monotonic()<deadline: time.sleep(2)
        assert p.exists(),str(p)
    wait(out/'design.json'); design=json.loads((out/'design.json').read_text(encoding='utf-8'))
    rows=[]; pair_count=0
    for pair in PAIRS:
        folder=out/'-'.join(pair); wait(folder/'complete.json')
        done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        for name,h in done['output_sha256'].items(): assert _sha256(folder/name)==h
        with gzip.open(BASELINE/'-'.join(pair)/'traces.jsonl.gz','rt',encoding='utf-8') as base, gzip.open(folder/'traces.jsonl.gz','rt',encoding='utf-8') as trace, (folder/'pairs.csv').open(encoding='utf-8-sig',newline='') as pf, (folder/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as sf:
            pr,sr=csv.DictReader(pf),csv.DictReader(sf); count=0
            for line in base:
                original=json.loads(line); donor=original['result']; case=original['case_id']; seed=donor['root_seed']
                if seed>design['end']: break
                if seed<design['start'] or not case.startswith('task_'): continue
                for profile in ('familiar','novel'):
                    record=json.loads(next(trace)); raw=record['result']
                    assert record['case_id']==case and record['profile']==profile and raw['root_seed']==seed
                    verify(raw,donor,case,profile=='novel')
                    pairs,row=extract_event(raw,case,DESIGN); row['profile']=profile
                    for p in pairs: p['profile']=profile
                    for c,run in raw['runs'].items():
                        logs=run['event_log']; info=run['information']['final']
                        row.update({f'task_progress_{c}':info['task_progress'],f'task_complete_{c}':int(info['complete']),
                            f'information_transfers_{c}':sum(e['transferred_token'] is not None for e in logs),
                            f'questions_{c}':sum(e.get('requested_token') is not None for e in logs),
                            f'changed_topics_{c}':sum(a['topic']!=b['topic'] for a,b in zip(run['turns'],donor['runs'][c]['turns']))})
                    assert next(sr)==strings(row); rows.append(row)
                    for p in pairs: assert next(pr)==strings(p); pair_count+=1
                count+=1
                if count%750==0: print(f'audit {"-".join(pair)}: {count//3} seeds',flush=True)
            assert count==3*(design['end']-design['start']+1)
            assert next(trace,None) is None and next(pr,None) is None and next(sr,None) is None
    wait(out/'manifest.json'); manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['pair_rows']==pair_count and manifest['condition_runs']==len(rows)*2
    for name,h in manifest['source_sha256'].items(): assert _sha256(ROOT/name)==h
    for name,h in manifest['output_sha256'].items(): assert _sha256(out/name)==h
    summary=summarize(rows)
    assert summary==json.loads((out/'summary.json').read_text(encoding='utf-8'))
    with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as f: assert list(csv.DictReader(f))==[strings(r) for r in summary]
    report=dict(status='pass',condition_runs=len(rows)*2,pair_rows=pair_count,all_contacts_and_familiar_baselines_exact=True,
        all_information_transfers_and_relationship_states_verified=True,all_csv_and_summary_recomputed=True,source_and_output_hashes_match=True)
    _write_json(ROOT/'validation'/(out.name+'-audit.json'),report); print(json.dumps(report))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--follow',action='store_true'); a=p.parse_args(); audit(a.output,a.follow)
