"""Read-back audit of every saved trace and metric, without rerunning generation."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.task_topic_pathway_experiment import inputs, profiles, verify, metrics, summary, PAIRS, PREVIOUS, DESIGN
from poc.ab_poc.event_format_experiment import extract_event
from poc.ab_poc.bridge_experiment import _sha256, _write_json

def strings(row): return {k:'' if v is None else str(v) for k,v in row.items()}

def audit(output):
    out=Path(output); design=json.loads((out/'design.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert _sha256(PREVIOUS/'manifest.json')==design['previous_manifest_sha256']
    source_manifest=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['output_sha256'].items(): assert _sha256(out/name)==h,name
    for name,h in design['source_sha256'].items(): assert _sha256(ROOT/name)==h,name
    rows=[]; donors=[]
    for pair in PAIRS:
        key='-'.join(pair)
        assert _sha256(PREVIOUS/key/'traces.jsonl.gz')==source_manifest['output_sha256'][f'{key}/traces.jsonl.gz']
        with gzip.open(out/key/'traces.jsonl.gz','rt',encoding='utf-8') as f, (out/key/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as sf:
            reader=csv.DictReader(sf)
            for case,donor,previous in inputs(key,design['start'],design['end']):
                donors.append(extract_event(donor,case,DESIGN)[1])
                for profile in profiles(case):
                    rec=json.loads(next(f)); assert rec['case_id']==case and rec['profile']==profile
                    assert rec['result']['root_seed']==donor['root_seed']
                    verify(rec['result'],donor,previous,case,profile)
                    _,row=metrics(rec['result'],donor,case,profile)
                    assert next(reader)==strings(row); rows.append(row)
            assert next(f,None) is None and next(reader,None) is None
        print(f'audit: {key} passed',flush=True)
    with (out/'baseline_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        assert list(csv.DictReader(f))==[strings(r) for r in donors]
    expected=summary(rows,donors)
    assert expected==json.loads((out/'summary.json').read_text(encoding='utf-8'))
    fields=list(dict.fromkeys(k for r in expected for k in r))
    with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
        assert list(csv.DictReader(f))==[strings({k:r.get(k) for k in fields}) for r in expected]
    assert len(rows)*2==manifest['condition_runs']==144*(design['end']-design['start']+1)
    report=dict(status='pass',condition_runs=len(rows)*2,original_reproduced=36*(design['end']-design['start']+1),
        summary_rows=len(expected),full_contacts_information_reaction_equations_relationships_and_tables_checked=True,
        manifest_sha256=_sha256(out/'manifest.json'))
    _write_json(ROOT/'validation'/f'{out.name}-audit.json',report); print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); audit(p.parse_args().output)
