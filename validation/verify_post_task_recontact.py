"""Read-back audit of the hypothetical recontact experiment."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.post_task_recontact_experiment import inputs, PREVIOUS, CASE, PAIRS, SETTINGS, verify, metrics, summarize
from poc.ab_poc.bridge_experiment import _sha256, _write_json

def strings(r): return {k:'' if v is None else str(v) for k,v in r.items()}

def audit(output):
    out=Path(output); design=json.loads((out/'design.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert design['settings']==SETTINGS
    assert _sha256(PREVIOUS/'manifest.json')==design['previous_manifest_sha256']
    previous=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['output_sha256'].items(): assert _sha256(out/name)==h,name
    for name,h in design['source_sha256'].items(): assert _sha256(ROOT/name)==h,name
    rows=[]
    for pair in PAIRS:
        key='-'.join(pair)
        assert _sha256(PREVIOUS/key/'traces.jsonl.gz')==previous['output_sha256'][f'{key}/traces.jsonl.gz']
        with gzip.open(out/key/'traces.jsonl.gz','rt',encoding='utf-8') as f,(out/key/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as sf:
            reader=csv.DictReader(sf)
            for case,donor,_ in inputs(key,design['start'],design['end']):
                if case!=CASE: continue
                for theta in SETTINGS['theta_values']:
                    rec=json.loads(next(f)); assert rec['theta']==theta and rec['result']['root_seed']==donor['root_seed']
                    verify(rec['result'],donor,theta)
                    row=metrics(rec['result'],theta); assert next(reader)==strings(row); rows.append(row)
            assert next(f,None) is None and next(reader,None) is None
        print(f'audit {key}: passed',flush=True)
    expected=summarize(rows)
    assert expected==json.loads((out/'summary.json').read_text(encoding='utf-8'))
    with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as f: assert list(csv.DictReader(f))==[strings(r) for r in expected]
    assert len(rows)*2==manifest['condition_runs']==36*(design['end']-design['start']+1)
    report=dict(status='pass',condition_runs=len(rows)*2,summary_rows=len(expected),prefix_turns_matched=48,
        selection_reaction_information_and_relationships_checked=True,manifest_sha256=_sha256(out/'manifest.json'))
    _write_json(ROOT/'validation'/f'{out.name}-audit.json',report); print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); audit(p.parse_args().output)
