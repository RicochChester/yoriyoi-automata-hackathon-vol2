"""Read-only audit of all six pair-identity studies."""
import csv
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.pair_identity_experiment import PAIRS, combined, pair_config
from poc.ab_poc.domain import to_jsonable
from poc.ab_poc.bridge_experiment import _sha256, _write_json
from validation.verify_duration import verify

def audit(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['source_sha256'].items(): assert _sha256(ROOT/name)==digest
    for name,digest in manifest['output_sha256'].items(): assert _sha256(output/name)==digest
    baseline=ROOT/'results/duration-101-1100/traces.jsonl.gz'
    groups={}
    reference={}
    matches=0
    for pair in PAIRS:
        key='-'.join(pair)
        folder=output/key
        verify(folder,write_report=False,compare_previous=False)
        with (folder/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            groups[key]=[{k:(int(v) if k in ('seed','turns') else float(v)) for k,v in r.items()
                         if k in ('seed','turns') or (v and v.replace('.','',1).replace('-','',1).isdigit())}
                         for r in csv.DictReader(f)]
        with gzip.open(folder/'traces.jsonl.gz','rt',encoding='utf-8') as traces:
            old=gzip.open(baseline,'rt',encoding='utf-8') if pair==('akane','midori') else None
            try:
                for line in traces:
                    r=json.loads(line)
                    h=r['runs']['A']['rules']['turns']; index=(r['root_seed'],h)
                    cfg=pair_config(pair,h)
                    for i,c in enumerate(('A','B')):
                        assert r['runs'][c]['condition']==to_jsonable(cfg.conditions[i])
                    import hashlib
                    d=hashlib.sha256(json.dumps(r['runs']['A'],sort_keys=True).encode()).hexdigest()
                    if index in reference: assert reference[index]==d
                    reference[index]=d
                    if old:
                        prior=json.loads(next(old))
                        assert r==prior
                        matches+=1
                if old: assert next(old,None) is None
            finally:
                if old: old.close()
    summary,contrasts=combined(groups)
    saved=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    for row in saved: row.pop('known_pair_name')
    assert summary==saved
    assert contrasts==json.loads((output/'contrasts.json').read_text(encoding='utf-8'))
    for name in ('summary','contrasts'):
        raw=json.loads((output/(name+'.json')).read_text(encoding='utf-8'))
        with (output/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:
            assert list(csv.DictReader(f))==[{k:'' if v is None else str(v) for k,v in row.items()} for row in raw]
    report=dict(status='pass',condition_runs=manifest['condition_runs'],pair_rows=manifest['pair_rows'],
        baseline_AB_records_matched=matches,A_invariance_checks=manifest['A_invariance_checks'],
        all_states_rebuilt=True,all_24_estimates_and_60_contrasts_recomputed=True)
    _write_json(ROOT/'validation/pair-identity-audit.json',report)
    print(json.dumps(report))

if __name__=='__main__': audit(sys.argv[1] if len(sys.argv)>1 else ROOT/'results/pair-identity-101-1100')
