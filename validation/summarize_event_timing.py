"""Additional descriptive timing and paired secondary summaries."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.ab_poc.event_format_experiment import CASES,PAIRS,positions,MEASURES,estimate_ci
from poc.ab_poc.bridge_experiment import _sha256,_write_csv,_write_json
from statistics import mean

def main():
    source=ROOT/'results/event-formats-101-1100'
    out=ROOT/'results/event-formats-analysis-101-1100'
    out.mkdir(exist_ok=False)
    lookup={}; timing=[]
    for p in PAIRS:
        key='-'.join(p); folder=source/key
        with (folder/'seed_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f): lookup[(key,int(r['seed']),r['case_id'])]=r
        buckets=defaultdict(Counter)
        with (folder/'pairs.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if not r['case_id'].startswith('task_') or r['pair_type']!='bridge_edge': continue
                case=r['case_id']; active=positions(case)
                if not r['first_acquaintance_turn']: continue
                t=int(r['first_acquaintance_turn'])
                category='during' if t in active else ('before_first' if t<active[0] else ('after_last' if t>active[-1] else 'between'))
                buckets[(case,r['condition'])][category]+=1
        for (case,c),counts in buckets.items():
            timing.append(dict(known_pair=key,case_id=case,condition=c,
                **{k:counts[k]/1000 for k in ('before_first','during','between','after_last')}))
    secondary=[]
    for case in CASES:
        for metric in MEASURES[2:]:
            for c in ('A','B'):
                values=[mean(float(lookup[('-'.join(p),s,case)][f'{metric}_{c}'])-
                             float(lookup[('-'.join(p),s,'free')][f'{metric}_{c}']) for p in PAIRS)
                        for s in range(101,1101)]
                e=estimate_ci(values)
                secondary.append(dict(case_id=case,condition=c,metric=metric,**{k:v for k,v in e.items() if not k.startswith('simultaneous')}))
    _write_csv(out/'task_timing.csv',timing); _write_csv(out/'secondary_paired.csv',secondary)
    _write_json(out/'secondary_paired.json',secondary)
    _write_json(out/'manifest.json',dict(status='complete',purpose='descriptive postprocessing, no additional simulations',
        input_manifest_sha256=_sha256(source/'manifest.json'),
        output_sha256={p.name:_sha256(p) for p in out.iterdir() if p.is_file()}))
    print(out)

from collections import Counter
if __name__=='__main__': main()
