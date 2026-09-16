"""Known-pair identity factorial study, retaining original simulation rules."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import gzip
from itertools import combinations
import json
from pathlib import Path
from statistics import NormalDist

from .bridge_experiment import PROJECT_ROOT, _sha256, _write_csv, _write_json
from .duration_experiment import HORIZONS, duration_config, extract, summarize, verify_prefix
from .domain import canonical_pair, to_jsonable
from .engine import run_ab
from .mechanism_experiment import compact_result, estimate

PAIRS = tuple(canonical_pair(*p) for p in combinations([p.id for p in duration_config(12).participants], 2))
VERSION = 'yoriyoi.pair_identity.v1'


def pair_config(pair, horizon):
    pair = tuple(pair)
    if len(pair) != 2 or canonical_pair(*pair) not in PAIRS:
        raise ValueError('expected one of the six participant pairs')
    pair = canonical_pair(*pair)
    config = duration_config(horizon)
    a, b = config.conditions
    if pair == b.online_known_pairs[0]:
        return config
    names = {p.id: p.name for p in config.participants}
    experience = replace(b.online_experience, participants=pair, id='_'.join(pair)+'_technology_exchange')
    b = replace(b, online_known_pairs=(pair,), online_experience=experience,
                label='B：'+'・'.join(names[p] for p in pair)+'だけ直接既知')
    return replace(config, conditions=(a,b))


def interval(values, family):
    e = estimate(values)
    z = NormalDist().inv_cdf(1-0.05/(2*family))
    se = e['standard_error']
    return dict(mean=e['mean'], se=se, simultaneous_low=e['mean']-z*se if se is not None else None,
                simultaneous_high=e['mean']+z*se if se is not None else None)


def combined(groups):
    expected = {'-'.join(p) for p in PAIRS}
    if set(groups) != expected: raise ValueError('six pair conditions required')
    seeds = None
    lookup = {}
    summary = []
    for key, rows in groups.items():
        current = {r['seed'] for r in rows}
        if seeds is None: seeds = current
        if seeds != current: raise ValueError('pair conditions must share seeds')
        lookup[key] = {(r['seed'],r['turns']):r['B_minus_A'] for r in rows}
        for row in summarize(rows):
            e = interval([lookup[key][(s,row['turns'])] for s in sorted(seeds)],24)
            summary.append(dict(known_pair=key, **row, simultaneous_low=e['simultaneous_low'],
                                simultaneous_high=e['simultaneous_high']))
    contrasts = []
    for left,right in combinations(groups,2):
        for h in HORIZONS:
            values = [lookup[left][(s,h)]-lookup[right][(s,h)] for s in sorted(seeds)]
            contrasts.append(dict(left_pair=left,right_pair=right,turns=h,**interval(values,60)))
    return summary, contrasts


def run_study(output, seed_start=101, seed_end=1100):
    if not 101 <= seed_start <= seed_end <= 1100: raise ValueError('seed range must lie within 101..1100')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    sources = sorted((PROJECT_ROOT/'poc').rglob('*.py')) + sorted((PROJECT_ROOT/'poc/config').glob('*.json'))
    sources += [PROJECT_ROOT/'docs/PAIR_IDENTITY_PLAN.md']
    hashes = {p.relative_to(PROJECT_ROOT).as_posix():_sha256(p) for p in sources}
    started = datetime.now(timezone.utc).isoformat()
    groups = {}
    a_reference = {}
    names = {p.id:p.name for p in duration_config(12).participants}
    for pair in PAIRS:
        key = '-'.join(pair)
        folder = output/key
        folder.mkdir()
        _write_json(folder/'design.json',dict(version=VERSION,seed_start=seed_start,seed_end=seed_end,
            known_pair=pair,configs={str(h):to_jsonable(pair_config(pair,h)) for h in HORIZONS},source_sha256=hashes))
        rows = []
        with gzip.open(folder/'traces.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as traces, \
             (folder/'pairs.csv').open('w',encoding='utf-8-sig',newline='') as pf, \
             (folder/'seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as sf:
            pw = sw = None
            for seed in range(seed_start,seed_end+1):
                short = None
                for h in HORIZONS:
                    result = run_ab(seed=seed,config=pair_config(pair,h))
                    if short is not None: verify_prefix(short,result)
                    short = result
                    pairs,row = extract(result)
                    if pw is None:
                        pw,sw = csv.DictWriter(pf,fieldnames=list(pairs[0])),csv.DictWriter(sf,fieldnames=list(row))
                        pw.writeheader(); sw.writeheader()
                    pw.writerows(pairs); sw.writerow(row)
                    rows.append(row)
                    raw = compact_result(result)
                    for c in ('A','B'): raw['runs'][c]['snapshots'] = result['runs'][c]['snapshots']
                    a = json.dumps(raw['runs']['A'],ensure_ascii=False,sort_keys=True)
                    import hashlib
                    digest = hashlib.sha256(a.encode()).hexdigest()
                    index = (seed,h)
                    if index in a_reference and a_reference[index] != digest:
                        raise ValueError('A changed when B known pair changed')
                    a_reference[index] = digest
                    traces.write(json.dumps(raw,ensure_ascii=False,separators=(',',':'))+'\n')
                if (seed-seed_start+1)%100==0: print(f'{key}: {seed-seed_start+1}/{seed_end-seed_start+1}',flush=True)
        groups[key] = rows
        summary = summarize(rows)
        _write_csv(folder/'summary.csv',summary); _write_json(folder/'summary.json',summary)
        _write_json(folder/'manifest.json',dict(status='complete',source_sha256=hashes,
            paired_runs=len(rows),pair_rows=len(rows)*12,prefix_checks=6*(seed_end-seed_start+1),
            output_sha256={p.name:_sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}))
    summary,contrasts = combined(groups)
    for row in summary: row['known_pair_name'] = '-'.join(names[p] for p in row['known_pair'].split('-'))
    _write_csv(output/'summary.csv',summary); _write_json(output/'summary.json',summary)
    _write_csv(output/'contrasts.csv',contrasts); _write_json(output/'contrasts.json',contrasts)
    for name,digest in hashes.items():
        if _sha256(PROJECT_ROOT/name) != digest: raise ValueError('source changed during execution')
    manifest = dict(version=VERSION,status='complete',started_at_utc=started,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),seed_start=seed_start,seed_end=seed_end,
        condition_runs=48*(seed_end-seed_start+1),pair_rows=288*(seed_end-seed_start+1),
        A_invariance_checks=20*(seed_end-seed_start+1),source_sha256=hashes,
        output_sha256={p.relative_to(output).as_posix():_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    _write_json(output/'manifest.json',manifest)
    print(f'Completed: {output.resolve()}',flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Six known pairs x four horizons, rule model only')
    parser.add_argument('--seed-start',type=int,default=101)
    parser.add_argument('--seed-end',type=int,default=1100)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    try:
        run_study(args.output or PROJECT_ROOT/'results'/datetime.now().strftime('pair-identity-%Y%m%d-%H%M%S-%f'),
                  args.seed_start,args.seed_end)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(1,f'Experiment failed: {exc}. Folder without top-level manifest is incomplete.\n')


if __name__ == '__main__': main()
