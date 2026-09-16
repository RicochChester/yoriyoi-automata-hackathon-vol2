"""Vary conversation opportunities without editing the original 12-turn model."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
from statistics import mean

from .bridge_experiment import analyze_ab, _sha256, _write_csv, _write_json, PROJECT_ROOT
from .config import load_config
from .domain import canonical_pair, to_jsonable
from .engine import run_ab
from .mechanism_experiment import compact_result, estimate

HORIZONS = (12, 24, 36, 48)
VERSION = 'yoriyoi.duration.v1'


def duration_config(turns):
    if type(turns) is not int or turns not in HORIZONS:
        raise ValueError('duration v1 supports 12, 24, 36, 48 turns only')
    config = load_config()
    if config.rules.turns != 12:
        raise ValueError('expected original 12-turn baseline')
    return replace(config, rules=replace(config.rules, turns=turns))


def verify_prefix(short, long):
    for c in ('A', 'B'):
        a, b = short['runs'][c], long['runs'][c]
        n = a['rules']['turns']
        if a['turns'] != b['turns'][:n] or a['turn_order'] != b['turn_order'][:n]:
            raise ValueError('longer run changed earlier events or actor order')
        for snap in a['snapshots']:
            phase = 'turn_end' if snap['phase'] == 'end' else snap['phase']
            match = next(s for s in b['snapshots'] if s['turn'] == snap['turn'] and s['phase'] == phase)
            if snap['pairs'] != match['pairs']:
                raise ValueError('longer run changed an earlier pair state')


def extract(result):
    pairs, summary = analyze_ab(result)
    horizon = result['runs']['A']['rules']['turns']
    first = {}
    for c, run in result['runs'].items():
        for snapshot in run['snapshots']:
            if snapshot['phase'] != 'turn_end': continue
            for state in snapshot['pairs']:
                key = (c, *canonical_pair(state['left_id'], state['right_id']))
                times = first.setdefault(key, {'acquaintance': None, 'familiar': None})
                if state['onsite_relationship'] != '未形成' and times['acquaintance'] is None:
                    times['acquaintance'] = snapshot['turn']
                if state['onsite_relationship'] == '親しみがある' and times['familiar'] is None:
                    times['familiar'] = snapshot['turn']
    for row in pairs:
        times = first[(row['condition'], row['left_id'], row['right_id'])]
        row.update(turns=horizon, first_acquaintance_turn=times['acquaintance'],
                   first_familiar_turn=times['familiar'], acquaintance_censored=times['acquaintance'] is None,
                   familiar_censored=times['familiar'] is None,
                   unformed_turns=horizon if times['acquaintance'] is None else times['acquaintance'] - 1)
    summary['turns'] = horizon
    known = canonical_pair(*result['runs']['B']['condition']['online_known_pairs'][0])
    for c in ('A', 'B'):
        bridge = [row for row in pairs if row['condition'] == c and row['pair_type'] == 'bridge_edge']
        summary[f'bridge_all_formed_{c}'] = int(all(row['formed'] for row in bridge))
        summary[f'bridge_unformed_turns_{c}'] = sum(row['unformed_turns'] for row in bridge) / 4
        turns = result['runs'][c]['turns']
        known_count = sum(canonical_pair(t['actor'], t['target']) == known for t in turns)
        outgoing = sum(t['actor'] in known and t['target'] not in known for t in turns)
        if known_count + outgoing != horizon // 2:
            raise ValueError('endpoint conversation budget violated')
        summary[f'known_conversations_{c}'] = known_count
        summary[f'known_share_{c}'] = known_count / horizon
        summary[f'endpoint_outbound_bridge_{c}'] = outgoing
    return pairs, summary


def summarize(rows):
    lookup = {(r['seed'], r['turns']): r for r in rows}
    if len(lookup) != len(rows): raise ValueError('duplicate seed/horizon')
    seeds = sorted({r['seed'] for r in rows})
    if set(lookup) != {(s, h) for s in seeds for h in HORIZONS}:
        raise ValueError('incomplete horizon matrix')
    result = []
    for h in HORIZONS:
        selected = [lookup[(s, h)] for s in seeds]
        difference = [r['B_minus_A'] for r in selected]
        delta = estimate(difference)
        recovery = estimate([lookup[(s, h)]['B_minus_A'] - lookup[(s, 12)]['B_minus_A'] for s in seeds])
        row = {'turns': h, 'seeds': len(seeds),
               'bridge_A': mean(r['bridge_edge_count_A'] for r in selected),
               'bridge_B': mean(r['bridge_edge_count_B'] for r in selected),
               'B_minus_A': delta['mean'], 'ci_low': delta['ci_low'], 'ci_high': delta['ci_high'],
               'gap_change_from_12': recovery['mean'], 'change_ci_low': recovery['ci_low'], 'change_ci_high': recovery['ci_high'],
               'B_greater': sum(d > 0 for d in difference), 'equal': sum(d == 0 for d in difference), 'B_less': sum(d < 0 for d in difference)}
        for metric in ('bridge_edge_rate', 'bridge_edge_familiar_count', 'bridge_all_formed', 'bridge_unformed_turns',
                       'known_conversations', 'known_share', 'endpoint_outbound_bridge', 'matched_new_count'):
            for c in ('A', 'B'): row[f'{metric}_{c}'] = mean(r[f'{metric}_{c}'] for r in selected)
        waiting = estimate([r['bridge_unformed_turns_B'] - r['bridge_unformed_turns_A'] for r in selected])
        row.update(unformed_turns_B_minus_A=waiting['mean'], unformed_ci_low=waiting['ci_low'], unformed_ci_high=waiting['ci_high'])
        result.append(row)
    return result


def run_study(output, seed_start=101, seed_end=1100):
    if not (101 <= seed_start <= seed_end <= 1100):
        raise ValueError('seed range must lie within 101..1100')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = PROJECT_ROOT / 'docs/DURATION_PLAN.md'
    sources = sorted((PROJECT_ROOT / 'poc').rglob('*.py')) + sorted((PROJECT_ROOT / 'poc/config').glob('*.json')) + [plan]
    hashes = {p.relative_to(PROJECT_ROOT).as_posix(): _sha256(p) for p in sources}
    started = datetime.now(timezone.utc).isoformat()
    _write_json(output / 'design.json', {'version': VERSION, 'started_at_utc': started,
        'seed_start': seed_start, 'seed_end': seed_end, 'horizons': HORIZONS,
        'configs': {str(h): to_jsonable(duration_config(h)) for h in HORIZONS}, 'source_sha256': hashes})
    rows = []
    with gzip.open(output / 'traces.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as traces, \
         (output / 'pairs.csv').open('w', encoding='utf-8-sig', newline='') as pf, \
         (output / 'seed_metrics.csv').open('w', encoding='utf-8-sig', newline='') as sf:
        pw = sw = None
        for seed in range(seed_start, seed_end + 1):
            short = None
            for horizon in HORIZONS:
                result = run_ab(seed=seed, config=duration_config(horizon))
                if short is not None: verify_prefix(short, result)
                short = result
                pairs, summary = extract(result)
                if pw is None:
                    pw, sw = csv.DictWriter(pf, fieldnames=list(pairs[0])), csv.DictWriter(sf, fieldnames=list(summary))
                    pw.writeheader(); sw.writeheader()
                pw.writerows(pairs); sw.writerow(summary)
                rows.append(summary)
                raw = compact_result(result)
                for c in ('A', 'B'): raw['runs'][c]['snapshots'] = result['runs'][c]['snapshots']
                traces.write(json.dumps(raw, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            if (seed - seed_start + 1) % 100 == 0:
                print(f'{seed-seed_start+1}/{seed_end-seed_start+1} seeds complete (all four horizons)', flush=True)
    summary = summarize(rows)
    _write_csv(output / 'summary.csv', summary)
    _write_json(output / 'summary.json', summary)
    for name, digest in hashes.items():
        if _sha256(PROJECT_ROOT / name) != digest: raise ValueError(f'source changed during execution: {name}')
    manifest = {'version': VERSION, 'status': 'complete', 'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'paired_runs': len(rows), 'condition_runs': 2*len(rows), 'pair_rows': 12*len(rows),
        'prefix_checks': 6*(seed_end-seed_start+1), 'source_sha256': hashes,
        'output_sha256': {p.name: _sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    _write_json(output / 'manifest.json', manifest)
    print(f'Completed: {output.resolve()}', flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='12/24/36/48 turn paired A/B study; no API calls.')
    parser.add_argument('--seed-start', type=int, default=101)
    parser.add_argument('--seed-end', type=int, default=1100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 101 <= args.seed_start <= args.seed_end <= 1100: parser.error('seed range must be within 101..1100')
    output = args.output or PROJECT_ROOT / 'results' / datetime.now().strftime('duration-%Y%m%d-%H%M%S-%f')
    if output.exists(): parser.error('output exists; choose a new directory')
    try: run_study(output, args.seed_start, args.seed_end)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f'Experiment failed: {exc}\nPartial output: {output}; without manifest it is incomplete.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
