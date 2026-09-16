"""Local causal-path interventions and bonus sensitivity for the fixed A/B model.

Original engine, thresholds, and config files remain untouched. Experimental
providers wrap the existing selection/topic/reaction implementations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import platform
from statistics import mean, stdev
import time
from typing import Any

from .bridge_experiment import analyze_ab, _write_csv, _write_json, _sha256, PROJECT_ROOT
from .config import load_config
from .contracts import Initiation
from .domain import OnlineKnown, canonical_pair, to_jsonable
from .engine import run_ab
from .rule_providers import RuleParticipantProvider, RuleTalkInitiator, RuleResponseProvider


VERSION = 'yoriyoi.mechanism.v1'
KNOWN = canonical_pair('akane', 'midori')
TURN_FIELDS = ('turn', 'actor', 'target', 'topic', 'response')
MEASURES = ('bridge_edge_count', 'bridge_edge_rate', 'bridge_edge_familiar_count',
            'matched_new_count', 'known_pair_count', 'known_pair_familiar_count',
            'known_conversations', 'known_share', 'bridge_conversations',
            'third_party_conversations', 'endpoint_outbound_bridge', 'third_party_to_known')


@dataclass(frozen=True)
class Case:
    selection: float
    reaction: int
    topic: bool
    caution: bool
    clamp: str = ''

    def __post_init__(self):
        if type(self.selection) not in (int, float) or not math.isfinite(self.selection) or self.selection < 0:
            raise ValueError('selection bonus must be finite and nonnegative')
        if type(self.reaction) is not int or self.reaction < 0:
            raise ValueError('reaction bonus must be a nonnegative integer')
        if type(self.topic) is not bool or type(self.caution) is not bool or self.clamp not in ('', 'A', 'B'):
            raise ValueError('invalid pathway switches or clamp donor')

    @property
    def id(self):
        if self.clamp:
            return f'clamp_{self.clamp}'
        return f'S{self.selection:g}_R{self.reaction}_T{int(self.topic)}_C{int(self.caution)}'

    @property
    def factorial(self):
        return not self.clamp and self.selection in (0, 1) and self.reaction in (0, 2)

    @property
    def sweep(self):
        return not self.clamp and self.topic and self.caution


def cases():
    design = [Case(s, r, True, True) for s, r in product((0, .5, 1, 2, 4), (0, 1, 2, 4))]
    design += [Case(s, r, t, c) for s, r, t, c in product((0, 1), (0, 2), (False, True), (False, True))]
    design += [Case(1, 2, True, True, donor) for donor in ('A', 'B')]
    return tuple(dict.fromkeys(design))


class MechanismInitiator(RuleTalkInitiator):
    def __init__(self, topic=True, schedule=()):
        self.topic_enabled = topic
        self.schedule = tuple(schedule)

    def _choose_topic(self, context, target, pair_events):
        if not self.topic_enabled:
            context = replace(context, online_memories=())
        return super()._choose_topic(context, target, pair_events)

    def initiate(self, context):
        if not self.schedule:
            return super().initiate(context)
        # Stock target selection consumes exactly one random() draw. Keep that
        # position even when externally fixing the target; do not draw a topic
        # for the discarded natural target.
        context.rng.random()
        actor, target_id = self.schedule[context.turn - 1]
        if actor != context.actor.id:
            raise ValueError('clamp donor actor order differs from this run')
        target = next((p for p in context.candidates if p.id == target_id), None)
        if target is None:
            raise ValueError('clamp target is not an eligible participant')
        pair = canonical_pair(actor, target_id)
        events = tuple(e for e in context.events if canonical_pair(e.actor, e.target) == pair)
        return Initiation(target_id, self._choose_topic(context, target, events), self._approach(context.actor.stance))


class MechanismResponder(RuleResponseProvider):
    def __init__(self, caution=True):
        self.caution_enabled = caution

    def reaction_points(self, context, target, topic, pair_events):
        pair = canonical_pair(context.actor.id, target.id)
        state = context.pair_states.get(pair)
        if self.caution_enabled or state is None or state.online_known is not OnlineKnown.DIRECT:
            return super().reaction_points(context, target, topic, pair_events)
        # Hide online-known only from this reaction calculation. This restores
        # ordinary caution penalties; add back the configured direct bonus so
        # that the R and C interventions stay separate. No state mutation.
        states = dict(context.pair_states)
        states[pair] = replace(state, online_known=OnlineKnown.NONE)
        masked = replace(context, pair_states=states)
        return super().reaction_points(masked, target, topic, pair_events) + context.rules.reaction.online_known_bonus


def config_for(case):
    config = load_config()
    if config.rules.turns != 12 or len(config.participants) != 4 or config.conditions[1].online_known_pairs != (KNOWN,):
        raise ValueError('experiment v1 requires the original four-person, 12-turn, akane-midori design')
    return replace(config, rules=replace(config.rules,
        selection=replace(config.rules.selection, online_known_bonus=float(case.selection)),
        reaction=replace(config.rules.reaction, online_known_bonus=case.reaction)))


def run_case(case, seed, *, donor_result=None):
    schedule = ()
    if case.clamp:
        donor_result = donor_result if donor_result is not None else run_ab(seed=seed)
        if donor_result['root_seed'] != seed:
            raise ValueError('clamp donor seed differs')
        schedule = tuple((t['actor'], t['target']) for t in donor_result['runs'][case.clamp]['turns'])
    def factory():
        provider = RuleParticipantProvider(initiator=MechanismInitiator(case.topic, schedule),
                                           responder=MechanismResponder(case.caution))
        # All-on natural is intentionally the exact legacy provider metadata.
        # Other pathway overrides are also named in each saved case record.
        if case.clamp or not case.topic or not case.caution:
            provider.name = 'mechanism-rule-participant'
            provider.version = VERSION
        return provider
    return run_ab(seed=seed, config=config_for(case), participant_provider_factory=factory)


def behavior(run, *, bridge_only=False):
    def include(pair):
        return not bridge_only or (pair != KNOWN and bool(set(pair) & set(KNOWN)))
    return {
        'turns': [{key: turn[key] for key in TURN_FIELDS} for turn in run['turns']
                  if include(canonical_pair(turn['actor'], turn['target']))],
        'pairs': [{key: value for key, value in state.items() if key != 'online_known'}
                  for state in run['final_relationships'] if include(canonical_pair(state['left_id'], state['right_id']))],
    }


def observations(result):
    _, old = analyze_ab(result)
    out = {'seed': result['root_seed']}
    for c in ('A', 'B'):
        for m in MEASURES[:6]:
            out[f'{m}_{c}'] = old[f'{m}_{c}']
        counts = dict.fromkeys(('known_conversations', 'bridge_conversations', 'third_party_conversations',
                               'endpoint_outbound_bridge', 'third_party_to_known'), 0)
        endpoint_turns = 0
        for turn in result['runs'][c]['turns']:
            pair = canonical_pair(turn['actor'], turn['target'])
            endpoint_turns += turn['actor'] in KNOWN
            if pair == KNOWN:
                counts['known_conversations'] += 1
            elif set(pair) & set(KNOWN):
                counts['bridge_conversations'] += 1
                counts['endpoint_outbound_bridge' if turn['actor'] in KNOWN else 'third_party_to_known'] += 1
            else:
                counts['third_party_conversations'] += 1
        if sum(counts[k] for k in ('known_conversations', 'bridge_conversations', 'third_party_conversations')) != 12:
            raise ValueError('12-turn conversation budget violated')
        if endpoint_turns != 6 or counts['known_conversations'] + counts['endpoint_outbound_bridge'] != 6:
            raise ValueError('six endpoint-initiated conversations budget violated')
        for key, value in counts.items():
            out[f'{key}_{c}'] = value
        out[f'known_share_{c}'] = counts['known_conversations'] / 12
    for m in MEASURES:
        out[f'{m}_delta'] = out[f'{m}_B'] - out[f'{m}_A']
    return out


def compact_result(result):
    """Keep actual events/end states needed to independently recompute outcomes.

    Discard repeated intermediate snapshots and presentation/affect fields;
    this is an explicitly reduced trace, not the full HTTP response.
    """
    runs = {}
    for c, run in result['runs'].items():
        runs[c] = {k: run[k] for k in ('root_seed', 'condition', 'participants', 'rules', 'turn_order',
                                     'participant_provider', 'relationship_evaluator', 'final_relationships', 'metrics')}
        runs[c]['turns'] = [{k: t[k] for k in TURN_FIELDS} for t in run['turns']]
        runs[c]['snapshots'] = [run['snapshots'][-1]]
    return {'root_seed': result['root_seed'], 'participants': result['participants'], 'runs': runs}


def estimate(values):
    values = tuple(values)
    if not values:
        raise ValueError('empty estimate')
    avg = mean(values)
    se = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {'n': len(values), 'mean': avg, 'standard_error': se,
            'ci_low': avg - 1.96 * se if se is not None else None,
            'ci_high': avg + 1.96 * se if se is not None else None}


def _result_digest(result):
    return hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _reference_result(seed):
    result = run_ab(seed=seed)
    return {'root_seed': seed, 'digest': _result_digest(result),
            'runs': {c: {'turns': [{k: t[k] for k in TURN_FIELDS} for t in r['turns']],
                         'final_relationships': r['final_relationships']}
                     for c, r in result['runs'].items()}}


def summarize(rows, design):
    grouped = {}
    for row in rows:
        block = 'confirmation_101_1100' if row['seed'] > 100 else 'calibration_1_100'
        grouped.setdefault((row['case_id'], block), []).append(row)
    aggregates = []
    for (case_id, block), selected in grouped.items():
        for measure in MEASURES:
            delta = [r[f'{measure}_delta'] for r in selected]
            aggregates.append({'case_id': case_id, 'block': block, 'measure': measure,
                'mean_A': mean(r[f'{measure}_A'] for r in selected),
                'mean_B': mean(r[f'{measure}_B'] for r in selected), **estimate(delta),
                'B_greater': sum(d > 0 for d in delta), 'equal': sum(d == 0 for d in delta), 'B_less': sum(d < 0 for d in delta)})
    by_key = {(r['case_id'], r['seed']): r for r in rows}
    seeds = sorted({r['seed'] for r in rows if r['seed'] > 100})
    contrasts = []
    if seeds:
        full = Case(1, 2, True, True).id
        comparison_cases = (Case(0, 2, True, True), Case(1, 0, True, True), Case(1, 2, False, True),
                            Case(1, 2, True, False), Case(0, 0, False, False),
                            Case(1, 2, True, True, 'A'), Case(1, 2, True, True, 'B'))
        for case in comparison_cases:
            for m in ('bridge_edge_count', 'known_conversations', 'endpoint_outbound_bridge'):
                differences = [by_key[(case.id, s)][f'{m}_delta'] - by_key[(full, s)][f'{m}_delta'] for s in seeds]
                contrasts.append({'contrast': f'{case.id} minus {full}', 'measure': m, **estimate(differences)})
    factorial = []
    flags_to_id = {(int(c.selection), c.reaction // 2, int(c.topic), int(c.caution)): c.id for c in design if c.factorial}
    if seeds:
        for indices in [*( (i,) for i in range(4)), *combinations(range(4), 2)]:
            background = [i for i in range(4) if i not in indices]
            for measure in ('bridge_edge_count', 'known_conversations', 'endpoint_outbound_bridge'):
                differences = []
                for seed in seeds:
                    background_effects = []
                    for bg in product((0, 1), repeat=len(background)):
                        total = 0
                        for levels in product((0, 1), repeat=len(indices)):
                            flags = [0] * 4
                            for i, val in zip(background, bg): flags[i] = val
                            for i, val in zip(indices, levels): flags[i] = val
                            sign = (-1) ** (len(indices) - sum(levels))
                            total += sign * by_key[(flags_to_id[tuple(flags)], seed)][f'{measure}_B']
                        background_effects.append(total)
                    differences.append(mean(background_effects))
                factorial.append({'effect': '*'.join('SRTC'[i] for i in indices), 'measure': measure, **estimate(differences)})
    return aggregates, contrasts, factorial


def run_study(output, seed_start=1, seed_end=1100):
    if not (1 <= seed_start <= seed_end <= 1100):
        raise ValueError('v1 seed range must be within 1..1100')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = PROJECT_ROOT / 'docs/MECHANISM_PLAN.md'
    files = sorted((PROJECT_ROOT / 'poc').rglob('*.py')) + sorted((PROJECT_ROOT / 'poc/config').glob('*.json')) + [plan]
    hashes = {p.relative_to(PROJECT_ROOT).as_posix(): _sha256(p) for p in files}
    design = cases()
    started = datetime.now(timezone.utc).isoformat()
    _write_json(output / 'design.json', {'version': VERSION, 'started_at_utc': started,
        'seed_start': seed_start, 'seed_end': seed_end, 'confirmation_start': 101,
        'plan_sha256': _sha256(plan), 'cases': [dict(id=c.id, **asdict(c), factorial=c.factorial, sweep=c.sweep) for c in design],
        'configs': {c.id: to_jsonable(config_for(c)) for c in design},
        'source_sha256': hashes})
    all_rows = []
    reference = {}
    for seed in range(seed_start, seed_end + 1):
        reference[seed] = _reference_result(seed)
        if seed % 100 == 0:
            print(f'reference seed {seed}/{seed_end}', flush=True)
    checks = dict(A_natural_invariant=0, all_off_equal=0, clamp_bridge_matches_donor=0, default_exact=0)
    import csv
    with gzip.open(output / 'traces.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as trace, \
         (output / 'pairs.csv').open('w', encoding='utf-8-sig', newline='') as pair_file, \
         (output / 'seed_metrics.csv').open('w', encoding='utf-8-sig', newline='') as metric_file:
        pair_writer = metric_writer = None
        for i, case in enumerate(design, 1):
            tic = time.perf_counter()
            for seed in range(seed_start, seed_end + 1):
                result = run_case(case, seed, donor_result=reference[seed])
                if not case.clamp:
                    if behavior(result['runs']['A']) != behavior(reference[seed]['runs']['A']):
                        raise ValueError(f'A unexpectedly changed: {case.id} seed={seed}')
                    checks['A_natural_invariant'] += 1
                else:
                    for c in ('A', 'B'):
                        if behavior(result['runs'][c], bridge_only=True) != behavior(reference[seed]['runs'][case.clamp], bridge_only=True):
                            raise ValueError(f'clamp did not preserve donor bridge behavior: {case.id} seed={seed}')
                        checks['clamp_bridge_matches_donor'] += 1
                if case == Case(1, 2, True, True):
                    if _result_digest(result) != reference[seed]['digest']:
                        raise ValueError('all-on condition differs from original')
                    checks['default_exact'] += 1
                if case == Case(0, 0, False, False):
                    if behavior(result['runs']['A']) != behavior(result['runs']['B']):
                        raise ValueError('all-off negative control has residual behavioral difference')
                    checks['all_off_equal'] += 1
                row = {'case_id': case.id, **observations(result)}
                pair_rows, _ = analyze_ab(result)
                for pair_row in pair_rows:
                    pair_row['case_id'] = case.id
                if pair_writer is None:
                    pair_writer = csv.DictWriter(pair_file, fieldnames=list(pair_rows[0]))
                    pair_writer.writeheader()
                    metric_writer = csv.DictWriter(metric_file, fieldnames=list(row))
                    metric_writer.writeheader()
                pair_writer.writerows(pair_rows)
                metric_writer.writerow(row)
                all_rows.append(row)
                trace.write(json.dumps({'case_id': case.id, 'result': compact_result(result)}, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            print(f'{i}/{len(design)} {case.id}: {seed_end-seed_start+1} seeds, {time.perf_counter()-tic:.1f}s', flush=True)
    aggregates, contrasts, factorial = summarize(all_rows, design)
    _write_csv(output / 'case_summary.csv', aggregates)
    _write_json(output / 'case_summary.json', aggregates)
    if contrasts:
        _write_csv(output / 'contrasts.csv', contrasts)
        _write_csv(output / 'factorial_effects.csv', factorial)
    for name, digest in hashes.items():
        if _sha256(PROJECT_ROOT / name) != digest:
            raise ValueError(f'source or plan changed during execution: {name}')
    manifest = {'version': VERSION, 'status': 'complete', 'python_version': platform.python_version(),
        'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'case_count': len(design), 'seeds_per_case': seed_end-seed_start+1,
        'paired_runs': len(all_rows), 'condition_runs': 2*len(all_rows), 'pair_rows': 12*len(all_rows),
        'trace_format': 'Reduced actual event logs and final snapshots; full presentation/affect snapshots omitted.',
        'ci_method': 'mean paired difference +/- 1.96 * sample_sd / sqrt(n); approximate Monte Carlo interval, not human uncertainty; no multiple-test declarations',
        'checks': checks, 'source_sha256': hashes,
        'output_sha256': {p.name: _sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    _write_json(output / 'manifest.json', manifest)
    print(f'Completed. {output.resolve()}', flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run fixed 34-case mechanism/bonus experiments without API calls.')
    parser.add_argument('--seed-start', type=int, default=1)
    parser.add_argument('--seed-end', type=int, default=1100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.seed_start <= args.seed_end <= 1100:
        parser.error('seed range must be within 1..1100')
    output = args.output or PROJECT_ROOT / 'results' / datetime.now().strftime('mechanism-%Y%m%d-%H%M%S-%f')
    if output.exists(): parser.error('output already exists; choose a new directory')
    try:
        run_study(output, args.seed_start, args.seed_end)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Experiment failed: {exc}\nPartial output: {output}. No manifest means incomplete.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
