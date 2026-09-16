"""Streaming audit: reduced raw events -> old evaluator -> pairs -> statistics."""
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poc.ab_poc.bridge_experiment import analyze_ab, _sha256
from poc.ab_poc.domain import Event, RelationshipThresholds, Stance, canonical_pair
from poc.ab_poc.rule_providers import RuleRelationshipEvaluator, RuleTalkInitiator
from poc.ab_poc.mechanism_experiment import cases, observations, summarize


def verify(output, *, write_report=True):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    design = json.loads((output / 'design.json').read_text(encoding='utf-8'))
    for name, digest in manifest['output_sha256'].items():
        assert _sha256(output / name) == digest, name
    for name, digest in manifest['source_sha256'].items():
        assert _sha256(ROOT / name) == digest, name
    expected = [(c.id, s) for c in cases() for s in range(design['seed_start'], design['seed_end'] + 1)]
    rows, seen = [], []
    reconstructed_pairs = 0
    with gzip.open(output / 'traces.jsonl.gz', 'rt', encoding='utf-8') as traces, \
         (output / 'seed_metrics.csv').open(encoding='utf-8-sig', newline='') as metrics_file, \
         (output / 'pairs.csv').open(encoding='utf-8-sig', newline='') as pairs_file:
        metrics = csv.DictReader(metrics_file)
        pairs = csv.DictReader(pairs_file)
        for line in traces:
            record = json.loads(line)
            result, case_id = record['result'], record['case_id']
            seen.append((case_id, result['root_seed']))
            row = {'case_id': case_id, **observations(result)}
            assert next(metrics) == {k: str(v) for k, v in row.items()}
            rows.append(row)
            pair_rows, _ = analyze_ab(result)
            for pair_row in pair_rows:
                pair_row['case_id'] = case_id
                assert next(pairs) == {k: str(v) for k, v in pair_row.items()}
            # Independent final-state reconstruction with the upstream evaluator.
            for run in result['runs'].values():
                evaluator = RuleRelationshipEvaluator(RelationshipThresholds(**run['rules']['relationship']))
                stances = {p['id']: Stance(p['stance']) for p in run['participants']}
                # Reduced traces omit fixed wording; the original rule maps
                # actor stance to this exact wording. Evaluator ignores it.
                events = [Event(**t, approach=RuleTalkInitiator._approach(stances[t['actor']])) for t in run['turns']]
                for state in run['final_relationships']:
                    key = canonical_pair(state['left_id'], state['right_id'])
                    pair_events = [e for e in events if canonical_pair(e.actor, e.target) == key]
                    if pair_events:
                        rebuilt = evaluator.evaluate(pair_events).pair
                        for field in ('onsite_relationship', 'conversation_count', 'positive_count', 'contact_history'):
                            actual = getattr(rebuilt, field)
                            assert actual == state[field], (case_id, result['root_seed'], key, field)
                        assert list(rebuilt.actor_ids) == state['actor_ids']
                    else:
                        assert state['onsite_relationship'] == '未形成' and state['conversation_count'] == state['positive_count'] == 0
                    reconstructed_pairs += 1
            if len(seen) % 5500 == 0:
                print(f'audited {len(seen)}/{len(expected)} paired runs', flush=True)
        assert next(metrics, None) is None and next(pairs, None) is None
    assert seen == expected
    aggregates, contrasts, factorial = summarize(rows, cases())
    assert aggregates == json.loads((output / 'case_summary.json').read_text(encoding='utf-8'))
    for filename, records in (('case_summary.csv', aggregates), ('contrasts.csv', contrasts), ('factorial_effects.csv', factorial)):
        if not records: continue
        with (output / filename).open(encoding='utf-8-sig', newline='') as handle:
            assert list(csv.DictReader(handle)) == [{k: '' if v is None else str(v) for k, v in row.items()} for row in records]
    assert reconstructed_pairs == manifest['pair_rows']
    audit = {'status': 'pass', 'paired_runs': len(seen), 'condition_runs': 2*len(seen),
             'pairs_rebuilt_from_events': reconstructed_pairs, 'no_missing_or_duplicate_cases_seeds': True,
             'raw_CSV_JSON_and_contrasts_consistent': True, 'source_and_output_hashes_match': True}
    target = ROOT / 'validation' / f'{output.name}-audit.json'
    if write_report:
        target.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit))
    return audit


if __name__ == '__main__':
    verify(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'results/mechanism-confirmed-001-1100')
