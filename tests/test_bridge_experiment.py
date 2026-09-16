"""Independent behavioral checks for matched-pair exports (standard library)."""

import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from poc.ab_poc.bridge_experiment import aggregate, analyze_ab, main, run_experiment
from poc.ab_poc.domain import canonical_pair
from poc.ab_poc.engine import run_ab
from poc.ab_poc.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]


def set_final_stages(result, stages):
    """Construct a final-state fixture, without using analyzer classification."""
    for condition, run in result['runs'].items():
        for state in run['final_relationships']:
            pair = canonical_pair(state['left_id'], state['right_id'])
            state['onsite_relationship'] = stages.get((condition, pair), '未形成')
        run['snapshots'][-1]['pairs'] = copy.deepcopy(run['final_relationships'])
        run['metrics'] = compute_metrics(run)
    return result


class BridgeExperimentTests(unittest.TestCase):
    def test_all_pairs_are_classified_identically_in_A_and_B(self):
        rows, summary = analyze_ab(run_ab(seed=1))
        expected = {
            ('akane', 'midori'): 'known_pair',
            ('akane', 'koharu'): 'bridge_edge',
            ('akane', 'kurumi'): 'bridge_edge',
            ('koharu', 'midori'): 'bridge_edge',
            ('kurumi', 'midori'): 'bridge_edge',
            ('koharu', 'kurumi'): 'third_party_pair',
        }
        self.assertEqual(len(rows), 12)
        for condition in ('A', 'B'):
            selected = [row for row in rows if row['condition'] == condition]
            self.assertEqual({(r['left_id'], r['right_id']): r['pair_type'] for r in selected}, expected)
            self.assertEqual(sum(r['online_known'] for r in selected), int(condition == 'B'))
        self.assertEqual(summary['bridge_edge_count_A'], 2)
        self.assertEqual(summary['bridge_edge_count_B'], 2)

    def test_known_pair_conversion_does_not_create_bridges(self):
        r = set_final_stages(run_ab(seed=1), {('B', ('akane', 'midori')): '親しみがある'})
        rows, s = analyze_ab(r)
        self.assertEqual(s['known_pair_count_B'], 1)
        self.assertEqual(s['bridge_edge_count_B'], 0)
        self.assertEqual(s['matched_new_count_B'], 0)
        self.assertEqual(s['B_minus_A'], 0)

    def test_familiar_counts_once_and_third_party_is_separate(self):
        r = set_final_stages(run_ab(seed=1), {
            ('B', ('akane', 'koharu')): '親しみがある',
            ('B', ('koharu', 'kurumi')): '顔見知り',
        })
        _, s = analyze_ab(r)
        self.assertEqual(s['bridge_edge_count_B'], 1)
        self.assertEqual(s['bridge_edge_familiar_count_B'], 1)
        self.assertEqual(s['bridge_edge_rate_B'], .25)
        self.assertEqual(s['matched_new_count_B'], 2)
        self.assertEqual(s['third_party_pair_count_B'], 1)
        self.assertEqual(s['B_minus_A'], 1)

    def test_same_network_has_same_matched_rates_despite_legacy_denominators(self):
        stages = {(c, p): '顔見知り' for c in ('A', 'B') for p in
                  (('akane', 'koharu'), ('akane', 'kurumi'), ('koharu', 'midori'))}
        _, s = analyze_ab(set_final_stages(run_ab(seed=1), stages))
        self.assertEqual(s['legacy_new_acquaintance_rate_A'], .5)
        self.assertEqual(s['legacy_new_acquaintance_rate_B'], .6)
        self.assertEqual(s['bridge_edge_rate_A'], .75)
        self.assertEqual(s['bridge_edge_rate_B'], .75)
        self.assertEqual(s['matched_new_rate_A'], .6)
        self.assertEqual(s['matched_new_rate_B'], .6)
        self.assertEqual(s['B_minus_A'], 0)

    def test_missing_duplicate_invalid_and_incomplete_states_are_rejected(self):
        for failure in ('missing', 'duplicate', 'invalid_stage', 'no_end', 'wrong_snapshot', 'wrong_seed', 'wrong_rules', 'wrong_online'):
            with self.subTest(failure=failure):
                r = run_ab(seed=1)
                run = r['runs']['B']
                if failure == 'missing':
                    run['final_relationships'].pop()
                    run['snapshots'][-1]['pairs'] = copy.deepcopy(run['final_relationships'])
                elif failure == 'duplicate':
                    run['final_relationships'][-1] = copy.deepcopy(run['final_relationships'][0])
                    run['snapshots'][-1]['pairs'] = copy.deepcopy(run['final_relationships'])
                elif failure == 'invalid_stage':
                    run['final_relationships'][0]['onsite_relationship'] = 'unknown'
                    run['snapshots'][-1]['pairs'] = copy.deepcopy(run['final_relationships'])
                elif failure == 'no_end':
                    run['snapshots'].pop()
                elif failure == 'wrong_snapshot':
                    run['snapshots'][-1]['pairs'] = []
                elif failure == 'wrong_seed':
                    run['root_seed'] = 99
                elif failure == 'wrong_rules':
                    run['rules']['turns'] = 8
                elif failure == 'wrong_online':
                    run['final_relationships'][0]['online_known'] = 'あり'
                with self.assertRaises(ValueError):
                    analyze_ab(r)

    def test_no_known_pair_or_more_than_one_is_rejected(self):
        for pairs in ([], [['akane', 'midori'], ['koharu', 'kurumi']]):
            r = run_ab(seed=1)
            r['runs']['B']['condition']['online_known_pairs'] = pairs
            with self.assertRaises(ValueError):
                analyze_ab(r)

    def test_analyzer_does_not_mutate_original_results(self):
        r = run_ab(seed=9)
        before = copy.deepcopy(r)
        analyze_ab(r)
        self.assertEqual(r, before)

    def test_manual_ten_seed_results_including_japanese_stages(self):
        # Source: user's handover, recorded BEFORE this implementation.
        expected = [
            (3, 3, 2, 1, '親しみがある', [], [], 2),
            (3, 2, 2, 1, '親しみがある', [], [], 2),
            (3, 3, 2, 2, '顔見知り', [], [], 2),
            (3, 3, 0, 0, '未形成', ['midori'], ['midori'], 2),
            (2, 2, 0, 0, '顔見知り', [], [], 2),
            (4, 3, 1, 1, '顔見知り', [], [], 2),
            (4, 4, 1, 1, '未形成', [], [], 4),
            (2, 2, 2, 0, '親しみがある', [], [], 2),
            (2, 1, 0, 1, '親しみがある', ['akane'], ['kurumi'], 1),
            (3, 2, 2, 2, '顔見知り', [], [], 2),
        ]
        for seed, values in enumerate(expected, 1):
            r = run_ab(seed=seed)
            a, b = r['metrics']['A'], r['metrics']['B']
            self.assertEqual((a['new_acquaintance_count'], b['new_acquaintance_count'],
                              a['new_familiar_count'], b['new_familiar_count'],
                              b['known_pair_onsite']['stage'], a['isolated_participants'],
                              b['isolated_participants'], len(b['third_party_observation_paths'])), values)

    def test_100_seeds_match_pre_edit_results_and_B_existing_paths(self):
        baseline = json.loads((ROOT / 'validation/baseline_run_hashes.json').read_text())
        for seed in range(1, 101):
            r = run_ab(seed=seed)
            serialized = json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
            self.assertEqual(hashlib.sha256(serialized).hexdigest(), baseline[str(seed)])
            rows, summary = analyze_ab(r)
            self.assertEqual(summary['bridge_edge_count_B'], len(r['metrics']['B']['third_party_observation_paths']))
            for c in ('A', 'B'):
                self.assertEqual(summary[f'bridge_edge_rate_{c}'], summary[f'bridge_edge_count_{c}'] / 4)
                self.assertEqual(summary[f'matched_new_rate_{c}'], summary[f'matched_new_count_{c}'] / 5)

    def test_exports_round_trip_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / '日本語の実験'
            result = run_experiment([1, 9], output)
            self.assertEqual(result['seed_count'], 2)
            self.assertTrue((output / 'pairs.csv').read_bytes().startswith(b'\xef\xbb\xbf'))
            with (output / 'pairs.csv').open(encoding='utf-8-sig', newline='') as handle:
                csv_rows = list(csv.DictReader(handle))
            json_rows = json.loads((output / 'pairs.json').read_text(encoding='utf-8'))
            self.assertEqual(csv_rows, [{k: str(v) for k, v in row.items()} for row in json_rows])
            raw_runs = [json.loads(line) for line in (output / 'raw_runs.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(json_rows, [row for raw in raw_runs for row in analyze_ab(raw)[0]])
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            for name, digest in manifest['output_sha256'].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            original = (output / 'pairs.csv').read_bytes()
            with self.assertRaises(FileExistsError):
                run_experiment([2], output)
            self.assertEqual((output / 'pairs.csv').read_bytes(), original)

    def test_cli_smoke_and_invalid_range(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'smoke'
            proc = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'poc.ab_poc.bridge_experiment',
                                   '--seed-start', '1', '--seed-end', '1', '--output', str(out)],
                                  cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out / 'manifest.json').exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(['--seed-start', '10', '--seed-end', '1'])
            self.assertEqual(error.exception.code, 2)

    def test_aggregation_is_paired_and_rejects_duplicate_seeds(self):
        summaries = [analyze_ab(run_ab(seed=s))[1] for s in (1, 2, 9)]
        report = aggregate(summaries)
        self.assertEqual(sum(report['bridge_paired_seeds'].values()), 3)
        self.assertEqual(report['means']['B_minus_A'], sum(s['B_minus_A'] for s in summaries) / 3)
        with self.assertRaises(ValueError):
            aggregate([summaries[0], summaries[0]])


if __name__ == '__main__':
    unittest.main()
