import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from poc.ab_poc.config import load_config
from poc.ab_poc.domain import to_jsonable
from poc.ab_poc.engine import run_ab
from poc.ab_poc.duration_experiment import HORIZONS, duration_config, extract, verify_prefix, summarize, run_study


class DurationTests(unittest.TestCase):
    def test_only_turns_change_and_12_is_original(self):
        original = to_jsonable(load_config())
        for horizon in HORIZONS:
            changed = to_jsonable(duration_config(horizon))
            self.assertEqual(changed['rules']['turns'], horizon)
            changed['rules']['turns'] = 12
            self.assertEqual(changed, original)
        self.assertEqual(run_ab(seed=101, config=duration_config(12)), run_ab(seed=101))

    def test_prefix_and_formation_times(self):
        for seed in (101, 105, 1001):
            short = None
            for h in HORIZONS:
                result = run_ab(seed=seed, config=duration_config(h))
                if short: verify_prefix(short, result)
                short = result
                pairs, summary = extract(result)
                self.assertEqual(len(pairs), 12)
                for row in pairs:
                    self.assertEqual(row['acquaintance_censored'], not row['formed'])
                    self.assertEqual(row['unformed_turns'], h if not row['formed'] else row['first_acquaintance_turn']-1)
                    self.assertTrue(0 <= row['unformed_turns'] <= h)

    def test_changed_prefix_rejected(self):
        a,b = (run_ab(seed=101, config=duration_config(h)) for h in (12,24))
        b['runs']['A']['turns'][0]['topic'] = 'altered'
        with self.assertRaises(ValueError): verify_prefix(a,b)

    def test_complete_matrix_required_and_paired_change(self):
        rows = []
        for s in (101,102):
            for h in HORIZONS: rows.append(extract(run_ab(seed=s, config=duration_config(h)))[1])
        report = summarize(rows)
        self.assertEqual(report[0]['gap_change_from_12'], 0)
        self.assertAlmostEqual(report[-1]['gap_change_from_12'], report[-1]['B_minus_A']-report[0]['B_minus_A'])
        with self.assertRaises(ValueError): summarize(rows[:-1])
        with self.assertRaises(ValueError): summarize(rows + [rows[0]])

    def test_invalid_horizons(self):
        for h in (True,0,13,24.0,60):
            with self.assertRaises(ValueError): duration_config(h)

    def test_cli_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(temp)/'study'
            result = run_study(out,101,102)
            self.assertEqual(result['condition_runs'],16)
            self.assertEqual(result['prefix_checks'],12)
            self.assertEqual(len(json.loads((out/'summary.json').read_text(encoding='utf-8'))),4)
            from validation.verify_duration import verify
            audit = verify(out, write_report=False, compare_previous=False)
            self.assertEqual(audit['pair_rows'], 96)
            self.assertTrue(audit['all_snapshots_rebuilt_from_events'])
            with self.assertRaises(FileExistsError): run_study(out,101,102)


if __name__ == '__main__': unittest.main()
