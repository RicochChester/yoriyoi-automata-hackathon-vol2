import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from poc.ab_poc.pair_mechanism_experiment import *
from poc.ab_poc.mechanism_experiment import run_case
from poc.ab_poc.duration_experiment import verify_prefix

class PairMechanismTests(unittest.TestCase):
    def test_direct_runs_equal_checkpoints_all_cases_pairs(self):
        for pair in PAIRS:
            for seed in (101,103):
                _,results=one_seed((pair,seed))
                for h in (12,24,36):
                    base=run_intervention(pair,seed,CASES[0],h)
                    for case in CASES:
                        direct=base if case.id==BASE else run_intervention(pair,seed,case,h,donor=base)
                        self.assertEqual(checkpoint(results[case.id],h),direct,(pair,seed,h,case.id))
                        verify_prefix(direct,results[case.id])
    def test_legacy_mechanism_match(self):
        for case in CASES:
            old=compact_result(run_case(case,101))
            new=checkpoint(run_intervention(('akane','midori'),101,case,donor=run_intervention(('akane','midori'),101,CASES[0])),12)
            for c in ('A','B'):
                new['runs'][c]['snapshots']=[new['runs'][c]['snapshots'][-1]]
            self.assertEqual(new,old)
    def test_caution_no_effect_without_cautious_person(self):
        _,r=one_seed((('akane','kurumi'),103))
        for c in ('A','B'):
            self.assertEqual(onsite_behavior(r[BASE]['runs'][c],('akane','kurumi')),
                             onsite_behavior(r[CASES[4].id]['runs'][c],('akane','kurumi')))
    def test_invalid_donor_and_invariant_corruption(self):
        donor=run_intervention(PAIRS[0],101,CASES[0])
        with self.assertRaises(ValueError): run_intervention(PAIRS[1],101,CASES[-1],donor=donor)
        with self.assertRaises(ValueError): checkpoint(donor,60)
        _,results=one_seed((PAIRS[0],101))
        results['clamp_A']['runs']['B']['turns'][0]['topic']='corruption'
        with self.assertRaises(ValueError): check_invariants(results,PAIRS[0])
    def test_outputs_and_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)/'study'
            manifest=run_study(out,101,102,workers=1)
            self.assertEqual(manifest['pair_rows'],4608)
            self.assertEqual(len(json.loads((out/'summary.json').read_text())),192)
            self.assertEqual(len(json.loads((out/'contrasts.json').read_text())),168)
            from validation.verify_pair_mechanism import audit
            report=audit(out,compare_previous=False,write_report=False)
            self.assertEqual(report['pair_rows'],4608)
            with self.assertRaises(FileExistsError): run_study(out,101,102,1)
            with self.assertRaises(ValueError): summarize([])
    def test_worker_results_identical(self):
        tasks=[(PAIRS[0],101),(PAIRS[1],102)]
        self.assertEqual(list(bounded_results(tasks,1)),list(bounded_results(tasks,2)))
