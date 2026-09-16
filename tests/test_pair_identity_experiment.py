import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from poc.ab_poc.pair_identity_experiment import PAIRS, pair_config, run_study, interval
from poc.ab_poc.duration_experiment import duration_config, extract
from poc.ab_poc.engine import run_ab
from validation.verify_duration import verify

class PairIdentityTests(unittest.TestCase):
    def test_only_pair_metadata_changes(self):
        self.assertEqual(len(set(PAIRS)),6)
        for pair in PAIRS:
            for h in (12,24,36,48):
                base=duration_config(h)
                cfg=pair_config(pair,h)
                self.assertEqual(replace(cfg,conditions=base.conditions),base)
                self.assertEqual(cfg.conditions[0],base.conditions[0])
                b=cfg.conditions[1]
                self.assertEqual(b.online_known_pairs,(pair,))
                self.assertEqual(b.online_experience.participants,pair)
                self.assertEqual(replace(b.online_experience,id=base.conditions[1].online_experience.id,
                    participants=base.conditions[1].online_experience.participants),base.conditions[1].online_experience)
        self.assertEqual(pair_config(('midori','akane'),12),duration_config(12))
        with self.assertRaises(ValueError): pair_config(('akane','akane'),12)
    def test_classification_and_A_invariance(self):
        reference=None
        for p in PAIRS:
            r=run_ab(seed=103,config=pair_config(p,24))
            if reference is not None: self.assertEqual(r['runs']['A'],reference)
            reference=r['runs']['A']
            rows,_=extract(r)
            for c in ('A','B'):
                self.assertEqual([sum(x['pair_type']==t and x['condition']==c for x in rows)
                    for t in ('known_pair','bridge_edge','third_party_pair')],[1,4,1])
    def test_output_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)/'study'
            m=run_study(out,101,102)
            self.assertEqual(m['condition_runs'],96)
            for p in PAIRS:
                a=verify(out/'-'.join(p),write_report=False,compare_previous=False)
                self.assertEqual(a['pair_rows'],96)
            with self.assertRaises(FileExistsError): run_study(out,101,102)
    def test_adjusted_interval(self):
        a,b=interval([-1,0,1],24),interval([-1,0,1],60)
        self.assertEqual(a['mean'],0)
        self.assertGreater(b['simultaneous_high'],a['simultaneous_high'])
