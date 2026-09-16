import unittest
from copy import deepcopy
from poc.ab_poc.post_task_recontact_experiment import inputs, PAIRS, CASE, simulate, verify, one

from tests.fixture_data import inputs

def donor(pair): return next(d for c,d,_ in inputs('-'.join(pair),101,101) if c==CASE)

class PostTaskTests(unittest.TestCase):
    def test_all_pairs_and_zero_matches_unmodified_free_initiator(self):
        for pair in PAIRS:
            d=donor(pair); outputs=one(d)
            standard=simulate(d,0,ordinary_free=True)
            for c in ('A','B'):
                zero=outputs[1][0]['result']['runs'][c]
                self.assertEqual(zero['turns'],standard['runs'][c]['turns'])
                self.assertEqual(zero['snapshots'],standard['runs'][c]['snapshots'])
                for _,row in outputs: self.assertEqual(row['bridge_before_'+c],outputs[1][1]['bridge_before_'+c])

    def test_no_information_exposure_means_no_theta_effect(self):
        d=donor(PAIRS[0]); results=[simulate(d,t,novel=False) for t in (-1,0,1)]
        for t,r in zip((-1,0,1),results): verify(r,d,t,novel=False)
        for c in ('A','B'):
            self.assertEqual(results[0]['runs'][c]['turns'],results[1]['runs'][c]['turns'])
            self.assertEqual(results[2]['runs'][c]['snapshots'],results[1]['runs'][c]['snapshots'])

    def test_audit_rejects_bad_selection_weight(self):
        d=donor(PAIRS[0]); raw=simulate(d,1)
        e=raw['runs']['A']['event_log'][48]; next(iter(e['selection'].values()))['weight']=999
        with self.assertRaises(AssertionError): verify(raw,d,1)

if __name__=='__main__': unittest.main()
