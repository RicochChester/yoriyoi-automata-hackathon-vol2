import unittest
from copy import deepcopy
from poc.ab_poc.early_recontact_experiment import *
from poc.ab_poc.event_format_experiment import run_event
from validation.verify_early_recontact import verify

class EarlyRecontactTest(unittest.TestCase):
    def test_neutral_matches_existing_model(self):
        for pair in PAIRS:
            actual=simulate(pair,101,'neutral'); base=run_event(pair,101,'task_weak',DESIGN)
            verify(actual,'neutral')
            for c in ('A','B'):
                for field in ('turns','snapshots','rules','condition'):
                    self.assertEqual(actual['runs'][c][field],base['runs'][c][field])
    def test_prefix_and_all_pairs(self):
        for pair in PAIRS:
            records,rows=one((pair,102))
            self.assertEqual(len(rows),15)
            for record in records:
                for c in ('A','B'):
                    counts=Counter(canonical_pair(t['actor'],t['target']) for t in record['result']['runs'][c]['turns'][:6])
                    self.assertEqual(counts,Counter({p:1 for p in PAIRS}))
    def test_no_information_negative_control(self):
        first=None
        for case in CASES:
            raw=simulate(PAIRS[0],103,case,novel=False); verify(raw,case,novel=False)
            social={c:(r['turns'],r['snapshots']) for c,r in raw['runs'].items()}
            if first is not None: self.assertEqual(first,social)
            first=social
    def test_corruption_detected(self):
        raw=simulate(PAIRS[0],104,'actual_promote'); verify(raw,'actual_promote')
        corrupt=deepcopy(raw)
        entry=corrupt['runs']['B']['event_log'][9]; next(iter(entry['selection'].values()))['weight']+=.1
        with self.assertRaises(AssertionError): verify(corrupt,'actual_promote')
        corrupt=deepcopy(raw); corrupt['runs']['A']['turns'][10]['topic']='改変'
        with self.assertRaises(AssertionError): verify(corrupt,'actual_promote')
        corrupt=deepcopy(raw); corrupt['runs']['B']['event_log'][0]['transferred_token']=999
        with self.assertRaises(AssertionError): verify(corrupt,'actual_promote')
    def test_metrics_match_direct_endpoints(self):
        raw=simulate(PAIRS[1],105,'permuted_suppress'); verify(raw,'permuted_suppress')
        known=set(PAIRS[1])
        for row in metrics(raw,'permuted_suppress'):
            for c,r in raw['runs'].items():
                pairs=[p for p in r['snapshots'][row['horizon']]['pairs'] if len({p['left_id'],p['right_id']}&known)==1]
                self.assertEqual(sum(p['onsite_relationship']!='未形成' for p in pairs),row['bridge_count_'+c])
                self.assertEqual(row['bridge_rate_'+c],row['bridge_count_'+c]/4)

if __name__=='__main__': unittest.main()
