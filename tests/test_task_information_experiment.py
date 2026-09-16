import gzip
import json
import unittest
from poc.ab_poc.task_information_experiment import InformationState, BASELINE, PAIRS, simulate, verify, one_record

from tests.fixture_data import records

def donor_records(pair=PAIRS[0],seed=101):
    yield from records('event-v2',pair,seed,seed)

class InformationTests(unittest.TestCase):
    def test_information_transfer_is_real_and_bounded(self):
        ids=['akane','koharu','midori','kurumi']; s=InformationState(ids,101,True)
        self.assertEqual(set(s.snapshot()['task_familiarity'].values()),{0})
        a,b=ids[:2]; token,available=s.query(a,b)
        self.assertTrue(available); s.receive(a,b,token)
        self.assertEqual(len(s.known[a]),2)
        with self.assertRaises(ValueError): s.receive(a,b,token)
        self.assertFalse(s.snapshot()['complete'])
        self.assertEqual(InformationState(ids,101,False).snapshot()['task_progress'],1)

    def test_all_pairs_strengths_replay_and_information_audit(self):
        for pair in PAIRS:
            for r in donor_records(pair):
                results=one_record((r['case_id'],r['result']))
                for c in ('A','B'):
                    self.assertEqual(results[0][2]['known_share_'+c],results[1][2]['known_share_'+c])
                    self.assertEqual(results[0][2]['bridge_share_'+c],results[1][2]['bridge_share_'+c])

    def test_information_without_topic_link_cannot_change_relationship(self):
        for r in donor_records():
            raw=simulate(r['result'],r['case_id'],True,topic_link=False)
            for c in ('A','B'):
                self.assertEqual(raw['runs'][c]['snapshots'],r['result']['runs'][c]['snapshots'])
                self.assertGreater(raw['runs'][c]['information']['final']['task_progress'],0)

if __name__=='__main__': unittest.main()
