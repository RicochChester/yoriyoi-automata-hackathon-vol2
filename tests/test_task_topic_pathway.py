import unittest
import random
from copy import deepcopy
from poc.ab_poc.task_topic_pathway_experiment import inputs, one, verify, PAIRS, TOPICS
from poc.ab_poc.task_information_experiment import clone_rng

from tests.fixture_data import inputs

class TopicPathwayTests(unittest.TestCase):
    def test_rng_state_copy_matches_deepcopy(self):
        for seed in range(100):
            rng=random.Random(seed); rng.gauss(0,1)
            a,b=clone_rng(rng),deepcopy(rng)
            self.assertEqual(a.getstate(),b.getstate())
            for size in (1,2,3,4,6,17): self.assertEqual(a.randrange(size),b.randrange(size))
            self.assertEqual(a.gauss(0,1),b.gauss(0,1))

    def test_every_pair_strength_profile_and_previous_reproduction(self):
        for pair in PAIRS:
            for case,donor,previous in inputs('-'.join(pair),101,101):
                outputs=one((case,donor,previous))
                for rec,_,row in outputs:
                    for c in ('A','B'):
                        self.assertEqual(row['known_share_'+c],outputs[0][2]['known_share_'+c])
                        self.assertEqual(row['bridge_share_'+c],outputs[0][2]['bridge_share_'+c])
                        self.assertEqual([e['jitter'] for e in rec['result']['runs'][c]['event_log']],
                                         [e['jitter'] for e in outputs[0][0]['result']['runs'][c]['event_log']])
                        if rec['profile']=='freeze_TR': self.assertEqual(row['other_positive_difference_'+c],0)

    def test_audit_rejects_corrupted_reaction_and_information(self):
        case,donor,previous=next(inputs('-'.join(PAIRS[0]),101,101))
        rec,_,_=one((case,donor,previous))[0]
        for field in ('reaction_points','transferred_token'):
            raw=deepcopy(rec['result']); raw['runs']['A']['event_log'][0][field]=999
            with self.assertRaises(AssertionError): verify(raw,donor,previous,case,'original')

    def test_outside_topics_are_not_person_interests(self):
        _,donor,_=next(inputs('-'.join(PAIRS[0]),101,101))
        for p in donor['participants']: self.assertFalse(set(p['interests'])&set(TOPICS['outside']))

if __name__=='__main__': unittest.main()
