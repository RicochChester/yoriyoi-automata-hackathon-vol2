import unittest
from copy import deepcopy
from poc.ab_poc.task_need_matched_experiment import one, inputs, PAIRS, verify, verify_matched, InformationState

from tests.fixture_data import inputs

class MatchedNeedTests(unittest.TestCase):
    def test_all_pairs_strengths_social_identity_and_information_difference(self):
        for pair in PAIRS:
            for item in inputs('-'.join(pair),101,101):
                result=one(item)
                for c in ('A','B'):
                    familiar,novel=[r[1] for r in result]
                    self.assertEqual(familiar['confirmation_requests_'+c],0)
                    self.assertGreater(novel['confirmation_requests_'+c],0)
                    self.assertGreater(novel['information_transfers_'+c],0)
                    for m in ('bridge_count','bridge_rate','unformed_turns','known_share','bridge_share','all_formed'):
                        self.assertEqual(familiar[m+'_'+c],novel[m+'_'+c])

    def test_information_cannot_be_acquired_from_someone_without_it(self):
        state=InformationState(['akane','koharu','midori','kurumi'],101,True)
        a,b=list(state.known)[:2]; token,available=state.query(a,b)
        self.assertTrue(available); state.receive(a,b,token)
        token,available=state.query(a,b)
        self.assertFalse(available)
        with self.assertRaises(ValueError): state.receive(a,b,token)

    def test_probability_and_information_tampering_rejected(self):
        case,donor,previous=next(inputs('-'.join(PAIRS[0]),101,101))
        result=one((case,donor,previous))
        for key in ('response_probabilities','transferred_token'):
            raw=deepcopy(result[1][0]['result']); raw['runs']['A']['event_log'][0][key]=999
            with self.assertRaises(AssertionError): verify(raw,donor,case,True)
        raw=deepcopy(result[1][0]['result']); raw['runs']['A']['event_log'][0]['jitter']=999
        with self.assertRaises(AssertionError): verify_matched(result[0][0]['result'],raw)

if __name__=='__main__': unittest.main()
