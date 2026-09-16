from collections import Counter
import contextlib
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from poc.ab_poc.event_format_experiment import *
from poc.ab_poc.event_format_48_experiment import DESIGN
from validation.verify_event_formats import verify_run, audit

class Event48Tests(unittest.TestCase):
    def test_exact_positions_and_balanced_blocks(self):
        expected={'weak':[8,15,22,29,36,43], 'medium':[8,11,14,17,20,23,26,29,32,35,38,41],
            'strong':[2,4,5,7,10,11,13,16,17,19,20,22,25,26,28,31,33,35,36,38,41,43,44,46]}
        blocks={'weak':[(22,27)],'medium':[(8,13),(36,41)],'strong':[(2,7),(15,20),(28,33),(41,46)]}
        for strength,n in (('weak',1),('medium',2),('strong',4)):
            active=DESIGN.positions('shuffle_'+strength)
            task=DESIGN.positions('task_'+strength)
            self.assertEqual(list(active),expected[strength])
            self.assertEqual(list(task),[t for a,b in blocks[strength] for t in range(a,b+1)])
            for seed in range(101,1101):
                plan=contact_plan(seed,'shuffle_'+strength,active)
                work=contact_plan(seed,'task_'+strength,task)
                self.assertEqual(list(plan.values()),list(work.values()))
                self.assertEqual(Counter(canonical_pair(*v) for v in plan.values()),Counter({p:n for p in PAIRS}))
                values=list(work.values())
                for start in range(0,len(values),6):
                    self.assertEqual({canonical_pair(*v) for v in values[start:start+6]},set(PAIRS))
                if n>1:
                    self.assertEqual(values[:6],[tuple(reversed(p)) for p in values[6:12]])

    def test_all_cases_and_task_timing_partition(self):
        for p in PAIRS:
            for case in DESIGN.cases:
                raw=run_event(p,101,case,DESIGN); verify_run(raw,case,DESIGN)
                pairs,row=extract_event(raw,case,DESIGN)
                for c in ('A','B'):
                    self.assertTrue(0<=row['unformed_turns_'+c]<=48)
                    self.assertEqual(row['bridge_rate_'+c],row['bridge_count_'+c]/4)
                    if case.startswith('task_'):
                        self.assertEqual(sum(row[f'task_formed_{s}_{c}'] for s in ('before','during','between','after')),row['bridge_count_'+c])

    def test_v1_saved_traces_unchanged(self):
        for p in PAIRS:
            from tests.fixture_data import records
            for old in records('event-v1',p,101,102):
                self.assertEqual(run_event(p,old['result']['root_seed'],old['case_id']),old['result'])

    def test_task_wording_alone_does_not_change_state(self):
        settings=deepcopy(DESIGN.settings); settings.pop('task_blocks')
        same_timing=EventDesign(settings)
        for strength in same_timing.strengths:
            a=run_event(PAIRS[0],101,'shuffle_'+strength,same_timing)
            b=run_event(PAIRS[0],101,'task_'+strength,same_timing)
            for c in ('A','B'): self.assertEqual(a['runs'][c]['snapshots'],b['runs'][c]['snapshots'])

    def test_invalid_settings(self):
        for change in ({'weak_turns':[8]*6},{'weak_turns':[8,15,22,29,36,49]}, {'task_blocks':{'weak':[[22,28]],'medium':[[8,13],[36,41]],'strong':[[2,7],[15,20],[28,33],[41,46]]}}):
            settings=deepcopy(DESIGN.settings); settings.update(change)
            with self.assertRaises(ValueError): EventDesign(settings)

    def test_full_smoke(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(temp)/'study'; m=run_study(out,101,101,1,DESIGN)
            self.assertEqual(m['condition_runs'],156)
            from tests.fixture_data import ROOT as fixture_root, MANIFEST
            import hashlib
            for pair in PAIRS:
                relative='pair-identity/'+'-'.join(pair)+'/traces.jsonl.gz'
                self.assertEqual(hashlib.sha256((fixture_root/relative).read_bytes()).hexdigest(),MANIFEST[relative]['sha256'])
            r=audit(out,compare_previous=True,write_report=False,previous_root=fixture_root/'pair-identity')
            self.assertEqual(r['previous_free_AB_runs_matched'],6)
            self.assertEqual(r['pair_rows'],936)

if __name__=='__main__': unittest.main()
