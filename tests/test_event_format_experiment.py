import contextlib
from dataclasses import replace
import io
from pathlib import Path
import random
import tempfile
import unittest
from poc.ab_poc.event_format_experiment import *
from poc.ab_poc.contracts import ParticipantContext
from validation.verify_event_formats import verify_run, audit, clean_baseline

class EventFormatTests(unittest.TestCase):
    def test_schedules_all_seeds_balanced(self):
        for seed in range(101,1101):
            for strength,n in (('weak',1),('strong',2)):
                p=contact_plan(seed,'shuffle_'+strength)
                self.assertEqual(Counter(canonical_pair(*v) for v in p.values()),Counter({pair:n for pair in PAIRS}))
                self.assertEqual(p,contact_plan(seed,'task_'+strength))
        self.assertEqual(positions('shuffle_weak'),(4,8,12,16,20,24))
    def test_all_cases_states_and_unchanged_baseline(self):
        for pair in PAIRS:
            for case in CASES:
                raw=run_event(pair,101,case); verify_run(raw,case)
                if case=='free':
                    full=run_ab(seed=101,config=pair_config(pair,24))
                    old=compact_result(full)
                    for c in ('A','B'): old['runs'][c]['snapshots']=full['runs'][c]['snapshots']
                    self.assertEqual(clean_baseline(raw),old)
    def test_host_bonus_fixed_and_temporary(self):
        cfg=pair_config(PAIRS[0],24); actor,target=cfg.participants[:2]
        context=ParticipantContext(4,actor,tuple(p for p in cfg.participants if p!=actor),{},(),target.id,cfg.rules,random.Random(1))
        for case in ('host_weak','host_strong'):
            provider=EventProvider(101,case,cfg.rules.common_topic)
            normal=RuleTalkInitiator().selection_weight(context,target)
            provider.introduced=canonical_pair(actor.id,target.id)
            self.assertAlmostEqual(provider.selection_weight(context,target)-normal,SETTINGS['host_selection_bonus']*cfg.rules.selection.repeated_target_multiplier)
            provider.select_event_actor(5,actor.id,cfg.participants,())
            self.assertEqual(provider.selection_weight(context,target),normal)
    def test_shared_override_uses_existing_rules_and_resets(self):
        cfg=pair_config(PAIRS[0],24); actor=cfg.participants[2]; target=cfg.participants[0]
        context=ParticipantContext(4,actor,tuple(p for p in cfg.participants if p!=actor),{},(),None,cfg.rules,random.Random(7))
        provider=EventProvider(101,'topic_weak',cfg.rules.common_topic)
        self.assertFalse(provider.initiator.shared_interests(actor,target))
        provider.topic_active=True
        self.assertTrue(provider.initiator.shared_interests(actor,target))
        self.assertEqual(provider.selection_weight(context,target)-RuleTalkInitiator().selection_weight(context,target),cfg.rules.selection.shared_interest_bonus)
        x=provider.reaction_points(replace(context,rng=random.Random(7)),target,cfg.rules.common_topic,())
        y=RuleResponseProvider().reaction_points(replace(context,rng=random.Random(7)),target,cfg.rules.common_topic,())
        self.assertEqual(x-y,-cfg.rules.reaction.cautious_actor_penalty)
        provider.select_event_actor(4,actor.id,cfg.participants,())
        provider.propose(context)
        self.assertFalse(provider.topic_active)
        self.assertFalse(provider.initiator.shared_interests(actor,target))
    def test_actor_hook_rejects_unknown_actor(self):
        class Bad(RuleParticipantProvider):
            def select_event_actor(self,*args): return 'unknown'
        with self.assertRaises(ValueError): run_ab(seed=101,participant_provider_factory=Bad)
    def test_full_smoke_audit_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(temp)/'study'; m=run_study(out,101,102,1)
            self.assertEqual(m['condition_runs'],216)
            a=audit(out,compare_previous=False,write_report=False)
            self.assertEqual(a['pair_rows'],1296)
            with self.assertRaises(FileExistsError): run_study(out,101,102,1)
            with self.assertRaises(ValueError): summarize_event([])
