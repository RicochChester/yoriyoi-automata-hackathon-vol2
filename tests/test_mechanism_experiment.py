import copy
from dataclasses import replace
import random
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from poc.ab_poc.config import load_config
from poc.ab_poc.domain import PairState, canonical_pair
from poc.ab_poc.contracts import ParticipantContext
from poc.ab_poc.engine import run_ab
from poc.ab_poc.rule_providers import RuleResponseProvider, RuleTalkInitiator
from poc.ab_poc.mechanism_experiment import (
    Case, MechanismInitiator, MechanismResponder, cases, run_case,
    behavior, observations, estimate, compact_result,
    run_study, summarize, MEASURES,
)


class MechanismTests(unittest.TestCase):
    def test_design_has_34_unique_cases_and_16_factorial_cells(self):
        design = cases()
        self.assertEqual(len(design), 34)
        self.assertEqual(len({c.id for c in design}), 34)
        self.assertEqual(sum(c.factorial for c in design), 16)
        self.assertEqual(sum(c.sweep for c in design), 20)

    def test_all_on_is_exactly_original_and_compact_can_be_analyzed(self):
        for seed in (1, 9, 66):
            original = run_ab(seed=seed)
            result = run_case(Case(1, 2, True, True), seed)
            self.assertEqual(result, original)
            self.assertEqual(json.dumps(result, ensure_ascii=False, sort_keys=True), json.dumps(original, ensure_ascii=False, sort_keys=True))
            self.assertEqual(observations(compact_result(result)), observations(result))

    def test_A_invariant_and_all_off_behavior_equal(self):
        for seed in (1, 9, 66):
            baseline = run_ab(seed=seed)
            for case in cases():
                if case.clamp:
                    continue
                result = run_case(case, seed)
                self.assertEqual(behavior(result['runs']['A']), behavior(baseline['runs']['A']))
                if case == Case(0, 0, False, False):
                    self.assertEqual(behavior(result['runs']['A']), behavior(result['runs']['B']))

    def test_both_clamps_preserve_donor_bridge_turns_and_outcomes(self):
        for seed in (1, 9, 66):
            natural = run_ab(seed=seed)
            for donor in ('A', 'B'):
                result = run_case(Case(1, 2, True, True, donor), seed, donor_result=natural)
                targets = [turn['target'] for turn in natural['runs'][donor]['turns']]
                for condition in ('A', 'B'):
                    self.assertEqual([turn['target'] for turn in result['runs'][condition]['turns']], targets)
                    self.assertEqual(behavior(result['runs'][condition], bridge_only=True),
                                     behavior(natural['runs'][donor], bridge_only=True))
                self.assertEqual(observations(result)['bridge_edge_count_delta'], 0)

    def test_contact_budget_identities(self):
        for seed in (1, 9, 66):
            row = observations(run_ab(seed=seed))
            for c in ('A', 'B'):
                self.assertEqual(row[f'known_conversations_{c}'] + row[f'bridge_conversations_{c}'] + row[f'third_party_conversations_{c}'], 12)
                self.assertEqual(row[f'known_conversations_{c}'] + row[f'endpoint_outbound_bridge_{c}'], 6)
            self.assertEqual(row['known_conversations_delta'], -row['endpoint_outbound_bridge_delta'])

    def context(self):
        config = load_config()
        people = {p.id: p for p in config.participants}
        condition = config.conditions[1]
        pair = canonical_pair('akane', 'midori')
        return ParticipantContext(
            turn=1, actor=people['akane'], candidates=tuple(p for p in config.participants if p.id != 'akane'),
            pair_states={pair: PairState(*pair, online_known=condition.online_known(*pair))},
            events=(), last_target=None, rules=config.rules, rng=random.Random(123),
            online_memories=condition.online_experience.memory_for('akane'),
        )

    def test_caution_switch_removes_only_exemption_and_preserves_rng(self):
        context = self.context()
        target = next(p for p in context.candidates if p.id == 'midori')
        normal = RuleResponseProvider().reaction_points(context, target, '技術', ())
        off_context = self.context()
        off = MechanismResponder(False).reaction_points(off_context, target, '技術', ())
        self.assertEqual(off, normal - 1)
        self.assertEqual(context.rng.getstate(), off_context.rng.getstate())
        self.assertEqual(off_context.pair_states, self.context().pair_states)

    def test_topic_switch_and_selection_bonus_are_independent(self):
        c = self.context()
        midori = next(p for p in c.candidates if p.id == 'midori')
        initiator = MechanismInitiator(False)
        self.assertEqual(RuleTalkInitiator()._choose_topic(c, midori, ()), '技術')
        self.assertIn(initiator._choose_topic(self.context(), midori, ()), ('地域', '暮らし'))
        original_weight = RuleTalkInitiator().selection_weight(c, midori)
        self.assertEqual(initiator.selection_weight(c, midori), original_weight)
        rules = replace(c.rules, selection=replace(c.rules.selection, online_known_bonus=4))
        self.assertEqual(initiator.selection_weight(replace(c, rules=rules), midori), original_weight + 3)

    def test_estimate_uses_paired_values(self):
        result = estimate([-1, 0, 1])
        self.assertEqual(result['mean'], 0)
        self.assertAlmostEqual(result['standard_error'], 1 / 3**.5)
        self.assertEqual(estimate([0] * 1000)['ci_low'], 0)
        self.assertIsNone(estimate([1])['ci_low'])

    def test_invalid_parameters_are_rejected(self):
        for args in ((-1, 2, True, True), (float('nan'), 2, True, True), (1, .5, True, True), (1, 2, True, True, 'C')):
            with self.assertRaises(ValueError):
                Case(*args)

    def test_factorial_effects_against_known_polynomial(self):
        rows = []
        for case in cases():
            s, r, t, c = case.selection, case.reaction / 2, int(case.topic), int(case.caution)
            value = 5*s + 2*r + 3*t + 4*c + 7*s*r
            for seed in (101, 102, 103):
                row = {'case_id': case.id, 'seed': seed}
                for metric in MEASURES:
                    row.update({f'{metric}_A': 0, f'{metric}_B': value, f'{metric}_delta': value})
                rows.append(row)
        _, _, effects = summarize(rows, cases())
        actual = {r['effect']: r['mean'] for r in effects if r['measure'] == 'bridge_edge_count'}
        self.assertEqual(actual, {'S': 8.5, 'R': 5.5, 'T': 3, 'C': 4, 'S*R': 7,
                                  'S*T': 0, 'S*C': 0, 'R*T': 0, 'R*C': 0, 'T*C': 0})

    def test_study_round_trip_event_reconstruction_and_overwrite_protection(self):
        from validation.verify_mechanism import verify
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / 'study'
            manifest = run_study(output, 101, 102)
            self.assertEqual(manifest['paired_runs'], 68)
            self.assertEqual(verify(output, write_report=False)['status'], 'pass')
            before = (output / 'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run_study(output, 101, 102)
            self.assertEqual((output / 'manifest.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
