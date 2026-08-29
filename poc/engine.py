from __future__ import annotations

import json
import random
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from .models import (
    Candidate,
    Decision,
    DecisionContext,
    ExperimentPreset,
    ParticipantState,
    Persona,
    Relationship,
    Scenario,
    TurnRecord,
)
from .providers import (
    ConversationProvider,
    DecisionProvider,
    OllamaDecisionProvider,
    RuleBasedDecisionProvider,
    TemplateConversationProvider,
)


CONFIG_PATH = Path(__file__).parent / "config" / "scenarios.json"
MAP_POINTS = (
    (12, 14), (31, 14), (50, 14), (20, 36), (41, 36), (60, 36),
    (12, 54), (31, 54), (50, 54), (60, 54),
)
MEETING_POINTS = ((29, 23), (42, 23), (35, 34))


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_personas(path: Path = CONFIG_PATH) -> dict[str, Persona]:
    config = load_config(path)
    return {
        row["id"]: Persona(
            id=row["id"],
            name=row["name"],
            interests=tuple(row["interests"]),
            visit_purpose=row["visit_purpose"],
            approachability=int(row["approachability"]),
            receptivity=int(row["receptivity"]),
            conversation_style=row["conversation_style"],
            outfit_color=row["outfit_color"],
        )
        for row in config["participants"]
    }


def load_scenarios(path: Path = CONFIG_PATH) -> dict[str, Scenario]:
    config = load_config(path)
    scenarios: dict[str, Scenario] = {}
    for venue in config["venues"]:
        for relation_mode in config["relationship_modes"]:
            scenario_id = f"{venue['id']}__{relation_mode['id']}"
            scenarios[scenario_id] = Scenario(
                id=scenario_id,
                label=f"{venue['label']} × {relation_mode['label']}",
                venue_id=venue["id"],
                venue_label=venue["label"],
                venue_description=venue["description"],
                relationship_mode_id=relation_mode["id"],
                relationship_mode_label=relation_mode["label"],
                arrival_steps=tuple(int(value) for value in venue["arrival_steps"]),
                conversation_bonus=int(venue["conversation_bonus"]),
                conversation_start_step=int(venue["conversation_start_step"]),
                shared_interest_bonus=int(venue["shared_interest_bonus"]),
                unrelated_pair_penalty=int(venue["unrelated_pair_penalty"]),
                venue_topic_is_primary=bool(venue["venue_topic_is_primary"]),
                venue_topic=venue["topic"],
                known_pairs=tuple(tuple(pair) for pair in relation_mode["known_pairs"]),
            )
    # Most scenarios intentionally remain the venue × relationship-mode
    # matrix above.  A shared online community is a single explicit scenario,
    # rather than another axis that would multiply every physical venue.
    for row in config.get("additional_scenarios", []):
        scenario_id = row["id"]
        if scenario_id in scenarios:
            raise ValueError(f"Duplicate scenario id: {scenario_id}")
        scenarios[scenario_id] = Scenario(
            id=scenario_id,
            label=row["label"],
            venue_id=row["venue_id"],
            venue_label=row["venue_label"],
            venue_description=row["venue_description"],
            relationship_mode_id=row["relationship_mode_id"],
            relationship_mode_label=row["relationship_mode_label"],
            arrival_steps=tuple(int(value) for value in row["arrival_steps"]),
            conversation_bonus=int(row["conversation_bonus"]),
            conversation_start_step=int(row["conversation_start_step"]),
            shared_interest_bonus=int(row["shared_interest_bonus"]),
            unrelated_pair_penalty=int(row["unrelated_pair_penalty"]),
            venue_topic_is_primary=bool(row["venue_topic_is_primary"]),
            venue_topic=row["topic"],
            known_pairs=tuple(tuple(pair) for pair in row["known_pairs"]),
            online_community=row.get("online_community"),
        )
    return scenarios


def load_experiment_presets(path: Path = CONFIG_PATH) -> dict[str, ExperimentPreset]:
    config = load_config(path)
    presets: dict[str, ExperimentPreset] = {}
    for row in config["experiment_presets"]:
        participant_ids = tuple(row["participant_ids"])
        arrival_steps_by_venue = {
            venue_id: tuple(int(value) for value in arrival_steps)
            for venue_id, arrival_steps in row["arrival_steps_by_venue"].items()
        }
        if len(participant_ids) == 0:
            raise ValueError(f"Experiment preset {row['id']} has no participants")
        if int(row["steps"]) <= 0:
            raise ValueError(f"Experiment preset {row['id']} must have at least one step")
        if any(len(arrivals) != len(participant_ids) for arrivals in arrival_steps_by_venue.values()):
            raise ValueError(f"Experiment preset {row['id']} arrival steps must match participants")
        presets[row["id"]] = ExperimentPreset(
            id=row["id"],
            label=row["label"],
            description=row["description"],
            participant_ids=participant_ids,
            steps=int(row["steps"]),
            arrival_steps_by_venue=arrival_steps_by_venue,
        )
    return presets


class Simulation:
    def __init__(
        self,
        scenario: Scenario,
        personas: dict[str, Persona],
        *,
        seed: int,
        decision_provider: DecisionProvider | None = None,
        conversation_provider: ConversationProvider | None = None,
        steps: int = 8,
        preset_id: str = "poc",
    ) -> None:
        self.scenario = scenario
        self.personas = personas
        self.seed = seed
        self.random = random.Random(seed)
        self.decision_provider = decision_provider or RuleBasedDecisionProvider()
        self.conversation_provider = conversation_provider or TemplateConversationProvider()
        self.steps = steps
        self.preset_id = preset_id
        self.participant_ids = tuple(personas)
        self.states = {
            participant_id: ParticipantState(participant_id, MAP_POINTS[index])
            for index, participant_id in enumerate(self.participant_ids)
        }
        known = {tuple(sorted(pair)) for pair in scenario.known_pairs}
        self.relationships = {
            pair: Relationship(
                pair[0],
                pair[1],
                known_before_visit=pair in known,
                pre_existing_online_connection=pair in known,
            )
            for pair in combinations(sorted(self.participant_ids), 2)
        }

    def run(self) -> dict:
        turns: list[TurnRecord] = []
        for step in range(1, self.steps + 1):
            arrivals = self._arrive(step)
            active_ids = self._active_ids()
            actor_id = self._choose_actor(active_ids, step)
            decision = self._decide(actor_id, active_ids, step) if actor_id else None
            conversation = self._apply(actor_id, decision) if actor_id and decision else None
            turns.append(
                TurnRecord(
                    step=step,
                    actor_id=actor_id or "",
                    decision=decision or self._empty_decision(),
                    conversation=conversation,
                    positions={key: value.position for key, value in self.states.items()},
                    relations=self._relations_snapshot(),
                    metrics=self._metrics(),
                    arrivals=arrivals,
                )
            )
        return {
            "schema_version": 1,
            "scenario": self.scenario.public_dict(),
            "seed": self.seed,
            "providers": {
                "decision": self.decision_provider.name,
                "conversation": self.conversation_provider.name,
            },
            "metadata": {
                "provider": {
                    "selected": self.decision_provider.name,
                    "fallback_count": getattr(self.decision_provider, "fallback_count", 0),
                }
            },
            "participants": [persona.public_dict() for persona in self.personas.values()],
            "turns": [turn.public_dict() for turn in turns],
            "metrics": self._metrics(),
        }

    def _arrive(self, step: int) -> list[str]:
        arrivals: list[str] = []
        for index, participant_id in enumerate(self.participant_ids):
            state = self.states[participant_id]
            if not state.arrived and self.scenario.arrival_steps[index] <= step:
                state.arrived = True
                arrivals.append(participant_id)
        return arrivals

    def _active_ids(self) -> list[str]:
        return [participant_id for participant_id in self.participant_ids if self.states[participant_id].arrived]

    def _choose_actor(self, active_ids: list[str], step: int) -> str | None:
        if len(active_ids) < 2:
            return None
        return active_ids[(step - 1) % len(active_ids)]

    def _decide(self, actor_id: str, active_ids: list[str], step: int):
        if step < self.scenario.conversation_start_step:
            return Decision("step_back", None, "共通のプログラムを観察している")

        actor = self.personas[actor_id]
        candidates: list[Candidate] = []
        for candidate_id in active_ids:
            if candidate_id == actor_id:
                continue
            target = self.personas[candidate_id]
            relation = self._relation(actor_id, candidate_id)
            shared = tuple(sorted(set(actor.interests) & set(target.interests)))
            venue_affordance = (
                self.scenario.shared_interest_bonus
                if shared
                else -self.scenario.unrelated_pair_penalty
            )
            score = (
                len(shared) * 3
                + actor.approachability
                + target.receptivity
                + self.scenario.conversation_bonus
                + venue_affordance
                + (2 if relation.known_before_visit else 0)
                + relation.strength
                + self.random.randint(0, 2)
            )
            candidates.append(
                Candidate(
                    participant_id=candidate_id,
                    name=target.name,
                    score=score,
                    shared_interests=shared,
                    relation_strength=relation.strength,
                    interests=target.interests,
                    visit_purpose=target.visit_purpose,
                    conversation_style=target.conversation_style,
                )
            )
        return self.decision_provider.decide(
            DecisionContext(step=step, actor=actor, candidates=tuple(candidates), scenario=self.scenario)
        )

    def _apply(self, actor_id: str, decision):
        actor_state = self.states[actor_id]
        if decision.action == "step_back" or decision.target_id is None:
            actor_state.last_partner_id = None
            actor_state.position = MAP_POINTS[self.random.randrange(len(MAP_POINTS))]
            return None

        target_id = decision.target_id
        relation = self._relation(actor_id, target_id)
        shared = tuple(sorted(set(self.personas[actor_id].interests) & set(self.personas[target_id].interests)))
        relation.talk_count += 1
        relation.onsite_talk_count += 1
        relation.onsite_relationship_formed = True
        if decision.action == "continue":
            relation.continuation_count += 1
            relation.strength = 3
        else:
            relation.strength = max(2, relation.strength)

        meeting_point = MEETING_POINTS[(relation.talk_count - 1) % len(MEETING_POINTS)]
        self.states[actor_id].position = meeting_point
        self.states[target_id].position = (meeting_point[0] + 5, meeting_point[1])
        self.states[actor_id].last_partner_id = target_id
        self.states[target_id].last_partner_id = actor_id
        conversation_interests = () if self.scenario.venue_topic_is_primary else shared
        return self.conversation_provider.create(
            actor=self.personas[actor_id],
            target=self.personas[target_id],
            decision=decision,
            shared_interests=conversation_interests,
            venue_topic=self.scenario.venue_topic,
        )

    def _relations_snapshot(self) -> list[dict]:
        return [relation.public_dict() for relation in self.relationships.values() if relation.strength > 0]

    def _metrics(self) -> dict:
        edges = [relation for relation in self.relationships.values() if relation.strength >= 2]
        continuations = [relation for relation in self.relationships.values() if relation.continuation_count > 0]
        onsite_new_relationships = [
            relation
            for relation in self.relationships.values()
            if relation.onsite_talk_count > 0
            and not (relation.pre_existing_online_connection or relation.known_before_visit)
        ]
        online_to_onsite_activations = [
            relation
            for relation in self.relationships.values()
            if relation.onsite_talk_count > 0
            and (relation.pre_existing_online_connection or relation.known_before_visit)
        ]
        degrees = {participant_id: 0 for participant_id in self.participant_ids}
        for relation in edges:
            degrees[relation.left_id] += 1
            degrees[relation.right_id] += 1
        total_degree = sum(degrees.values())
        return {
            "relationship_count": len(edges),
            "relationship_continuity": round(
                sum(relation.strength for relation in edges) / len(edges), 2
            ) if edges else 0,
            "continued_relationship_count": len(continuations),
            "onsite_new_relationship_count": len(onsite_new_relationships),
            "online_to_onsite_activation_count": len(online_to_onsite_activations),
            "relationship_bias": round(max(degrees.values()) / total_degree, 2) if total_degree else 0,
            "degree_by_participant": degrees,
        }

    def _relation(self, left_id: str, right_id: str) -> Relationship:
        return self.relationships[tuple(sorted((left_id, right_id)))]

    @staticmethod
    def _empty_decision():
        return Decision("step_back", None, "来訪者がそろっていない")


def run_scenario(
    scenario_id: str,
    *,
    seed: int = 1,
    steps: int | None = None,
    provider: str = "rule_based",
    preset: str = "poc",
) -> dict:
    scenarios = load_scenarios()
    if scenario_id not in scenarios:
        raise KeyError(f"Unknown scenario: {scenario_id}")
    decision_providers: dict[str, DecisionProvider] = {
        "rule_based": RuleBasedDecisionProvider(),
        "ollama": OllamaDecisionProvider(),
    }
    if provider not in decision_providers:
        raise ValueError(f"Unknown provider: {provider}")
    presets = load_experiment_presets()
    if preset not in presets:
        raise ValueError(f"Unknown experiment preset: {preset}")
    experiment_preset = presets[preset]
    if scenarios[scenario_id].venue_id not in experiment_preset.arrival_steps_by_venue:
        raise ValueError(
            f"Experiment preset {preset} has no arrival steps for {scenarios[scenario_id].venue_id}"
        )
    personas = load_personas()
    missing_personas = set(experiment_preset.participant_ids) - set(personas)
    if missing_personas:
        raise ValueError(f"Experiment preset {preset} has unknown participants: {sorted(missing_personas)}")
    scenario = replace(
        scenarios[scenario_id],
        arrival_steps=experiment_preset.arrival_steps_by_venue[scenarios[scenario_id].venue_id],
    )
    result = Simulation(
        scenario,
        {participant_id: personas[participant_id] for participant_id in experiment_preset.participant_ids},
        seed=seed,
        steps=steps if steps is not None else experiment_preset.steps,
        decision_provider=decision_providers[provider],
        preset_id=preset,
    ).run()
    result["metadata"]["experiment_preset"] = experiment_preset.public_dict()
    return result


def list_scenarios() -> list[dict]:
    return [scenario.public_dict() for scenario in load_scenarios().values()]


def list_experiment_presets() -> list[dict]:
    return [preset.public_dict() for preset in load_experiment_presets().values()]
