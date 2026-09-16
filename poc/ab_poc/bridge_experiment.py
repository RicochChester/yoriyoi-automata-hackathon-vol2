"""Matched-pair analysis of the existing four-person, rule-based A/B model.

This is a read-only observer of run_ab results. It does not change simulation
rules, relationship thresholds, randomness, or the established UI metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .config import load_config
from .domain import OnlineKnown, OnsiteRelationship, canonical_pair, to_jsonable
from .engine import run_ab
from .metrics import compute_metrics


SCHEMA_VERSION = "yoriyoi.bridge_experiment.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIR_TYPES = ("known_pair", "bridge_edge", "third_party_pair")
FORMED = {OnsiteRelationship.ACQUAINTANCE.value, OnsiteRelationship.FAMILIAR.value}


def analyze_ab(result: Mapping[str, Any], *, require_same_actor_order: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a complete matched A/B run and extract all six pairs per side.

    Pair types in BOTH conditions refer to B's configured known pair. A's
    reference pair is not itself online-known. Missing states are errors,
    never inferred to mean 'not formed'.
    """
    seed = result["root_seed"]
    if type(seed) is not int:
        raise ValueError("root_seed must be an integer")
    runs = result["runs"]
    if set(runs) != {"A", "B"}:
        raise ValueError("exactly conditions A and B are required")
    people = result["participants"]
    names = {person["id"]: person["name"] for person in people}
    if len(people) != 4 or len(names) != 4:
        raise ValueError("this experiment requires four distinct participants")
    expected = {canonical_pair(*pair) for pair in combinations(names, 2)}
    known_pairs = runs["B"]["condition"]["online_known_pairs"]
    if runs["A"]["condition"]["online_known_pairs"] or len(known_pairs) != 1:
        raise ValueError("A must have no known pairs and B exactly one")
    known = canonical_pair(*known_pairs[0])
    if known not in expected:
        raise ValueError("B's known pair must belong to the participant set")

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"seed": seed}
    for condition in ("A", "B"):
        run = runs[condition]
        if run["root_seed"] != seed or run["condition"]["id"] != condition:
            raise ValueError("run seed or condition does not match the comparison")
        if run["participants"] != people:
            raise ValueError("A/B must contain identical participants")
        for field in ("rules", "turn_order", "participant_provider", "relationship_evaluator"):
            if field == "turn_order" and not require_same_actor_order:
                continue
            if run[field] != runs["A"][field]:
                raise ValueError(f"A/B must share {field}")
        final = run["final_relationships"]
        ends = [snapshot for snapshot in run["snapshots"] if snapshot["phase"] == "end"]
        if len(ends) != 1 or ends[0]["turn"] != run["rules"]["turns"]:
            raise ValueError("a completed end snapshot is required")
        if final != ends[0]["pairs"]:
            raise ValueError("final_relationships disagree with the end snapshot")
        if len(run["turns"]) != run["rules"]["turns"]:
            raise ValueError("run is incomplete")
        states = {canonical_pair(state["left_id"], state["right_id"]): state for state in final}
        if len(final) != 6 or len(states) != 6 or set(states) != expected:
            raise ValueError("each condition must contain all six pairs exactly once")
        if compute_metrics(run) != run["metrics"]:
            raise ValueError("recorded metrics disagree with the final snapshot")

        condition_rows = []
        for pair in sorted(expected):
            state = states[pair]
            stage = OnsiteRelationship(state["onsite_relationship"]).value
            online_known = condition == "B" and pair == known
            if OnlineKnown(state["online_known"]) != (OnlineKnown.DIRECT if online_known else OnlineKnown.NONE):
                raise ValueError("pair online state disagrees with the condition")
            pair_type = (
                "known_pair" if pair == known
                else "bridge_edge" if set(pair) & set(known)
                else "third_party_pair"
            )
            condition_rows.append({
                "seed": seed,
                "condition": condition,
                "pair": "-".join(names[person] for person in pair),
                "left_id": pair[0],
                "right_id": pair[1],
                "pair_type": pair_type,
                "final_relationship_stage": stage,
                "online_known": online_known,
                "formed": int(stage in FORMED),
                "familiar": int(stage == OnsiteRelationship.FAMILIAR.value),
                "conversation_count": state["conversation_count"],
                "positive_count": state["positive_count"],
            })
        rows.extend(condition_rows)
        for group, size in (("bridge_edge", 4), ("matched_new", 5), ("third_party_pair", 1), ("known_pair", 1)):
            selected = [row for row in condition_rows if
                        (row["pair_type"] != "known_pair" if group == "matched_new" else row["pair_type"] == group)]
            if len(selected) != size:
                raise ValueError(f"{group} must have {size} candidate pairs")
            count = sum(row["formed"] for row in selected)
            summary[f"{group}_count_{condition}"] = count
            summary[f"{group}_rate_{condition}"] = count / size
            summary[f"{group}_familiar_count_{condition}"] = sum(row["familiar"] for row in selected)
        for metric in ("new_acquaintance_count", "new_acquaintance_eligible_count", "new_acquaintance_rate", "new_familiar_count"):
            summary[f"legacy_{metric}_{condition}"] = run["metrics"][metric]
        summary[f"isolated_participants_{condition}"] = ",".join(run["metrics"]["isolated_participants"])
        summary[f"isolated_participant_count_{condition}"] = len(run["metrics"]["isolated_participants"])

    for group in ("bridge_edge", "matched_new", "third_party_pair", "known_pair"):
        for measure in ("count", "rate", "familiar_count"):
            key = f"{group}_{measure}"
            summary[f"{key}_B_minus_A"] = summary[f"{key}_B"] - summary[f"{key}_A"]
    # The requested short name always means a COUNT difference, not a rate.
    summary["B_minus_A"] = summary["bridge_edge_count_B_minus_A"]
    return rows, summary


def aggregate(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive paired-seed statistics; edges are not independent samples."""
    if not summaries or len({row["seed"] for row in summaries}) != len(summaries):
        raise ValueError("summaries must have nonempty, unique seeds")
    numeric = [key for key, value in summaries[0].items() if key != "seed" and type(value) in (int, float)]
    differences = [row["B_minus_A"] for row in summaries]
    return {
        "seed_count": len(summaries),
        "means": {key: mean(row[key] for row in summaries) for key in numeric},
        "totals": {key: sum(row[key] for row in summaries) for key in numeric if "count" in key and "rate" not in key},
        "bridge_paired_seeds": {
            "B_greater_than_A": sum(value > 0 for value in differences),
            "B_equals_A": sum(value == 0 for value in differences),
            "B_less_than_A": sum(value < 0 for value in differences),
        },
        "bridge_difference_distribution": {str(value): differences.count(value) for value in sorted(set(differences))},
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    # BOM allows Excel on Windows to recognize Japanese as UTF-8.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(seeds: Sequence[int], output: Path) -> dict[str, Any]:
    """Run the UI's default rules once per condition per seed, without APIs.

    Existing output directories are never overwritten. manifest.json is
    written last: its presence and checksums mark a completed experiment.
    """
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty, unique integers")
    config = load_config()
    config_payload = to_jsonable(config)
    source_files = sorted((PROJECT_ROOT / "poc").rglob("*.py")) + sorted((PROJECT_ROOT / "poc" / "config").glob("*.json"))
    source_hashes = {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in source_files}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, summaries = [], []
    with (output / "raw_runs.jsonl").open("w", encoding="utf-8", newline="\n") as raw:
        for seed in seeds:
            result = run_ab(seed=seed, config=config)
            pair_rows, summary = analyze_ab(result)
            raw.write(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            rows.extend(pair_rows)
            summaries.append(summary)
            print(f"seed {seed}: Bridge A={summary['bridge_edge_count_A']}/4 B={summary['bridge_edge_count_B']}/4 B-A={summary['B_minus_A']:+d}", flush=True)
    if any(_sha256(PROJECT_ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("source changed during the experiment; results are incomplete")
    totals = aggregate(summaries)
    _write_csv(output / "pairs.csv", rows)
    _write_json(output / "pairs.json", rows)
    _write_csv(output / "seed_summary.csv", summaries)
    bridge_columns = ("seed", "bridge_edge_count_A", "bridge_edge_count_B",
                      "bridge_edge_rate_A", "bridge_edge_rate_B", "B_minus_A",
                      "bridge_edge_rate_B_minus_A")
    _write_csv(output / "bridge_summary.csv", [{key: row[key] for key in bridge_columns} for row in summaries])
    _write_json(output / "summary.json", totals)
    _write_json(output / "config.json", config_payload)
    lines: list[str] = []
    previous = None
    for row in rows:
        current = (row["seed"], row["condition"])
        if current != previous:
            lines.append(f"\nseed = {row['seed']} / Condition {row['condition']}\n")
            previous = current
        lines.append(f"{row['pair']}  {row['final_relationship_stage']}  [{row['pair_type']}]\n")
    (output / "pairs.txt").write_text("".join(lines), encoding="utf-8-sig")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "seeds": list(seeds),
        "conditions": ["A", "B"],
        "trials_per_condition_per_seed": 1,
        "provider": "rule-participant",
        "turns": config.rules.turns,
        "reference_known_pair": list(next(condition for condition in config.conditions if condition.id == "B").online_known_pairs[0]),
        "candidate_counts": {"known_pair": 1, "bridge_edge": 4, "third_party_pair": 1, "matched_new": 5},
        "formed_stages": sorted(FORMED),
        "pair_type_reference": "B's known pair, applied to both A and B; A has no online-known pair",
        "B_minus_A_unit": "bridge edge count",
        "interpretation": "Descriptive results of the configured artificial society; no claim about human causal effects or a mediated mechanism.",
        "source_sha256": source_hashes,
        "output_sha256": {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    provenance = PROJECT_ROOT / "UPSTREAM.json"
    if provenance.is_file():
        manifest["upstream"] = json.loads(provenance.read_text(encoding="utf-8"))
    _write_json(output / "manifest.json", manifest)
    return totals


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export matched A/B pairs using the existing rule model (no API calls).")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=100, help="inclusive")
    parser.add_argument("--output", type=Path, help="new directory; existing directories are refused")
    args = parser.parse_args(argv)
    if args.seed_end < args.seed_start:
        parser.error("--seed-end must be >= --seed-start")
    output = args.output or PROJECT_ROOT / "results" / datetime.now().strftime("bridge-%Y%m%d-%H%M%S-%f")
    if output.exists():
        parser.error(f"output already exists; choose a NEW directory: {output}")
    try:
        totals = run_experiment(range(args.seed_start, args.seed_end + 1), output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Experiment failed: {error}\nInspect partial output at {output}; no complete manifest means unfinished.\n")
    print(f"Completed {totals['seed_count']} seeds. Results: {output.resolve()}")
    print(f"Mean bridge B-A: {totals['means']['B_minus_A']:+.3f} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
