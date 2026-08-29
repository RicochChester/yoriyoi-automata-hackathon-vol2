from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .ab_history import load_history, list_history, save_history
from .ab_poc import (
    M51SplitParticipantProvider,
    DeepSeekUtteranceGenerator,
    DeepSeekRuleCandidateInitiator,
    DeepSeekNaturalConversationInitiator,
    DeepSeekNaturalResponseProvider,
    DeepSeekAutonomousConversationInitiator,
    DeepSeekAutonomousResponseProvider,
    DeepSeekNarrativeProvider,
    OllamaParticipantProvider,
    OllamaTalkInitiator,
    RuleResponseProvider,
    RuleTalkInitiator,
    SplitConversationProvider,
    TwoStageTalkInitiator,
    run_ab,
    run_condition,
)
from .engine import list_experiment_presets, list_scenarios, run_scenario
from .experiments import ExperimentSpec, run_experiment


WEB_ROOT = Path(__file__).parent / "web"
PHASER_BUNDLE = Path(__file__).parent / "node_modules" / "phaser" / "dist" / "phaser.min.js"


def make_m51_split_provider() -> M51SplitParticipantProvider:
    """Build one fresh M5.1 split provider for an API run.

    The initiator and responder are intentionally constructed inside this
    factory so ``run_ab`` can call it once for each condition without sharing
    Ollama audit state between A and B.
    """

    return M51SplitParticipantProvider(
        initiator=OllamaTalkInitiator(),
        responder=RuleResponseProvider(),
    )


def make_m611_two_stage_provider() -> M51SplitParticipantProvider:
    """Build one fresh M6.11 planner/utterance split provider."""

    return M51SplitParticipantProvider(
        initiator=TwoStageTalkInitiator(),
        responder=RuleResponseProvider(),
    )


def make_m611_planner_deepseek_provider() -> M51SplitParticipantProvider:
    """Build M6.11 with the local planner and cloud utterance adapter.

    DeepSeek reads its key at construction time.  If it is absent or
    unavailable, the existing two-stage initiator falls back to rules for the
    turn; this factory therefore remains safe to expose in the UI by default.
    """

    utterance_generator = DeepSeekUtteranceGenerator()
    if not utterance_generator.config.api_key:
        # Avoid even attempting the local planner when cloud mode is not
        # configured. This makes an explicit cloud selection complete quickly
        # through the established deterministic rule provider.
        return M51SplitParticipantProvider(
            initiator=RuleTalkInitiator(),
            responder=RuleResponseProvider(),
        )
    return M51SplitParticipantProvider(
        initiator=TwoStageTalkInitiator(
            utterance_generator=utterance_generator,
        ),
        responder=RuleResponseProvider(),
    )


def make_m82_rule_candidate_deepseek_provider() -> M51SplitParticipantProvider:
    """Build rule candidates + DeepSeek without constructing any local LLM."""

    initiator = DeepSeekRuleCandidateInitiator()
    if not initiator.config.api_key:
        return M51SplitParticipantProvider(
            initiator=RuleTalkInitiator(),
            responder=RuleResponseProvider(),
        )
    return M51SplitParticipantProvider(
        initiator=initiator,
        responder=RuleResponseProvider(),
    )


def make_m14_natural_conversation_provider() -> SplitConversationProvider:
    """Build the opt-in M14 natural-conversation pilot.

    Candidate selection and topic grounding remain rule-owned.  DeepSeek is
    used for validated wording and for the target's response only when an API
    key is configured; otherwise both roles use their deterministic providers.
    """

    initiator = DeepSeekNaturalConversationInitiator()
    responder = DeepSeekNaturalResponseProvider()
    if not initiator.config.api_key or not responder.config.api_key:
        return SplitConversationProvider(
            RuleTalkInitiator(), RuleResponseProvider(), mode="deepseek_natural"
        )
    return SplitConversationProvider(initiator, responder, mode="deepseek_natural")


def make_m15_autonomous_conversation_provider() -> SplitConversationProvider:
    """Build the opt-in M15 autonomous conversation mode."""

    initiator = DeepSeekAutonomousConversationInitiator()
    responder = DeepSeekAutonomousResponseProvider(config=initiator.config)
    narrative = DeepSeekNarrativeProvider(config=initiator.config)
    if not initiator.config.api_key:
        return SplitConversationProvider(
            RuleTalkInitiator(), RuleResponseProvider(), mode="deepseek_autonomous",
            narrative_provider=narrative,
        )
    return SplitConversationProvider(
        initiator, responder, mode="deepseek_autonomous", narrative_provider=narrative
    )


class POCRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/ab-poc/history":
            return self._json({"history": list_history()})
        if parsed.path.startswith("/api/ab-poc/history/"):
            history_id = parsed.path.rsplit("/", 1)[-1]
            try:
                return self._json(load_history(history_id))
            except KeyError as error:
                return self._json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
            except ValueError as error:
                return self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/health":
            return self._json({"service": "yoriyoi-poc", "status": "ok"})
        if parsed.path == "/vendor/phaser.min.js":
            return self._file(PHASER_BUNDLE, content_type="text/javascript; charset=utf-8")
        if parsed.path == "/api/scenarios":
            return self._json(list_scenarios())
        if parsed.path == "/api/experiment-presets":
            return self._json(list_experiment_presets())
        if parsed.path == "/api/ab-poc/compare":
            params = parse_qs(parsed.query, keep_blank_values=True)
            try:
                seed = int(params.get("seed", ["1"])[0])
                participant_provider = self._participant_provider_mode(params)
                record_history = params.get("record_history", [""])[0] == "1"
                if participant_provider == "ollama":
                    return self._run_json(run_ab(seed=seed, participant_provider_factory=OllamaParticipantProvider), participant_provider, "comparison", record_history=record_history)
                if participant_provider == "ollama_split":
                    return self._run_json(run_ab(seed=seed, participant_provider_factory=make_m51_split_provider), participant_provider, "comparison", record_history=record_history)
                if participant_provider == "ollama_two_stage":
                    return self._run_json(run_ab(seed=seed, participant_provider_factory=make_m611_two_stage_provider), participant_provider, "comparison", record_history=record_history)
                if participant_provider == "ollama_planner_deepseek":
                    return self._run_json(run_ab(seed=seed, participant_provider_factory=make_m611_planner_deepseek_provider), participant_provider, "comparison", record_history=record_history)
                if participant_provider == "rule_candidates_deepseek":
                    return self._run_json(run_ab(seed=seed, participant_provider_factory=make_m82_rule_candidate_deepseek_provider), participant_provider, "comparison", record_history=record_history)
                if participant_provider in {"deepseek_natural", "deepseek_natural_conversation"}:
                    return self._run_json(run_ab(seed=seed, conversation_provider_factory=make_m14_natural_conversation_provider), participant_provider, "comparison", record_history=record_history)
                if participant_provider == "deepseek_autonomous":
                    if (os.environ.get("YORIYOI_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")) and params.get("external_consent", [""])[0] != "deepseek":
                        raise ValueError("deepseek_autonomous with a configured API requires external_consent=deepseek")
                    return self._run_json(run_ab(seed=seed, conversation_provider_factory=make_m15_autonomous_conversation_provider), participant_provider, "comparison", record_history=record_history)
                return self._json(run_ab(seed=seed))
            except (KeyError, ValueError) as error:
                return self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/ab-poc/run":
            params = parse_qs(parsed.query, keep_blank_values=True)
            condition = params.get("condition", ["A"])[0]
            try:
                seed = int(params.get("seed", ["1"])[0])
                if condition not in {"A", "B"}:
                    raise ValueError("condition must be A or B")
                participant_provider = self._participant_provider_mode(params)
                record_history = params.get("record_history", [""])[0] == "1"
                if participant_provider == "ollama":
                    return self._run_json(
                        run_condition(
                            condition,
                            seed=seed,
                            participant_provider=OllamaParticipantProvider(),
                        ), participant_provider, "single", record_history=record_history)
                if participant_provider == "ollama_split":
                    return self._run_json(
                        run_condition(
                            condition,
                            seed=seed,
                            participant_provider=make_m51_split_provider(),
                        ), participant_provider, "single", record_history=record_history)
                if participant_provider == "ollama_two_stage":
                    return self._run_json(
                        run_condition(
                            condition,
                            seed=seed,
                            participant_provider=make_m611_two_stage_provider(),
                        ), participant_provider, "single", record_history=record_history)
                if participant_provider == "ollama_planner_deepseek":
                    return self._run_json(
                        run_condition(
                            condition,
                            seed=seed,
                            participant_provider=make_m611_planner_deepseek_provider(),
                        ), participant_provider, "single", record_history=record_history)
                if participant_provider == "rule_candidates_deepseek":
                    return self._run_json(
                        run_condition(
                            condition,
                            seed=seed,
                            participant_provider=make_m82_rule_candidate_deepseek_provider(),
                        ), participant_provider, "single", record_history=record_history)
                if participant_provider in {"deepseek_natural", "deepseek_natural_conversation"}:
                    return self._run_json(
                        run_condition(
                            condition,
                            seed=seed,
                            conversation_provider=make_m14_natural_conversation_provider(),
                        ), participant_provider, "single", record_history=record_history)
                if participant_provider == "deepseek_autonomous":
                    if (os.environ.get("YORIYOI_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")) and params.get("external_consent", [""])[0] != "deepseek":
                        raise ValueError("deepseek_autonomous with a configured API requires external_consent=deepseek")
                    return self._run_json(
                        run_condition(
                            condition,
                            seed=seed,
                            conversation_provider=make_m15_autonomous_conversation_provider(),
                        ), participant_provider, "single", record_history=record_history)
                return self._json(run_condition(condition, seed=seed))
            except (KeyError, ValueError) as error:
                return self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/run":
            params = parse_qs(parsed.query)
            scenario_id = params.get("scenario_id", [""])[0]
            provider = params.get("provider", ["rule_based"])[0]
            preset = params.get("preset", ["poc"])[0]
            try:
                seed = int(params.get("seed", ["1"])[0])
                return self._json(
                    run_scenario(scenario_id, seed=seed, provider=provider, preset=preset)
                )
            except (KeyError, ValueError) as error:
                return self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        if parsed.path == "/":
            # Keep the browser URL under /ab-lab/ so its relative CSS and JS
            # assets resolve correctly. The legacy lab and replay viewers
            # remain available under /lab/ and /replay/.
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/ab-lab/")
            self.end_headers()
            return
        return super().do_GET()

    def _run_json(self, payload: object, provider: str, run_range: str, *, record_history: bool = False):
        """Return a run and persist only explicitly recorded non-rule runs."""
        if record_history and provider != "rule" and isinstance(payload, dict):
            try:
                save_history(payload, provider=provider, run_range=run_range)
            except (OSError, ValueError):
                # History is a convenience for replay; a successful simulation
                # must remain available even if local storage is unavailable.
                pass
        return self._json(payload)

    @staticmethod
    def _participant_provider_mode(params: dict[str, list[str]]) -> str:
        mode = params.get("participant_provider", ["rule"])[0]
        # Generic public alias; retain deepseek_autonomous as the wire-level
        # identifier used by existing bookmarks and integrations.
        if mode == "llm_autonomous":
            mode = "deepseek_autonomous"
        if mode not in {"rule", "ollama", "ollama_split", "ollama_two_stage", "ollama_planner_deepseek", "rule_candidates_deepseek", "deepseek_natural", "deepseek_natural_conversation", "deepseek_autonomous"}:
            raise ValueError("participant_provider must be 'rule', 'ollama', 'ollama_split', 'ollama_two_stage', 'ollama_planner_deepseek', 'rule_candidates_deepseek', 'deepseek_natural', 'deepseek_natural_conversation', or 'deepseek_autonomous'")
        return mode

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/experiments":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Request body must not be empty")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            spec = ExperimentSpec.from_payload(payload)
            return self._json(run_experiment(spec))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json(
                {"error": "Request body must be valid UTF-8 JSON"},
                status=HTTPStatus.BAD_REQUEST,
            )
        except ValueError as error:
            return self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def _json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, *, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Phaser bundle was not installed")
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Yoriyoi POC replay viewer.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), POCRequestHandler)
    print(f"Yoriyoi POC: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
