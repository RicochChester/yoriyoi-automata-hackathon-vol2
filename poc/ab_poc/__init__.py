"""Public API for the deterministic A/B POC."""

from .config import SCENARIO_VERSION, load_config, load_legacy_config, load_m11_config
from .contracts import (
    ConversationEpisode,
    ConversationUtterance,
    EPISODE_END_REASONS,
    EpisodeUtterance,
    EpisodeInitiator,
    EpisodeResponder,
    VerbalEpisodeEvaluation,
    VerbalEpisodeEvaluator,
    Initiation,
    ParticipantContext,
    ResponseContext,
    ResponseOutput,
    ParticipantProvider,
    RelationshipEvaluator,
    ResponseProvider,
    TalkInitiator,
)
from .engine import run_ab, run_condition
from .affect import AffectTransition, AffectUpdater, DirectionalAffectState
from .domain import DialogueTurn, OnlineExperience, OnlineMemory
from .rule_providers import RuleParticipantProvider, RuleRelationshipEvaluator
from .llm_config import LLMConfig, OllamaConfig, load_llm_config
from .ollama_participant import OllamaParticipantProvider
from .ollama_initiator import OllamaTalkInitiator
from .two_stage_initiator import (
    ConversationPlan,
    OllamaConversationPlanner,
    OllamaUtteranceGenerator,
    Planner,
    TwoStageTalkInitiator,
    UtteranceGenerator,
)
from .deepseek_adapter import (
    DEFAULT_MAX_REQUEST_BYTES,
    DeepSeekAdapterError,
    DeepSeekCallBudgetExceeded,
    DeepSeekConfig,
    DeepSeekJsonClient,
    DeepSeekUtteranceGenerator,
    load_deepseek_config,
)
from .deepseek_rule_candidate_initiator import (
    DeepSeekRuleCandidateInitiator,
    DeepSeekNaturalConversationInitiator,
    RulePlanCandidate,
    RulePlanCandidateBuilder,
)
from .deepseek_response import DeepSeekNaturalResponseProvider, DeepSeekResponseProvider
from .deepseek_grounded_response import (
    DeepSeekGroundedResponseProvider,
    RuleResponseCandidate,
    RuleResponseCandidateBuilder,
)
from .conversation_provider import SplitConversationProvider
from .verbal_evaluator import (
    DeterministicVerbalEpisodeEvaluator,
    RuleVerbalEpisodeEvaluator,
)
from .deepseek_autonomous import (
    AutonomousOutputError,
    DeepSeekAutonomousConversationInitiator,
    DeepSeekAutonomousResponseProvider,
)
from .narrative import (
    DeepSeekNarrativeProvider,
    LocalNarrativeProvider,
    NarrativeOutputError,
)
from .composite_participant import CompositeParticipantProvider, M51SplitParticipantProvider
from .rule_providers import RuleResponseProvider, RuleTalkInitiator
from .quality_review import (
    BlindReviewItem,
    FatalIssue,
    REVIEW_DIMENSIONS,
    ReviewRecord,
    aggregate_review_records,
    extract_blind_review_items,
    score_label,
    validate_review_record,
)
from .m6_quality_review import (
    M6FatalIssue,
    M6ReviewRecord,
    M6_FATAL_ISSUES,
    aggregate_m6_review_records,
    build_m6_blind_items,
    extract_m6_blind_review_items,
    validate_m6_review_record,
)

__all__ = [
    "ParticipantProvider",
    "ConversationEpisode",
    "ConversationUtterance",
    "EPISODE_END_REASONS",
    "EpisodeUtterance",
    "EpisodeInitiator",
    "EpisodeResponder",
    "VerbalEpisodeEvaluation",
    "VerbalEpisodeEvaluator",
    "RelationshipEvaluator",
    "Initiation",
    "ParticipantContext",
    "ResponseContext",
    "ResponseOutput",
    "TalkInitiator",
    "ResponseProvider",
    "RuleParticipantProvider",
    "RuleTalkInitiator",
    "RuleResponseProvider",
    "CompositeParticipantProvider",
    "M51SplitParticipantProvider",
    "RuleRelationshipEvaluator",
    "LLMConfig",
    "OllamaConfig",
    "OllamaParticipantProvider",
    "OllamaTalkInitiator",
    "ConversationPlan",
    "Planner",
    "UtteranceGenerator",
    "OllamaConversationPlanner",
    "OllamaUtteranceGenerator",
    "TwoStageTalkInitiator",
    "DeepSeekAdapterError",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DeepSeekCallBudgetExceeded",
    "DeepSeekConfig",
    "DeepSeekJsonClient",
    "DeepSeekUtteranceGenerator",
    "DeepSeekRuleCandidateInitiator",
    "DeepSeekNaturalConversationInitiator",
    "DeepSeekResponseProvider",
    "DeepSeekNaturalResponseProvider",
    "DeepSeekGroundedResponseProvider",
    "SplitConversationProvider",
    "DeterministicVerbalEpisodeEvaluator",
    "RuleVerbalEpisodeEvaluator",
    "AutonomousOutputError",
    "DeepSeekAutonomousConversationInitiator",
    "DeepSeekAutonomousResponseProvider",
    "DeepSeekNarrativeProvider",
    "LocalNarrativeProvider",
    "NarrativeOutputError",
    "RulePlanCandidate",
    "RulePlanCandidateBuilder",
    "RuleResponseCandidate",
    "RuleResponseCandidateBuilder",
    "BlindReviewItem",
    "FatalIssue",
    "REVIEW_DIMENSIONS",
    "ReviewRecord",
    "aggregate_review_records",
    "extract_blind_review_items",
    "score_label",
    "validate_review_record",
    "M6FatalIssue",
    "M6ReviewRecord",
    "M6_FATAL_ISSUES",
    "aggregate_m6_review_records",
    "build_m6_blind_items",
    "extract_m6_blind_review_items",
    "validate_m6_review_record",
    "load_llm_config",
    "load_deepseek_config",
    "load_config",
    "load_legacy_config",
    "load_m11_config",
    "SCENARIO_VERSION",
    "OnlineExperience",
    "OnlineMemory",
    "DialogueTurn",
    "run_ab",
    "run_condition",
    "AffectTransition",
    "AffectUpdater",
    "DirectionalAffectState",
]
