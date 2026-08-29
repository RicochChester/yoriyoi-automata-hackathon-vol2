"""Configuration for the M4 local Ollama participant adapter.

The simulation itself has no dependency on this module.  Keeping the small
set of transport and generation settings in a validated dataclass makes an
experiment reproducible and makes it possible to construct providers without
reading the process environment.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b-instruct-2507-q4_K_M"
    timeout: float = 10.0
    max_attempts: int = 2
    temperature: float = 0.2
    num_predict: int = 128
    think: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty URL")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an http(s) URL")
        if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("base_url must be a loopback root URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain credentials")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("base_url must use a loopback host")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(float(self.timeout)) or not 0 < float(self.timeout) <= 120:
            raise ValueError("timeout must be finite and in the range (0, 120]")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or not 1 <= self.max_attempts <= 8:
            raise ValueError("max_attempts must be an integer in the range 1..8")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a finite non-negative number")
        if not math.isfinite(float(self.temperature)) or not 0 <= float(self.temperature) <= 2:
            raise ValueError("temperature must be finite and in the range 0..2")
        if isinstance(self.num_predict, bool) or not isinstance(self.num_predict, int) or not 1 <= self.num_predict <= 2048:
            raise ValueError("num_predict must be an integer in the range 1..2048")
        if not isinstance(self.think, bool):
            raise ValueError("think must be a boolean")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "OllamaConfig":
        """Read optional ``YORIYOI_OLLAMA_*`` settings from the environment."""

        env = os.environ if environ is None else environ

        def value(name: str, default: str, legacy_name: str | None = None) -> str:
            # Accept the concise OLLAMA_* spelling as well as the namespaced
            # project spelling used by the launcher.
            if name in env:
                return env[name]
            if legacy_name and legacy_name in env:
                return env[legacy_name]
            return default

        def boolean_value(name: str, default: str, legacy_name: str | None = None) -> bool:
            raw = value(name, default, legacy_name)
            if not isinstance(raw, str):
                raise ValueError(f"{name} must be true or false")
            normalized = raw.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
            raise ValueError(f"{name} must be true or false")

        try:
            return cls(
                base_url=value("YORIYOI_OLLAMA_BASE_URL", cls.base_url, "OLLAMA_BASE_URL"),
                model=value("YORIYOI_OLLAMA_MODEL", cls.model, "OLLAMA_MODEL"),
                timeout=float(value("YORIYOI_OLLAMA_TIMEOUT", str(cls.timeout), "OLLAMA_TIMEOUT")),
                max_attempts=int(value("YORIYOI_OLLAMA_MAX_ATTEMPTS", str(cls.max_attempts), "OLLAMA_MAX_ATTEMPTS")),
                temperature=float(value("YORIYOI_OLLAMA_TEMPERATURE", str(cls.temperature), "OLLAMA_TEMPERATURE")),
                num_predict=int(value("YORIYOI_OLLAMA_NUM_PREDICT", str(cls.num_predict), "OLLAMA_NUM_PREDICT")),
                think=boolean_value("YORIYOI_OLLAMA_THINK", "false", "OLLAMA_THINK"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid YORIYOI_OLLAMA_* configuration") from exc


    @property
    def temp(self) -> float:
        """Short read-only spelling used in experiment notes."""

        return float(self.temperature)


# A short alias is convenient for callers that do not care which local
# transport implements the config.  Both names intentionally refer to the
# same validated type.
LLMConfig = OllamaConfig


def load_llm_config(environ: dict[str, str] | None = None) -> OllamaConfig:
    return OllamaConfig.from_env(environ)


load_ollama_config = load_llm_config


__all__ = ["LLMConfig", "OllamaConfig", "load_llm_config", "load_ollama_config"]
