"""Small, local-only persistence for completed LLM A/B runs.

History files intentionally live outside the repository's tracked output and
contain only the already-sanitised run result.  The module does not know how
to call a provider and is therefore safe to use from both the HTTP handler
and tests.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_DEFAULT_DIR = Path(__file__).resolve().parent / ".ab-lab-history"
HISTORY_DIR = Path(os.environ.get("YORIYOI_HISTORY_DIR", str(_DEFAULT_DIR)))
MAX_HISTORY = 30
MAX_RESULT_BYTES = 8 * 1024 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,79}$")


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError("invalid history id")
    return value


def _strip_sensitive(value: Any, key: str = "") -> Any:
    """Defensively remove credentials and request-shaped fields."""
    lowered = key.lower()
    if any(token in lowered for token in ("api_key", "apikey", "authorization", "secret", "access_token")):
        return None
    if lowered in {"request", "request_body", "raw_request", "raw_response", "headers"}:
        return None
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item in value.items():
            safe_item = _strip_sensitive(item, str(key))
            if safe_item is not None:
                cleaned[str(key)] = safe_item
        return cleaned
    if isinstance(value, list):
        return [_strip_sensitive(item, key) for item in value]
    return value


def _files() -> list[Path]:
    if not HISTORY_DIR.is_dir():
        return []
    return sorted(
        (path for path in HISTORY_DIR.glob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def save_history(result: Mapping[str, Any], *, provider: str, run_range: str) -> dict[str, Any]:
    """Persist an LLM result atomically and return its metadata envelope."""
    if provider == "rule":
        raise ValueError("rule results are not persisted in LLM history")
    safe_result = _strip_sensitive(copy.deepcopy(dict(result)))
    encoded_result = json.dumps(safe_result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded_result) > MAX_RESULT_BYTES:
        raise ValueError("run result is too large to persist")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_id = f"{int(time.time() * 1000):x}-{uuid.uuid4().hex[:12]}"
    envelope = {
        "schema_version": 1,
        "id": history_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "run_range": run_range,
        "seed": safe_result.get("root_seed"),
        "result": safe_result,
    }
    target = HISTORY_DIR / f"{history_id}.json"
    temporary = HISTORY_DIR / f".{history_id}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        # If writing or replacing fails, do not leave a misleading partial
        # history artifact behind.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    for old in _files()[MAX_HISTORY:]:
        try:
            old.unlink()
        except OSError:
            pass
    return {key: envelope[key] for key in ("id", "created_at", "provider", "run_range", "seed")}


def list_history() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _files()[:MAX_HISTORY]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            history_id = _safe_id(payload.get("id", path.stem))
            entries.append({
                "id": history_id,
                "created_at": payload.get("created_at"),
                "provider": payload.get("provider", "llm"),
                "run_range": payload.get("run_range", "comparison"),
                "seed": payload.get("seed"),
            })
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            continue
    return entries


def load_history(history_id: str) -> dict[str, Any]:
    path = HISTORY_DIR / f"{_safe_id(history_id)}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeyError("history entry not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("history entry is not valid") from exc
    if payload.get("id") != history_id or not isinstance(payload.get("result"), Mapping):
        raise ValueError("history entry is not valid")
    return payload


__all__ = ["HISTORY_DIR", "MAX_HISTORY", "load_history", "list_history", "save_history"]
