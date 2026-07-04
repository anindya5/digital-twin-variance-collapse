"""Anthropic client wrapper: forced tool-use for structured output, disk
caching (so re-running a script never re-pays for a call already made), and
simple retry/backoff on transient errors.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .config import Config

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def _cache_key(system: str, user: str, tool: dict, model: str, temperature: float) -> str:
    payload = json.dumps(
        {"system": system, "user": user, "tool": tool, "model": model, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def call_tool(
    cfg: Config,
    system: str,
    user: str,
    tool: dict,
    cache_subdir: str = "llm_calls",
    max_tokens: int | None = None,
) -> dict:
    """Call the model, forcing it to respond via `tool`, and return the
    parsed tool input dict. Results are cached to disk keyed by the exact
    prompt/tool/model/temperature, so identical calls are free after the
    first time.
    """
    cache_dir = cfg.cache_dir / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(system, user, tool, cfg.model, cfg.temperature)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import anthropic

    client = _get_client()
    last_err: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            response = client.messages.create(
                model=cfg.model,
                max_tokens=max_tokens or cfg.max_tokens,
                temperature=cfg.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
                timeout=cfg.request_timeout_s,
            )
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                raise ValueError(f"Model did not return a tool_use block: {response.content}")
            result = tool_use_blocks[0].input
            with open(cache_path, "w") as f:
                json.dump(result, f)
            return result
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_err = e
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)
    raise RuntimeError(f"Exceeded {cfg.max_retries} retries calling Anthropic API") from last_err
