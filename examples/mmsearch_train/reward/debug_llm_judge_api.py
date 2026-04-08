#!/usr/bin/env python3
"""
One-shot debug for MMS LLM judge API calls.

Uses the same env vars as examples/mmsearch_train/reward/llm_judge.py:
  MMS_JUDGE_BASE_URL  - OpenAI-compatible base URL (required)
  MMS_JUDGE_MODEL     - Model name (required)
  MMS_JUDGE_API_KEY   - API key (required)

Optional:
  MMS_JUDGE_TEMPERATURE  - default 0.0
  MMS_JUDGE_MAX_TOKENS   - default 256

Run from repo root (or anywhere with OPENAI deps installed):
  python examples/mmsearch_train/reward/debug_llm_judge_api.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for question-answering. "
    "Given a question, a model prediction, and one or more reference answers, "
    "decide whether the prediction is semantically correct. "
    "Be robust to minor formatting differences, capitalization, and punctuation, "
    "but do not accept incorrect entities, numbers, or factual mismatches. "
    "If prediction is empty, whitespace-only, or missing, it must be judged as incorrect."
)


def _sample_user_prompt() -> str:
    # Minimal prompt aligned with build_prompt() in llm_judge.py
    return (
        "Please judge whether the prediction correctly answers the question.\n\n"
        "Question:\nWhat is 2+2?\n\n"
        "Prediction:\n4\n\n"
        "Reference answers:\n"
        "- 4\n\n"
        "Output rule:\n"
        "- Respond with ONLY ONE word: 'correct' or 'incorrect'.\n"
        "- No JSON, no punctuation, no extra words.\n"
        "- If Prediction is empty or whitespace-only, respond 'incorrect'."
    )


def _redact_key(key: str | None) -> str:
    if not key:
        return "(empty)"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _raw_http_body(raw: Any) -> str | None:
    if hasattr(raw, "http_response") and raw.http_response is not None:
        return raw.http_response.text
    if hasattr(raw, "text"):
        text = raw.text
        return text if isinstance(text, str) else None
    return None


def _parse_raw_response(raw: Any):
    if hasattr(raw, "parse"):
        return raw.parse()
    if hasattr(raw, "data") and raw.data is not None:
        return raw.data
    return raw


def main() -> int:
    base_url = os.getenv("MMS_JUDGE_BASE_URL", "").strip()
    model = os.getenv("MMS_JUDGE_MODEL", "").strip()
    api_key = os.getenv("MMS_JUDGE_API_KEY", "").strip()
    temperature = float(os.getenv("MMS_JUDGE_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("MMS_JUDGE_MAX_TOKENS", "256"))

    print("=== env (api key redacted) ===")
    print(f"  MMS_JUDGE_BASE_URL={base_url!r}")
    print(f"  MMS_JUDGE_MODEL={model!r}")
    print(f"  MMS_JUDGE_API_KEY={_redact_key(api_key)!r}")
    print(f"  MMS_JUDGE_TEMPERATURE={temperature}")
    print(f"  MMS_JUDGE_MAX_TOKENS={max_tokens}")

    if not base_url or not model or not api_key:
        print(
            "\nMissing required env. Set MMS_JUDGE_BASE_URL, MMS_JUDGE_MODEL, MMS_JUDGE_API_KEY.",
            file=sys.stderr,
        )
        return 2

    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _sample_user_prompt()},
    ]

    print("\n=== request ===")
    print(json.dumps({"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}, indent=2, ensure_ascii=False))

    print("\n=== raw HTTP body (what the server actually returned) ===")
    raw = client.with_raw_response.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    body = _raw_http_body(raw)
    if body is None:
        print("(could not read raw body from SDK response object)")
    else:
        try:
            print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(body)

    response = _parse_raw_response(raw)

    print("\n=== parsed ChatCompletion (model_dump) ===")
    try:
        dumped = response.model_dump(mode="json")
        print(json.dumps(dumped, indent=2, ensure_ascii=False, default=str))
    except Exception as exc:  # pragma: no cover - debug utility
        print(f"(model_dump failed: {exc})\n{response!r}")

    msg = response.choices[0].message
    print("\n=== message.model_dump() (avoid dir() / pydantic deprecation) ===")
    try:
        md = msg.model_dump(mode="python")
        print(json.dumps(md, indent=2, ensure_ascii=False, default=str))
    except Exception as exc:
        print(f"(model_dump failed: {exc})")

    content = (msg.content or "").strip()
    print("\n=== summary ===")
    print(f"  choices[0].finish_reason: {response.choices[0].finish_reason!r}")
    print(f"  message.content (strip): {content!r}")
    if not content:
        print(
            "\n  Interpretation: If raw JSON above has no assistant text under the usual keys\n"
            "  (e.g. choices[0].message.content), the gateway is not returning text for this\n"
            "  model/endpoint — fix the provider config or use their documented API (e.g. Responses).\n"
            "  If raw JSON *does* contain text but model_dump shows content=null, the response\n"
            "  shape is non-standard and the OpenAI SDK is dropping fields (open an issue with the provider)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
