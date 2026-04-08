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

    print("\n=== response (model_dump) ===")
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    try:
        dumped = response.model_dump(mode="json")
        print(json.dumps(dumped, indent=2, ensure_ascii=False, default=str))
    except Exception as exc:  # pragma: no cover - debug utility
        print(f"(model_dump failed: {exc})\n{response!r}")

    msg = response.choices[0].message
    print("\n=== message fields (non-callable) ===")
    for name in sorted(dir(msg)):
        if name.startswith("_"):
            continue
        try:
            val = getattr(msg, name)
        except Exception as exc:
            print(f"  {name}: <error {exc}>")
            continue
        if callable(val):
            continue
        print(f"  {name}: {val!r}")

    raw = getattr(response.choices[0], "message", None)
    content = (getattr(raw, "content", None) or "").strip()
    print("\n=== summary ===")
    print(f"  choices[0].finish_reason: {response.choices[0].finish_reason!r}")
    print(f"  message.content (strip): {content!r}")
    if not content:
        print(
            "\n  NOTE: content is empty but usage shows completion_tokens > 0.\n"
            "  Some gateways/models put text in other message fields (e.g. reasoning) or require\n"
            "  a different API (Responses) or parameters — compare message fields above."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
