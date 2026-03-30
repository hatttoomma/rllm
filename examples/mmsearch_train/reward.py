import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from typing import Any


from openai import AsyncOpenAI

from rllm.agents.agent import Action
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput


logger = logging.getLogger(__name__)

RewardResult = RewardOutput | Awaitable[RewardOutput]
MMSearchRewardFunction = Callable[[dict, str | Action | ModelOutput], RewardResult]

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for question-answering. "
    "Given a question, a model prediction, and one or more reference answers, "
    "decide whether the prediction is semantically correct. "
    "Be robust to minor formatting differences, capitalization, and punctuation, "
    "but do not accept incorrect entities, numbers, or factual mismatches. "
    "If prediction is empty, whitespace-only, or missing, it must be judged as incorrect."
)

JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["is_correct", "score", "reason"],
    "additionalProperties": False,
}

def build_prompt(item: dict[str, Any]) -> str:
    query = item.get("query", "")
    prediction = item.get("prediction", "")
    labels = item.get("labels", [])
    labels_text = "\n".join(f"- {x}" for x in labels) if labels else "- (no labels provided)"
    return (
        "Please evaluate whether the prediction correctly answers the question.\n\n"
        f"Question:\n{query}\n\n"
        f"Prediction:\n{prediction}\n\n"
        f"Reference answers:\n{labels_text}\n\n"
        "Important rule: if Prediction is empty or whitespace-only, set is_correct=false and score=0.\n\n"
        "Return JSON with fields:\n"
        "- is_correct: boolean\n"
        "- score: float in [0,1]\n"
        "- reason: short explanation"
    )


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _string_match(pred: str, labels: list[str]) -> bool:
    p = _normalize(pred)
    return any(p == _normalize(x) for x in labels)


def _normalize_action(action: str | Action | ModelOutput) -> str:
    if isinstance(action, Action):
        action = action.action
    if isinstance(action, ModelOutput):
        action = action.content or action.text or ""
    if action is None:
        return ""
    return str(action)


def _get_labels(task_info: dict) -> list[str]:
    labels = task_info.get("answer", [])
    if labels is None:
        return []
    if isinstance(labels, str):
        return [labels]
    if isinstance(labels, Sequence):
        return [str(label) for label in labels if label is not None]
    return [str(labels)]


def exact_match_reward_fn(task_info: dict, action: str | Action | ModelOutput) -> RewardOutput:
    prediction = _normalize_action(action)
    labels = _get_labels(task_info)
    is_correct = _string_match(prediction, labels)
    return RewardOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        metadata={"exact_match": is_correct},
    )


async def _judge_one(
    client: AsyncOpenAI,
    model: str,
    task_info: dict,
    prediction: str,
    temperature: float,
    max_tokens: int,
) -> RewardOutput:
    if not prediction.strip():
        return RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={"judge_reason": "prediction is empty or whitespace-only; skip judge inference."},
        )

    item = {
        "query": task_info.get("query", ""),
        "prediction": prediction,
        "labels": _get_labels(task_info),
    }
    user_prompt = build_prompt(item)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "judge_output", "schema": JUDGE_OUTPUT_SCHEMA, "strict": True},
            },
            messages=messages,
        )
    except Exception:
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY a valid JSON object with keys "
                        '{"is_correct": boolean, "score": number, "reason": string}.'
                    ),
                }
            ],
        )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Empty response from judge model.")

    judged = json.loads(content)
    score = float(judged["score"])
    is_correct = bool(judged["is_correct"])
    reason = str(judged["reason"])
    return RewardOutput(
        reward=score,
        is_correct=is_correct,
        metadata={"judge_score": score, "judge_reason": reason, "judge_model": model},
    )


def make_llm_judge_reward_fn(
    base_url: str,
    model: str,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> RewardFunction:
    if not base_url or not model:
        raise ValueError(
            "reward_type=llm_judge requires non-empty judge config: "
            "set MMS_JUDGE_BASE_URL and MMS_JUDGE_MODEL environment variables."
        )

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
    )

    async def llm_judge_reward_fn(task_info: dict, action: str | Action | ModelOutput) -> RewardOutput:
        prediction = _normalize_action(action)
        return await _judge_one(
            client=client,
            model=model,
            task_info=task_info,
            prediction=prediction,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return llm_judge_reward_fn
