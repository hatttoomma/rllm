import argparse
import json
import os
from typing import Any

from openai import OpenAI


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for question-answering. "
    "Given a question, a model prediction, and one or more reference answers, "
    "decide whether the prediction is semantically correct. "
    "Be robust to minor formatting differences, capitalization, and punctuation, "
    "but do not accept incorrect entities, numbers, or factual mismatches."
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


def load_records(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported input JSON format for {path}. Expected a list or jsonl lines.")


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
        "Return JSON with fields:\n"
        "- is_correct: boolean\n"
        "- score: float in [0,1]\n"
        "- reason: short explanation"
    )


def judge_one(client: OpenAI, model: str, item: dict[str, Any], temperature: float, max_tokens: int) -> dict[str, Any]:
    user_prompt = build_prompt(item)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = client.chat.completions.create(
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
        # Fallback for OpenAI-compatible providers that don't support json_schema.
        fallback_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Return ONLY a valid JSON object with keys "
                    '{"is_correct": boolean, "score": number, "reason": string}.'
                ),
            }
        ]
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=fallback_messages,
        )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Empty response from judge model.")
    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for MMSearch eval outputs.")
    parser.add_argument("--input", required=True, help="Path to evaluate_mmsearch output (.jsonl or .json).")
    parser.add_argument("--output", default=None, help="Output path for judged jsonl.")
    parser.add_argument("--summary-output", default=None, help="Optional summary json path.")

    parser.add_argument("--model", default="gpt-4o-mini", help="Judge model name.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1 or http://localhost:8000/v1",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only judge first N items.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    if args.output is None:
        args.output = args.input + ".llm_judge.jsonl"
    if args.summary_output is None:
        args.summary_output = args.output + ".summary.json"

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary_dir = os.path.dirname(args.summary_output)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    total = 0
    judged_correct = 0
    exact_match_available = 0
    agree_with_exact_match = 0
    parse_errors = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for item in records:
            total += 1
            row = dict(item)
            try:
                j = judge_one(
                    client=client,
                    model=args.model,
                    item=item,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                judge_correct = bool(j["is_correct"])
                row["judge"] = {
                    "is_correct": judge_correct,
                    "score": float(j["score"]),
                    "reason": str(j["reason"]),
                }
            except Exception as e:
                parse_errors += 1
                judge_correct = False
                row["judge"] = {
                    "is_correct": False,
                    "score": 0.0,
                    "reason": f"judge_error: {repr(e)}",
                }

            if judge_correct:
                judged_correct += 1

            if "exact_match" in item:
                exact_match_available += 1
                if bool(item["exact_match"]) == judge_correct:
                    agree_with_exact_match += 1

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if total % 20 == 0:
                print(
                    json.dumps(
                        {
                            "done": total,
                            "judge_accuracy": judged_correct / total if total else 0.0,
                            "parse_errors": parse_errors,
                        },
                        ensure_ascii=False,
                    )
                )

    summary = {
        "input": args.input,
        "output": args.output,
        "total": total,
        "judge_correct": judged_correct,
        "judge_accuracy": judged_correct / total if total else 0.0,
        "parse_errors": parse_errors,
        "model": args.model,
        "base_url": args.base_url,
    }
    if exact_match_available > 0:
        summary["exact_match_available"] = exact_match_available
        summary["judge_exact_match_agreement"] = agree_with_exact_match / exact_match_available

    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
