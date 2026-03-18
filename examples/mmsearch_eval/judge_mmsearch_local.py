import argparse
import json
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for question-answering. "
    "Given a question, a model prediction, and one or more reference answers, "
    "decide whether the prediction is semantically correct. "
    "Be robust to minor formatting differences, capitalization, and punctuation, "
    "but do not accept incorrect entities, numbers, or factual mismatches."
)


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
        "- reason: short explanation\n\n"
        "Return ONLY one valid JSON object."
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if not content:
        raise RuntimeError("Empty response from judge model.")

    # Strip common markdown fences if the model still emits them.
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: extract first balanced {...} span.
    start = content.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response: {content[:200]}")

    depth = 0
    end = -1
    for i in range(start, len(content)):
        ch = content[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end < 0:
        raise ValueError(f"Unclosed JSON object in response: {content[:200]}")

    obj_text = content[start : end + 1]
    parsed = json.loads(obj_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed JSON is not an object: {obj_text[:200]}")
    return parsed


def _normalize_judge_output(raw: dict[str, Any]) -> dict[str, Any]:
    if "is_correct" not in raw or "score" not in raw or "reason" not in raw:
        raise ValueError(f"Missing required fields in judge output: {raw}")

    score = float(raw["score"])
    score = max(0.0, min(1.0, score))
    return {
        "is_correct": bool(raw["is_correct"]),
        "score": score,
        "reason": str(raw["reason"]),
    }


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def judge_one(
    model,
    tokenizer,
    item: dict[str, Any],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    user_prompt = build_prompt(item)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(_model_device(model))
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][input_len:]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    parsed = _extract_json_object(content)
    return _normalize_judge_output(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM-as-a-judge for MMSearch eval outputs.")
    parser.add_argument("--input", required=True, help="Path to evaluate_mmsearch output (.jsonl or .json).")
    parser.add_argument("--output", default=None, help="Output path for judged jsonl.")
    parser.add_argument("--summary-output", default=None, help="Optional summary json path.")

    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Local HF model id/path.")
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Hugging Face device_map. Common values: "auto", "cuda:0", "cpu".',
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype for loading.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code when loading.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only judge first N items.")
    args = parser.parse_args()

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    if args.output is None:
        args.output = args.input + ".llm_judge.local.jsonl"
    if args.summary_output is None:
        args.summary_output = args.output + ".summary.json"

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary_dir = os.path.dirname(args.summary_output)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    torch_dtype = _resolve_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

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
                    model=model,
                    tokenizer=tokenizer,
                    item=item,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                judge_correct = bool(j["is_correct"])
                row["judge"] = j
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
        "device_map": args.device_map,
        "dtype": args.dtype,
    }
    if exact_match_available > 0:
        summary["exact_match_available"] = exact_match_available
        summary["judge_exact_match_agreement"] = agree_with_exact_match / exact_match_available

    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
