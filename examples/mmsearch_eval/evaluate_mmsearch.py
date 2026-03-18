import argparse
import asyncio
import json
import os
import uuid

from datasets import load_dataset

from mmsearch_workflow import MMSearchWorkflow

from rllm.engine.rollout.openai_engine import OpenAIEngine


async def _run_eval(args) -> None:
    ds = load_dataset(args.dataset, split=args.split)

    engine = OpenAIEngine(
        model=args.model,
        base_url="http://localhost:30000/v1",
        api_key="EMPTY",
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
    )

    wf = MMSearchWorkflow(engine)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    correct = 0
    total = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            uid = str(uuid.uuid4())
            ep = await wf.run(
                task={
                    "query": ex["query"],
                    "query_image": ex["query_image"],
                    "gt_answer": ex["gt_answer"],
                    "alternative_gt_answers": ex.get("alternative_gt_answers", []),
                },
                uid=uid,
            )

            total += 1
            correct += 1 if ep.is_correct else 0

            row = {
                "idx": i,
                "uid": uid,
                "query": ex["query"],
                "prediction": ep.metrics["prediction"],
                "labels": ep.metrics["labels"],
                "exact_match": ep.metrics["exact_match"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    acc = correct / total if total else 0.0
    summary = {"dataset": args.dataset, "split": args.split, "total": total, "correct": correct, "accuracy": acc}
    with open(args.output + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="CaraJ/MMSearch")
    p.add_argument("--split", default="end2end")
    p.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")

    p.add_argument("--output", default="logs/mmsearch.jsonl")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-prompt-length", type=int, default=8192)
    p.add_argument("--max-response-length", type=int, default=2048)

    args = p.parse_args()
    asyncio.run(_run_eval(args))


if __name__ == "__main__":
    main()

