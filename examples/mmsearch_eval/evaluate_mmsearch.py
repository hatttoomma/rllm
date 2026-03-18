import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import closing

import requests
from datasets import load_dataset

from mmsearch_workflow import MMSearchWorkflow
from rllm.engine.rollout.openai_engine import OpenAIEngine


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_vllm(base_url: str, timeout: int = 600) -> None:
    url = base_url.rstrip("/") + "/models"
    start = time.time()
    last_err = None

    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = repr(e)

        time.sleep(2)

    raise RuntimeError(f"vLLM server not ready within {timeout}s. Last error: {last_err}")


def start_vllm_server(args):
    port = args.vllm_port or find_free_port()
    host = args.vllm_host
    base_url = f"http://{host}:{port}/v1"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if args.limit_mm_per_prompt:
        cmd.extend(["--limit-mm-per-prompt", args.limit_mm_per_prompt])

    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])

    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.disable_frontend_multiprocessing:
        cmd.append("--disable-frontend-multiprocessing")

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("Launching vLLM:")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,
        stderr=None,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    try:
        wait_for_vllm(base_url, timeout=args.vllm_startup_timeout)
    except Exception:
        stop_process_tree(proc)
        raise

    return proc, base_url


def stop_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


async def _run_eval(args, base_url: str) -> None:
    ds = load_dataset(args.dataset, args.split, split=args.split)

    engine = OpenAIEngine(
        model=args.model,
        base_url=base_url,
        api_key=args.api_key,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
    )

    wf = MMSearchWorkflow(engine)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

            if (i + 1) % args.log_every == 0:
                acc = correct / total if total else 0.0
                print(
                    json.dumps(
                        {
                            "done": i + 1,
                            "total": total,
                            "correct": correct,
                            "accuracy": acc,
                        },
                        ensure_ascii=False,
                    )
                )

    acc = correct / total if total else 0.0
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "model": args.model,
        "base_url": base_url,
    }

    with open(args.output + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()

    # dataset / eval
    p.add_argument("--dataset", default="CaraJ/MMSearch")
    p.add_argument("--split", default="end2end")
    p.add_argument("--output", default="logs/mmsearch.jsonl")
    p.add_argument("--log-every", type=int, default=10)

    # model
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--api-key", default="EMPTY")

    # sampling / engine
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-prompt-length", type=int, default=8192)
    p.add_argument("--max-response-length", type=int, default=2048)

    # vLLM server args
    p.add_argument("--vllm-host", default="127.0.0.1")
    p.add_argument("--vllm-port", type=int, default=None)
    p.add_argument("--vllm-startup-timeout", type=int, default=600)

    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=None)

    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--disable-frontend-multiprocessing", action="store_true")
    p.add_argument("--cuda-visible-devices", default=None)

    # for VL model
    p.add_argument(
        "--limit-mm-per-prompt",
        default='{"image": 4}',
    )

    args = p.parse_args()

    proc = None
    try:
        proc, base_url = start_vllm_server(args)
        asyncio.run(_run_eval(args, base_url))
    finally:
        if proc is not None:
            stop_process_tree(proc)


if __name__ == "__main__":
    main()