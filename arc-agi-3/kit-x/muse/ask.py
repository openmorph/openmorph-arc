# SPDX-License-Identifier: MIT-0
"""Ask every prompt of the muse test to one or more OpenAI-compatible chat servers; log every answer to JSONL.

Standard library only. Works against the RunPod pod (`pod.py`), a local vLLM / SGLang / llama.cpp server, or the
owner's Mac. Every request and answer is appended to the log as it completes, so a run can be killed and resumed
(prompts already answered for a config are skipped) and scored again for free (`muse.py score`).

    python ask.py data/prompts.jsonl --configs configs.json --out answers/run.jsonl [--limit 20] [--workers 8]

configs.json: a list of {"config": name, "base_url": "http://host:port/v1", "model": served model name,
"max_tokens": int, "temperature": float, "extra": {...merged into the request body...}}. `base_url` may use
`{pod}` (replaced by --pod).
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def stream(url, body, timeout):
    """POST a streaming chat request; return (content, reasoning, finish_reason, usage). Streaming keeps bytes
    flowing, so a long (thinking) answer is not cut by an HTTP proxy's idle timeout (RunPod's is about 100 s)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                         "User-Agent": "openmorph-muse/1.0"})
    content, reasoning, finish, usage = [], [], None, {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                finish = ch.get("finish_reason") or finish
    return "".join(content), ("".join(reasoning) or None), finish, usage


def ask_one(cfg, p, timeout, retries=3):
    body = dict(model=cfg["model"], messages=p["messages"], max_tokens=cfg.get("max_tokens", 1024),
                temperature=cfg.get("temperature", 0.0), seed=cfg.get("seed", 0))
    body.update(cfg.get("extra", {}))
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            content, reasoning, finish, usage = stream(url, body, timeout)
            return dict(prompt_id=p["prompt_id"], config=cfg["config"], model=cfg["model"],
                        request={k: v for k, v in body.items() if k != "messages"}, answer=content,
                        reasoning=reasoning, finish=finish, usage=usage, latency_s=round(time.time() - t0, 3),
                        ts=time.time())
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return dict(prompt_id=p["prompt_id"], config=cfg["config"], model=cfg["model"], error=err, ts=time.time())


def run(prompts, configs, out, workers=8, timeout=300, limit=None, on_answer=None):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not a.get("error"):
                done.add((a["config"], a["prompt_id"]))
    jobs = [(c, p) for c in configs for p in (prompts[:limit] if limit else prompts)
            if (c["config"], p["prompt_id"]) not in done]
    lock = threading.Lock()
    n_ok = n_err = 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(ask_one, c, p, timeout) for c, p in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            a = fut.result()
            with lock:
                f.write(json.dumps(a) + "\n")
                f.flush()
                if a.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
            if on_answer:
                on_answer(a)
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} answered ({n_err} errors)", flush=True)
    return n_ok, n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pod", default="")
    ap.add_argument("--only", help="comma-separated config names")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
    configs = json.load(open(args.configs))
    for c in configs:
        c["base_url"] = c["base_url"].replace("{pod}", args.pod)
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c["config"] in keep]
    ok, err = run(prompts, configs, args.out, args.workers, args.timeout, args.limit)
    print(f"{ok} answers, {err} errors -> {args.out}")


if __name__ == "__main__":
    main()
