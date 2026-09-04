#!/usr/bin/env python3
"""Concurrent-request throughput for ds4-server.

Runs N streaming completions at once and reports, per request and in
aggregate: time to first token, decode tokens/s, and the per-step gap
distribution.  The point is the shape of the scaling curve, so the default
sweep is 1, 2, 4 and 8 clients over the same prompt set.

  ./tests/bench_concurrency.py --url http://127.0.0.1:8899 --sweep 1,2,4,8

Each client gets its own prompt so no two share a prefix: a shared prefix
would be answered from the prompt cache and measure the cache, not decode.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

# Distinct openings so every client prefills its own path through the cache.
TOPICS = [
    "the history of the Byzantine water supply",
    "how a sodium-ion battery differs from lithium-ion",
    "the grammar of evidentiality in Turkish",
    "why bridges are built with expansion joints",
    "the role of yeast strains in Belgian brewing",
    "how radio astronomers calibrate an interferometer",
    "the economics of container shipping routes",
    "what a compiler does during register allocation",
    "the domestication history of the horse",
    "how tides are predicted at a given harbour",
    "the metallurgy of Damascus steel",
    "why some volcanoes erupt explosively",
    "the design of the Roman road network",
    "how error-correcting codes protect stored data",
    "the migration of the Arctic tern",
    "what makes a violin's tone distinctive",
]


class Result:
    def __init__(self, idx):
        self.idx = idx
        self.ttft = None          # seconds to first streamed piece
        self.start = None
        self.end = None
        self.tokens = 0           # exact, from the usage chunk
        self.chunks = 0           # streamed pieces; one per decode step
        self.gaps = []            # inter-chunk arrival gaps, seconds
        self.error = None


def run_one(url, model, prompt, max_tokens, res, barrier):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    res.start = time.perf_counter()
    last = res.start
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                if not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    res.tokens = usage["completion_tokens"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                # Thinking models stream the chain of thought under
                # reasoning_content; both are decode output.
                piece = delta.get("content") or delta.get("reasoning_content")
                if not piece:
                    continue
                now = time.perf_counter()
                if res.ttft is None:
                    res.ttft = now - res.start
                else:
                    res.gaps.append(now - last)
                last = now
                res.chunks += 1
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        res.error = repr(exc)
    res.end = time.perf_counter()


def sweep_once(url, model, n, max_tokens, prompt_words):
    prompts = [
        f"Write about {TOPICS[i % len(TOPICS)]}. "
        f"Give roughly {prompt_words} words of background first, then continue at length."
        for i in range(n)
    ]
    results = [Result(i) for i in range(n)]
    barrier = threading.Barrier(n)
    threads = [
        threading.Thread(target=run_one,
                         args=(url, model, prompts[i], max_tokens, results[i], barrier))
        for i in range(n)
    ]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start
    return results, wall


def report(n, results, wall):
    ok = [r for r in results if r.error is None and r.tokens > 0]
    failed = [r for r in results if r.error is not None]
    print(f"\n=== {n} concurrent ===")
    for r in failed:
        print(f"  req {r.idx}: ERROR {r.error}")
    if not ok:
        print("  no successful requests")
        return None

    total_tokens = sum(r.tokens for r in ok)
    print(f"  {'req':>4} {'ttft_s':>8} {'decode_s':>9} {'tok':>6} {'tok/s':>8} "
          f"{'steps':>6} {'step_p50_ms':>12} {'step_p99_ms':>12}")
    for r in ok:
        decode_s = r.end - r.start - r.ttft
        rate = r.tokens / decode_s if decode_s > 0 else 0.0
        p50 = statistics.median(r.gaps) * 1000 if r.gaps else 0.0
        p99 = (statistics.quantiles(r.gaps, n=100)[98] * 1000
               if len(r.gaps) >= 100 else max(r.gaps) * 1000 if r.gaps else 0.0)
        print(f"  {r.idx:>4} {r.ttft:>8.3f} {decode_s:>9.3f} {r.tokens:>6} "
              f"{rate:>8.2f} {r.chunks:>6} {p50:>12.2f} {p99:>12.2f}")

    agg = total_tokens / wall
    ttfts = sorted(r.ttft for r in ok)
    print(f"  ---- wall {wall:.2f}s  total_tokens {total_tokens}  "
          f"AGGREGATE {agg:.2f} tok/s")
    print(f"  ---- ttft  min {ttfts[0]:.3f}  median {statistics.median(ttfts):.3f}  "
          f"max {ttfts[-1]:.3f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--model", default="ds4")
    ap.add_argument("--sweep", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--prompt-words", type=int, default=60)
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds between sweep points")
    args = ap.parse_args()

    counts = [int(x) for x in args.sweep.split(",") if x.strip()]
    aggs = {}
    for n in counts:
        results, wall = sweep_once(args.url, args.model, n,
                                   args.max_tokens, args.prompt_words)
        agg = report(n, results, wall)
        if agg is not None:
            aggs[n] = agg
        time.sleep(args.settle)

    if aggs:
        base_n = min(aggs)
        base = aggs[base_n]
        print("\n=== scaling ===")
        print(f"  {'clients':>8} {'agg_tok/s':>11} {'vs_' + str(base_n):>9} {'ideal':>8}")
        for n in sorted(aggs):
            print(f"  {n:>8} {aggs[n]:>11.2f} {aggs[n] / base:>9.2f}x "
                  f"{n / base_n:>7.0f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
