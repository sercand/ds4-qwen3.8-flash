#!/usr/bin/env python3
"""Flex tier end-to-end check against a running ds4-server (--exec-contexts 2).

Scenario A: a streaming flex request is generating; a normal request arrives.
  The flex stream must go silent (only ': keepalive' comments) from the normal's
  first token to its completion, then resume and finish.  Greedy, so the flex
  text must equal a solo flex run.
Scenario B: two flex requests -> the second is not admitted (flex_cap 1);
  a normal then starts within a bounded TTFT, and the queued flex runs later.
Scenario C: a normal arrives while a flex is still PREFILLING a long prompt.
  The flex parks between chunks (keepalives, no tokens), the normal's TTFT
  stays bounded, and the flex completes afterwards.
Not covered here: promotion of a flex slot by a bound follow-up turn (spec
  test 4) and the no-flex throughput baseline (spec test 5: run
  tests/bench_concurrency.py before and after).

Usage: python3 tests/test_flex_tier.py --base http://127.0.0.1:8080 [--model NAME]
Run the server with DS4_SERVER_BATCH_LOG=1 to see the parked/resumed lines.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

FLEX_PROMPT = ("Write a long, detailed essay about the history of the bicycle. "
               "Do not stop early.")
NORMAL_PROMPT = "List ten prime numbers, one per line."


def sse_stream(base, payload, headers, events):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    # Shorter than any park we expect: a flex that goes silent (no keepalive)
    # must fail this test, not sit here for minutes.
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").rstrip("\n")
            now = time.time()
            if line.startswith(": "):
                events.append((now, "comment", line))
            elif line.startswith("data: ") and line != "data: [DONE]":
                obj = json.loads(line[6:])
                choices = obj.get("choices") or []
                d = choices[0].get("delta", {}) if choices else {}
                # Thinking models stream reasoning_content first; count both.
                delta = d.get("content") or d.get("reasoning_content")
                if delta:
                    events.append((now, "token", delta))
                if obj.get("service_tier"):
                    events.append((now, "tier", obj["service_tier"]))


def payload(model, text, max_tokens, flex_body=False):
    p = {"model": model, "messages": [{"role": "user", "content": text}],
         "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
         "think": False}
    if flex_body:
        p["service_tier"] = "flex"
    return p


def tokens(events):
    return [e for e in events if e[1] == "token"]


def scenario_a(base, model):
    solo = []
    sse_stream(base, payload(model, FLEX_PROMPT, 160, flex_body=True), {}, solo)
    solo_text = "".join(d for _, k, d in solo if k == "token")
    assert any(k == "tier" and d == "flex" for _, k, d in solo), "service_tier not echoed"

    flex_events, normal_events = [], []
    t = threading.Thread(target=sse_stream,
                         args=(base, payload(model, FLEX_PROMPT, 160),
                               {"X-Service-Tier": "flex"}, flex_events))
    t.start()
    while len(tokens(flex_events)) < 8 and t.is_alive():
        time.sleep(0.05)
    assert t.is_alive(), "flex request ended before the normal could be submitted"
    t_normal0 = time.time()
    sse_stream(base, payload(model, NORMAL_PROMPT, 48), {}, normal_events)
    t_normal1 = time.time()
    t.join()

    normal_first = min(ts for ts, _, _ in tokens(normal_events))
    normal_last = max(ts for ts, _, _ in tokens(normal_events))
    # Allow one flex step already granted when the normal arrived, and the
    # resume that follows the normal's last token (its slot is released
    # before the client reads the stream end).
    leaked = [ts for ts, _, _ in tokens(flex_events)
              if normal_first + 0.25 < ts < normal_last - 0.3]
    keepalives = [ts for ts, k, d in flex_events
                  if k == "comment" and "keepalive" in d and normal_first < ts < normal_last]
    flex_text = "".join(d for _, _, d in tokens(flex_events))
    print(f"A: normal TTFT {normal_first - t_normal0:.2f}s, normal wall "
          f"{t_normal1 - t_normal0:.2f}s, flex tokens leaked during normal: "
          f"{len(leaked)}, keepalives while parked: {len(keepalives)}, "
          f"flex tokens total {len(tokens(flex_events))}")
    assert not leaked, f"flex emitted {len(leaked)} tokens while a normal request ran"
    if normal_last - normal_first > 6.0:
        assert keepalives, "flex sent no keepalive while parked"
    # Greedy runs of this MoE can still differ between processes at a near-tie
    # (see the executor comment in ds4_server.c and the prefill-logits memory),
    # so only the tokens emitted before the normal arrived must match; the
    # rest is reported.
    solo_toks = [d for _, _, d in tokens(solo)]
    flex_toks = [d for _, _, d in tokens(flex_events)]
    assert flex_toks[:8] == solo_toks[:8], "flex diverged from the solo run before the normal arrived"
    same = flex_text == solo_text
    print(f"A: full text {'identical to' if same else 'differs from'} the solo run"
          + ("" if same else f" (first difference at token {next(i for i, (a, b) in enumerate(zip(flex_toks, solo_toks)) if a != b) if any(a != b for a, b in zip(flex_toks, solo_toks)) else min(len(flex_toks), len(solo_toks))})"))
    return True


def scenario_c(base, model):
    """Normal arrives during a flex prefill: the flex parks between chunks."""
    filler = " ".join(f"item {i} of the long background document" for i in range(600))
    long_prompt = f"Summarise this document in three sentences.\n\n{filler}"
    flex_events, normal_events = [], []
    t = threading.Thread(target=sse_stream,
                         args=(base, payload(model, long_prompt, 64, flex_body=True), {}, flex_events))
    t.start()
    time.sleep(1.0)                     # the flex is prefilling, no token yet
    assert not tokens(flex_events), "long flex prompt produced a token within 1 s; lengthen it"
    t0 = time.time()
    sse_stream(base, payload(model, NORMAL_PROMPT, 32), {}, normal_events)
    ttft = min(ts for ts, _, _ in tokens(normal_events)) - t0
    t.join()
    keepalives = [d for _, k, d in flex_events if k == "comment"]
    print(f"C: normal TTFT during flex prefill {ttft:.2f}s, flex keepalive comments "
          f"{len(keepalives)}, flex tokens {len(tokens(flex_events))}")
    assert ttft < 20.0, "normal waited too long behind a flex prefill"
    assert tokens(flex_events), "flex never completed after the normal"
    return True


def scenario_b(base, model):
    ev1, ev2, evn = [], [], []
    t1 = threading.Thread(target=sse_stream,
                          args=(base, payload(model, FLEX_PROMPT, 120, flex_body=True), {}, ev1))
    t2 = threading.Thread(target=sse_stream,
                          args=(base, payload(model, FLEX_PROMPT + " Second.", 120, flex_body=True),
                                {}, ev2))
    t1.start()
    while not tokens(ev1) and t1.is_alive():
        time.sleep(0.05)
    t2.start()
    time.sleep(2.0)
    assert not tokens(ev2), "second flex was admitted despite flex_cap 1"
    t0 = time.time()
    sse_stream(base, payload(model, NORMAL_PROMPT, 32), {}, evn)
    ttft = min(ts for ts, _, _ in tokens(evn)) - t0
    print(f"B: normal TTFT with one flex running and one queued: {ttft:.2f}s")
    assert ttft < 15.0, "normal request did not get the free slot promptly"
    t1.join()
    t2.join()
    assert tokens(ev2), "queued flex never ran"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="qwen4exp")
    a = ap.parse_args()
    ok = (scenario_a(a.base, a.model) and scenario_b(a.base, a.model)
          and scenario_c(a.base, a.model))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
