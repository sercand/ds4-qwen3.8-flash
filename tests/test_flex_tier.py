#!/usr/bin/env python3
"""Flex tier end-to-end check against a running ds4-server (--exec-contexts 2).

Scenario A: a streaming flex request is generating; a normal request arrives.
  The flex stream must go silent (only ': keepalive' comments) from the normal's
  first token to its completion, then resume and finish.  Greedy, so the flex
  text must equal a solo flex run.
Scenario B: two flex requests -> the second is not admitted (flex_cap 1);
  a normal then starts within a bounded TTFT, and the queued flex runs later.

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
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").rstrip("\n")
            now = time.time()
            if line.startswith(": "):
                events.append((now, "comment", line))
            elif line.startswith("data: ") and line != "data: [DONE]":
                obj = json.loads(line[6:])
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if delta:
                    events.append((now, "token", delta))
                if obj.get("service_tier"):
                    events.append((now, "tier", obj["service_tier"]))


def payload(model, text, max_tokens, flex_body=False):
    p = {"model": model, "messages": [{"role": "user", "content": text}],
         "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
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
    while len(tokens(flex_events)) < 8:
        time.sleep(0.05)
    t_normal0 = time.time()
    sse_stream(base, payload(model, NORMAL_PROMPT, 48), {}, normal_events)
    t_normal1 = time.time()
    t.join()

    normal_first = min(ts for ts, _, _ in tokens(normal_events))
    # Allow one flex step already granted when the normal arrived.
    leaked = [ts for ts, _, _ in tokens(flex_events)
              if normal_first + 0.25 < ts < t_normal1 - 0.05]
    flex_text = "".join(d for _, _, d in tokens(flex_events))
    print(f"A: normal TTFT {normal_first - t_normal0:.2f}s, normal wall "
          f"{t_normal1 - t_normal0:.2f}s, flex tokens leaked during normal: "
          f"{len(leaked)}, flex tokens total {len(tokens(flex_events))}")
    assert not leaked, f"flex emitted {len(leaked)} tokens while a normal request ran"
    assert flex_text == solo_text, "flex text diverged from the solo run"
    return True


def scenario_b(base, model):
    ev1, ev2, evn = [], [], []
    t1 = threading.Thread(target=sse_stream,
                          args=(base, payload(model, FLEX_PROMPT, 120, flex_body=True), {}, ev1))
    t2 = threading.Thread(target=sse_stream,
                          args=(base, payload(model, FLEX_PROMPT + " Second.", 120, flex_body=True),
                                {}, ev2))
    t1.start()
    while not tokens(ev1):
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
    ok = scenario_a(a.base, a.model) and scenario_b(a.base, a.model)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
