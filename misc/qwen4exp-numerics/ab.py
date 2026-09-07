"""Fire N concurrent distinct greedy chat requests at a ds4-server and report
aggregate decode throughput.  usage: ab.py <port> <concurrency> [max_tokens]"""
import json, sys, time, threading, urllib.request, hashlib
PORT = int(sys.argv[1]); CONC = int(sys.argv[2]); MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 300
PROMPTS = [
    "List the first 60 prime numbers separated by commas.",
    "Explain in detail how an internal combustion engine works, step by step.",
    "Write a 200-word story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe the water cycle for a ten-year-old, with one example per stage.",
    "Summarize the causes of the First World War in about 250 words.",
    "Give a recipe for a vegetable soup, with quantities and timings.",
    "Explain what a hash table is and when you would not use one.",
    "Write a short dialogue between a detective and a suspect about a missing painting.",
][:CONC]
def call(p, out, idx):
    body = json.dumps({"model": "qwen3.8-flash", "messages": [{"role": "user", "content": p}],
                       "max_tokens": MAXTOK, "temperature": 0, "stream": False}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                               body, {"content-type": "application/json"}), timeout=600)
    j = json.load(r); t = time.time() - t0
    m = j["choices"][0]["message"]
    txt = (m.get("content") or "") + " <REASON> " + (m.get("reasoning_content") or "")
    out[idx] = (txt, j.get("usage", {}).get("completion_tokens", 0), t)
out = {}
t_all = time.time()
ths = [threading.Thread(target=call, args=(p, out, i)) for i, p in enumerate(PROMPTS)]
[t.start() for t in ths]; [t.join() for t in ths]
wall = time.time() - t_all
tot = sum(out[i][1] for i in out)
print(f"conc={CONC} wall={wall:.2f}s total_tokens={tot} aggregate={tot/wall:.1f} tok/s")
for i in sorted(out):
    print(f"  req{i}: tokens={out[i][1]} time={out[i][2]:.2f}s per-stream={out[i][1]/out[i][2]:.1f} tok/s sha={hashlib.sha1(out[i][0].encode()).hexdigest()[:10]}")
