import os, sys, struct, numpy as np
base = sys.argv[1]; N = int(sys.argv[2]); thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
A, B = os.path.join(base, 'A'), os.path.join(base, 'B')
files = sorted(os.listdir(A), key=lambda f: os.stat(os.path.join(A, f)).st_mtime_ns)
print(f"{'tensor':28s} {'relL2':>10s} {'maxabs':>10s}  {'|A|rms':>8s}")
shown = 0
for f in files:
    pa, pb = os.path.join(A, f), os.path.join(B, f)
    if not os.path.exists(pb): continue
    a = np.fromfile(pa, dtype=np.float32); b = np.fromfile(pb, dtype=np.float32)
    if b.size == a.size * N: b = b[-a.size:]
    elif b.size != a.size: print(f"{f:28s} size A={a.size} B={b.size} (skip)"); continue
    d = b.astype(np.float64) - a.astype(np.float64)
    rel = np.sqrt((d*d).sum() / max((a.astype(np.float64)**2).sum(), 1e-30))
    flag = "  <==" if rel > thresh else ""
    if rel > thresh or shown < 6 or f.startswith(("result", "output")):
        print(f"{f[:-4]:28s} {rel:10.2e} {np.abs(d).max():10.4f}  {np.sqrt((a*a).mean()):8.3f}{flag}")
        shown += 1
# routing: last 48 route records per step
R = os.path.join(base, 'route')
if os.path.isdir(R):
    recs = sorted(os.listdir(R))
    def load(p):
        with open(p, 'rb') as fp:
            h = struct.unpack('<6i', fp.read(24)); ids = np.frombuffer(fp.read(h[2]*h[3]*4), dtype=np.int32).reshape(h[2], h[3])
        return h[1], ids
    # A step is the 48 records ending at the A/B boundary; B step is the last 48
    b_recs = recs[-48:]; a_recs = recs[-96:-48]
    flips = []
    for ra, rb in zip(a_recs, b_recs):
        la, ia = load(os.path.join(R, ra)); lb, ib = load(os.path.join(R, rb))
        sa, sb = set(ia[-1].tolist()), set(ib[-1].tolist())
        if sa != sb: flips.append((la, sorted(sa - sb), sorted(sb - sa)))
    print(f"routing: layers with a different expert set at the last row: {len(flips)}")
    for l, only_a, only_b in flips[:10]: print(f"  layer {l}: seq-only {only_a} prefill-only {only_b}")
