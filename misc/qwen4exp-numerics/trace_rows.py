import os, sys, numpy as np
base=sys.argv[1]; N=int(sys.argv[2]); layers=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else [0,1,2,3]
tensors=sys.argv[4].split(',') if len(sys.argv)>4 else ['hc_mixed','ffn_moe_logits','ffn_moe_out','attn_output','attn_gated','l_last']
def load(p):
    return np.fromfile(p,dtype=np.float32).astype(np.float64) if os.path.exists(p) else None
for L in layers:
    for t in tensors:
        b=load(f'{base}/B/{t}-{L}.bin')
        if b is None: continue
        refs=[load(f'{base}/A{r}/{t}-{L}.bin') for r in range(N)]
        if any(r is None for r in refs): continue
        row=refs[0].size
        if b.size!=row*N: print(f'{t}-{L}: size mismatch B={b.size} row={row}'); continue
        rel=[]
        for r in range(N):
            d=b[r*row:(r+1)*row]-refs[r]; rel.append(np.sqrt((d*d).sum()/max((refs[r]**2).sum(),1e-30)))
        print(f'{t+"-"+str(L):22s} '+' '.join(f'{v:7.0e}' for v in rel))
