#!/usr/bin/env python3
"""Token frequencies for ds4's reduced-vocabulary MTP drafting (qwen4exp).

Counts how often every token id occurs in a corpus and writes `id count`
lines, most frequent first.  ds4 (--mtp-vocab FILE) reads this file at load
and slices the draft head down to the ids that carry most of the mass, at the
granularity its weight format allows (EXL3: 128-row blocks, the output
Hadamard couples them).  Counts, not a fixed id list, so the slicer can score
blocks itself.

The drafter predicts the model's *output* distribution, so the corpus should
look like what the model writes: prose, code, markdown, tool-call JSON.  A
token the vocabulary misses costs one rejected draft, never a wrong token.

  python3 gguf-tools/qwen4exp_draft_vocab.py --tokenizer DIR_OR_JSON \
      --out draft_vocab.txt CORPUS [CORPUS ...]

Each CORPUS is a file or a directory (walked for text-like files).  A path
may carry ':N' to weight it N times.  --max-bytes-per-source caps how much of
each source is read (default 64 MiB).
"""
import argparse, os, sys, collections, io

TEXT_EXT = {'.txt', '.md', '.rst', '.py', '.c', '.h', '.cu', '.cuh', '.cpp', '.hpp', '.json',
            '.js', '.ts', '.sh', '.toml', '.yaml', '.yml', '.html', '.css', '.go', '.rs', '.java'}

def iter_files(path):
    if os.path.isfile(path):
        yield path
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'third_party')]
        for f in files:
            if os.path.splitext(f)[1].lower() in TEXT_EXT or f.upper().startswith('README'):
                yield os.path.join(root, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', nargs='+')
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-bytes-per-source', type=int, default=64 << 20)
    ap.add_argument('--chunk', type=int, default=1 << 20)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok_path = args.tokenizer
    if os.path.isdir(tok_path):
        tok_path = os.path.join(tok_path, 'tokenizer.json')
    tok = Tokenizer.from_file(tok_path)
    vocab_size = tok.get_vocab_size(with_added_tokens=True)

    counts = collections.Counter()
    per_source = []
    for spec in args.corpus:
        weight = 1
        path = spec
        if ':' in spec and spec.rsplit(':', 1)[1].isdigit():
            path, w = spec.rsplit(':', 1)
            weight = int(w)
        read = 0
        local = collections.Counter()
        for f in iter_files(path):
            if read >= args.max_bytes_per_source:
                break
            try:
                with open(f, 'rb') as h:
                    data = h.read(min(args.max_bytes_per_source - read, 8 << 20))
            except OSError:
                continue
            if b'\0' in data[:4096]:
                continue
            read += len(data)
            text = data.decode('utf-8', 'replace')
            for i in range(0, len(text), args.chunk):
                local.update(tok.encode(text[i:i + args.chunk], add_special_tokens=False).ids)
        total = sum(local.values())
        per_source.append((spec, read, total))
        for k, v in local.items():
            counts[k] += v * weight
        print(f'  {spec}: {read / 1048576:.1f} MiB, {total:,} tokens, weight {weight}', file=sys.stderr, flush=True)

    total = sum(counts.values())
    if not total:
        print('no tokens', file=sys.stderr)
        sys.exit(1)
    ranked = counts.most_common()
    print(f'{len(ranked):,} distinct ids of {vocab_size:,}; {total:,} weighted occurrences', file=sys.stderr)
    cum = 0
    marks = {8192, 16384, 32768, 49152, 65536, 98304, 131072}
    for i, (_, c) in enumerate(ranked, 1):
        cum += c
        if i in marks:
            print(f'  top {i:>7,} ids cover {100.0 * cum / total:7.3f}%', file=sys.stderr)
    # 128-aligned blocks (EXL3 head granularity)
    blocks = collections.Counter()
    for k, v in counts.items():
        blocks[k // 128] += v
    bcum = 0
    for i, (_, c) in enumerate(blocks.most_common(), 1):
        bcum += c
        if i * 128 in marks:
            print(f'  top {i:>5,} 128-blocks ({i * 128:>7,} ids) cover {100.0 * bcum / total:7.3f}%', file=sys.stderr)
    with open(args.out, 'w') as h:
        h.write(f'# qwen4exp draft vocabulary counts: {total} occurrences, {len(ranked)} ids, vocab {vocab_size}\n')
        for k, c in ranked:
            h.write(f'{k} {c}\n')
    print(f'wrote {args.out}', file=sys.stderr)

if __name__ == '__main__':
    main()
