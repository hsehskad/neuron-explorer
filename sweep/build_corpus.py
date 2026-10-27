"""Builds the stats corpus once, as a local token file, so neuron_stats_mlx.py never streams from the Hub.

Streaming fineweb-edu and ultrachat through `datasets` held 4 GB before the first window and then hung in an arrow
thread pool, which on a 16 GB Mac is enough to push the model run into swap. This downloads one parquet shard of
each (about 2.3 GB), reads them a few rows at a time, and packs the same 1:1 web/chat interleave with EOS separators
that neuron_stats.py uses. Row groups are read in a seeded random order in place of the streaming shuffle.

    uv run python sweep/build_corpus.py --tokens 2100000 --out stats/corpus-q35.npy
"""
import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

WEB = ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", "text")
CHAT = ("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00003-a3ecf92756993583.parquet", "messages")


def rows(spec, seed):
    repo, name, col = spec
    f = pq.ParquetFile(hf_hub_download(repo, name, repo_type="dataset"))
    for g in np.random.default_rng(seed).permutation(f.num_row_groups):
        for batch in f.iter_batches(batch_size=32, row_groups=[int(g)], columns=[col], use_threads=False):
            yield from batch.column(0).to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tokens", type=int, default=2_100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="stats/corpus-q35.npy")
    a = ap.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)

    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.eos_token_id
    buf = np.empty(a.tokens, np.uint32)
    n, docs = 0, 0
    for w, c in zip(rows(WEB, a.seed), rows(CHAT, a.seed)):
        for text in (w, tok.apply_chat_template(c, tokenize=False, enable_thinking=False)):
            ids = tok(text, add_special_tokens=False)["input_ids"] + [eos]
            take = min(len(ids), a.tokens - n)
            buf[n : n + take] = ids[:take]
            n += take
            docs += 1
        if n >= a.tokens:
            break
    if n < a.tokens:
        raise SystemExit(f"only {n:,} tokens available from one shard each; ask for fewer")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp.npy"
    np.save(tmp, buf)
    os.replace(tmp, a.out)
    json.dump({"model": a.model, "tokens": n, "docs": docs, "seed": a.seed, "web": WEB[:2], "chat": CHAT[:2]},
              open(a.out.replace(".npy", ".json"), "w"), indent=1)
    print(f"{n:,} tokens from {docs:,} documents -> {a.out}")


if __name__ == "__main__":
    main()
