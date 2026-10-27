"""neuron_stats.py for Apple Silicon: the same per-neuron activation ranges and npz keys, computed with MLX.

The corpus is a local token file from build_corpus.py (streaming the datasets during the run used 4 GB on its own
and once took a 16 GB Mac down). Progress is checkpointed, so a killed run resumes where it stopped.

    uv run python sweep/build_corpus.py --tokens 2100000 --out stats/corpus-q35.npy
    uv run python sweep/memguard.py --max-gb 10 -- \
        uv run python sweep/neuron_stats_mlx.py --corpus stats/corpus-q35.npy --layers 22 --tokens 2000000 --out stats/q35-4b

abs_max is a running maximum, so it grows with the token budget: stats from far fewer tokens than upstream's 2M
make absmax-unit doses stronger than upstream's. Rel-unit doses (--dose-unit rel) only use the residual norm median.
"""
import argparse
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mlx_qwen import DEFAULT_MLX_MODEL, hf_down_columns, limit_memory, load_model, text_layers  # noqa: E402
from mlx_lm.models.base import create_attention_mask  # noqa: E402

KEYS = ("pp", "nn", "pn", "np")


def save_atomic(path, **arrays):
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **arrays)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="HF repo: tokenizer, W_down and the npz's model field")
    ap.add_argument("--mlx-model", default=DEFAULT_MLX_MODEL, help="MLX weights to run (default: the 8-bit conversion)")
    ap.add_argument("--allow-full", action="store_true", help="allow unquantized weights on a Mac under 24 GB")
    ap.add_argument("--corpus", required=True, help="uint32 token file from build_corpus.py")
    ap.add_argument("--layers", default="22")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem-gb", type=float, default=8.0, help="MLX memory limit (the model is ~4.2 GB at 8 bits)")
    ap.add_argument("--checkpoint-every", type=int, default=20, help="steps between resumable checkpoints")
    ap.add_argument("--out", default="stats/q35-4b")
    a = ap.parse_args()
    targets = sorted(int(x) for x in a.layers.split(","))
    limit_memory(a.mem_gb)

    corpus = np.load(a.corpus, mmap_mode="r")
    per_step = a.seq_len * a.batch
    n_steps = -(-a.tokens // per_step)
    if n_steps * per_step > len(corpus):
        raise SystemExit(f"{a.corpus} has {len(corpus):,} tokens; {n_steps} steps of {per_step} need "
                         f"{n_steps * per_step:,}. Rebuild it with more --tokens.")
    run_id = {"model": a.model, "engine": f"mlx:{a.mlx_model}/{a.dtype}", "corpus": os.path.abspath(a.corpus),
              "seq_len": a.seq_len, "batch": a.batch}

    w_down = {l: hf_down_columns(a.model, l) for l in targets}  # fetched first so a failure costs nothing
    model = load_model(a.mlx_model, a.dtype, a.allow_full)
    layers = text_layers(model)[: max(targets) + 1]
    embed = model.language_model.model.embed_tokens
    I = model.language_model.args.intermediate_size
    assert all(w.shape[0] == I for w in w_down.values()), "W_down from --model doesn't match --mlx-model"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ckpt = f"{a.out}_L{'-'.join(map(str, targets))}.partial.npz"
    S = {l: {k: mx.zeros((I,), mx.float32) for k in KEYS} | {"resid": []} for l in targets}
    start = 0
    if os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=False)
        if json.loads(str(z["run_id"])) != run_id:
            raise SystemExit(f"{ckpt} is from a different setup; delete it to start over")
        start = int(z["steps_done"])
        for l in targets:
            S[l].update({k: mx.array(z[f"L{l}_{k}"]) for k in KEYS})
            S[l]["resid"] = [z[f"L{l}_resid"]]
        print(f"resuming at step {start}/{n_steps} from {ckpt}", flush=True)

    def checkpoint(done):
        arrays = {"run_id": json.dumps(run_id), "steps_done": done}
        for l in targets:
            arrays |= {f"L{l}_{k}": np.array(S[l][k]) for k in KEYS}
            arrays[f"L{l}_resid"] = np.concatenate(S[l]["resid"]) if S[l]["resid"] else np.zeros(0, np.float32)
        save_atomic(ckpt, **arrays)

    def run(ids):
        h = embed(mx.array(ids))
        fa_mask = create_attention_mask(h, None)
        for i, layer in enumerate(layers):
            mask = None if layer.is_linear else fa_mask
            if i not in S:
                h = layer(h, mask=mask, cache=None)
            else:  # DecoderLayer.__call__ written out, to see gate and up
                x_in = layer.input_layernorm(h)
                h = h + (layer.linear_attn(x_in, mask, None) if layer.is_linear else layer.self_attn(x_in, mask, None))
                mlp = layer.mlp
                xn = layer.post_attention_layernorm(h)
                g, u = mlp.gate_proj(xn), mlp.up_proj(xn)
                act = nn.silu(g) * u
                h = h + mlp.down_proj(act)
                s = S[i]
                # position 0 of each window is an attention sink with outsized activations
                g, u, act = (t[:, 1:].reshape(-1, I).astype(mx.float32) for t in (g, u, act))
                gp, upos = g > 0, u > 0
                ninf, pinf = mx.array(-mx.inf), mx.array(mx.inf)
                s["pp"] = mx.maximum(s["pp"], mx.where(gp & upos, act, ninf).max(0))
                s["nn"] = mx.maximum(s["nn"], mx.where(~gp & ~upos, act, ninf).max(0))
                s["pn"] = mx.minimum(s["pn"], mx.where(gp & ~upos, act, pinf).min(0))
                s["np"] = mx.minimum(s["np"], mx.where(~gp & upos, act, pinf).min(0))
                norm = mx.linalg.norm(h[:, 1:].astype(mx.float32), axis=-1).reshape(-1)
                mx.eval(*(s[k] for k in KEYS), norm)
                s["resid"].append(np.array(norm))
            mx.eval(h)  # one layer's graph at a time

    t0 = time.time()
    for step in range(start, n_steps):
        run(np.asarray(corpus[step * per_step : (step + 1) * per_step], np.int64).reshape(a.batch, a.seq_len))
        done = step + 1
        if done % a.checkpoint_every == 0 or done == n_steps:
            checkpoint(done)
        rate = (done - start) * per_step / (time.time() - t0)
        print(f"{done}/{n_steps} steps, {done * per_step:,} tokens, {rate:,.0f} tok/s, "
              f"mlx active {mx.get_active_memory() / 2**30:.2f} GB peak {mx.get_peak_memory() / 2**30:.2f} GB",
              flush=True)

    for l, s in S.items():
        pp, nn_, pn, np_ = (np.array(s[k]) for k in KEYS)
        abs_max = np.maximum.reduce([pp, nn_, -pn, -np_])
        if not all(np.isfinite(x).all() for x in (pp, nn_, pn, np_)):
            raise SystemExit(f"L{l}: non-finite activations; rerun with --dtype bfloat16 (checkpoint kept)")
        resid_q = np.quantile(np.concatenate(s["resid"]), [0.1, 0.5, 0.9])
        path = f"{a.out}_L{l}.npz"
        save_atomic(path, pp_max=pp, nn_max=nn_, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down[l],
                    resid_norm_q10_50_90=resid_q, model=a.model, engine=run_id["engine"], layer=l,
                    tokens=n_steps * per_step)
        print(f"L{l}: residual norm p10/p50/p90 = {resid_q.round(1)}; abs_max median {np.median(abs_max):.3f} "
              f"[{abs_max.min():.3f}, {abs_max.max():.3f}]; (-,-) holds the positive max for {(nn_ > pp).mean():.1%}; "
              f"{(abs_max == 0).sum()} neurons never fired -> {path}")
    os.remove(ckpt)


if __name__ == "__main__":
    main()
