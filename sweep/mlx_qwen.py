"""Qwen3.5 on Apple Silicon through MLX: loading, a per-row steering push at one MLP layer, and batched greedy decoding
with per-step logprobs. Shared by steer_sweep_mlx.py and neuron_stats_mlx.py.

Rows of a batch share one prompt and differ only in their push, so there is no padding: three of every four layers
are GatedDeltaNet with recurrent state, which left-padding would corrupt.
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from mlx_lm import load

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
# made with: mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 8 --mlx-path ~/.cache/mlx-models/Qwen3.5-4B-q8
DEFAULT_MLX_MODEL = os.path.expanduser("~/.cache/mlx-models/Qwen3.5-4B-q8")


def limit_memory(gb):
    """Caps MLX. Past the memory limit MLX waits for queued work to free buffers before allocating more, and the cache
    limit stops it hoarding freed buffers. Neither is a hard ceiling (MLX still swaps rather than fail), so long runs
    also go under sweep/memguard.py. The default limit is 1.5x the GPU working set, 17 GB on a 16 GB M1."""
    mx.set_memory_limit(int(gb * 2**30))
    mx.set_cache_limit(int(min(1.0, gb / 8) * 2**30))


def _config(path):
    f = os.path.join(path, "config.json") if os.path.isdir(path) else hf_hub_download(path, "config.json")
    return json.load(open(f))


def load_model(path, dtype="bfloat16", allow_full=False):
    """path: a local mlx_lm.convert output (the 8-bit one fits a 16 GB Mac) or an HF repo id. Unquantized weights
    take about 9.3 GB for the 4B model, which leaves a 16 GB Mac no headroom, so they need allow_full."""
    if not _config(path).get("quantization") and not allow_full and mx.device_info()["memory_size"] < 24 * 2**30:
        raise SystemExit(f"{path} is unquantized (~2 bytes/param) and this Mac has under 24 GB; pass an 8-bit "
                         f"--mlx-model or --allow-full to load it anyway")
    model, _ = load(path)
    if dtype != "bfloat16":
        model.set_dtype(DTYPES[dtype])
    return model


def hf_down_columns(model_id, layer):
    """W_down columns of one layer, [I, H] float16, read from the original HF weights (a quantized MLX model's are
    packed). Steering pushes use these, so a push is the same vector whatever precision the model runs at. Only the
    shard holding this layer is fetched."""
    from safetensors import safe_open

    index = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json")))["weight_map"]
    name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    with safe_open(hf_hub_download(model_id, index[name]), framework="pt") as f:
        return f.get_tensor(name).float().T.numpy().astype(np.float16)


def text_layers(model):
    return model.language_model.model.layers


def eos_ids(model_id, tok):
    """generation_config's eos ids when the repo has one, else the tokenizer's (what vLLM stops on)."""
    path = os.path.join(snapshot_download(model_id, allow_patterns=["*.json"]), "generation_config.json")
    if os.path.exists(path):
        e = json.load(open(path)).get("eos_token_id", [])
        return set(e if isinstance(e, list) else [e])
    return {tok.eos_token_id}


class SteeredMLP(nn.Module):
    """Wraps an MLP: adds push[b] to row b's MLP output at every position (prompt and generated). Adding
    add * W_down[:, n] here is the same as adding `add` to neuron n's activation."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.push = None  # [B, H] or None

    def __call__(self, x):
        out = self.inner(x)
        if self.push is not None:
            out = out + self.push[:, None, :].astype(out.dtype)
        return out


def install_steering(model, layer):
    layers = text_layers(model)
    if not isinstance(layers[layer].mlp, SteeredMLP):
        layers[layer].mlp = SteeredMLP(layers[layer].mlp)
    return layers[layer].mlp


def filter_cache(cache, idx):
    idx = mx.array(idx)
    for c in cache:
        if hasattr(c, "filter") and not hasattr(c, "keys"):
            c.filter(idx)
        elif getattr(c, "keys", None) is not None:
            c.keys, c.values = c.keys[idx], c.values[idx]


def generate_batch(model, steer, prompt_ids, pushes, max_tokens, top_k, eos):
    """Greedy-decodes len(pushes) rows of one prompt. pushes: [B, H] float32 numpy (zeros = unsteered).
    Returns per row: token ids (EOS included when hit, as vLLM reports it), chosen logprobs, top-k ids and logprobs,
    finish reason."""
    steer.push = mx.array(pushes)
    try:
        return _decode(model, steer, prompt_ids, len(pushes), max_tokens, top_k, eos)
    finally:
        steer.push = None
        mx.clear_cache()


def _decode(model, steer, prompt_ids, B, max_tokens, top_k, eos):
    cache = model.make_cache()
    x = mx.array([prompt_ids] * B)
    rows = [{"ids": [], "lp": [], "top_ids": [], "top_lps": [], "finish": "length"} for _ in range(B)]
    live = list(range(B))  # original row index of each batch position
    logits = model(x, cache=cache)[:, -1]
    for step in range(max_tokens):
        lp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        top = mx.argpartition(-lp, kth=top_k - 1, axis=-1)[:, :top_k]
        top_lp = mx.take_along_axis(lp, top, axis=-1)
        order = mx.argsort(-top_lp, axis=-1)
        top, top_lp = mx.take_along_axis(top, order, axis=-1), mx.take_along_axis(top_lp, order, axis=-1)
        mx.eval(top, top_lp)
        top, top_lp = np.array(top), np.array(top_lp)
        if not np.isfinite(top_lp).all():  # fp16 overflow under a large push, or a broken model: never record it
            raise FloatingPointError(f"non-finite logprobs at step {step}; run in bfloat16 or lower the dose")
        keep = []
        for pos, r in enumerate(live):
            t = int(top[pos, 0])
            rec = rows[r]
            rec["ids"].append(t)
            rec["lp"].append(round(float(top_lp[pos, 0]), 4))
            rec["top_ids"].append(top[pos].tolist())
            rec["top_lps"].append([round(float(v), 4) for v in top_lp[pos]])
            if t in eos:
                rec["finish"] = "stop"
            else:
                keep.append(pos)
        if not keep or step == max_tokens - 1:
            break
        if len(keep) < len(live):  # drop finished rows so the batch shrinks instead of decoding dead rows
            filter_cache(cache, keep)
            steer.push = steer.push[mx.array(keep)]
            live = [live[p] for p in keep]
        nxt = mx.array([[rows[r]["ids"][-1]] for r in live])
        logits = model(nxt, cache=cache)[:, -1]
    return rows
