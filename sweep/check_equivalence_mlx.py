"""Checks for the MLX port, the check_equivalence.py of steer_sweep_mlx.py. Each run saves greedy rollouts of the same
cases (baselines plus neurons at main_rel doses); --compare lines two saves up. One model at a time, since a 16 GB Mac
can't hold two.

  push semantics: --push out adds add * W_down[:, n] at the MLP output (what the sweep does); --push act adds `add` to
                  neuron n's activation before down_proj (the reference definition in check_equivalence.py)
  precision:      the same cases on an 8-bit conversion vs the bf16 weights (--mlx-model)
  engine:         --engine hf reruns the cases in transformers on MPS with a down_proj pre-hook

    uv run python sweep/check_equivalence_mlx.py --stats stats/q35-4b_L22.npz --mlx-model ~/.cache/mlx-models/Qwen3.5-4B-q8 --out runs/eq_q8_out.json
    uv run python sweep/check_equivalence_mlx.py --stats stats/q35-4b_L22.npz --mlx-model ~/.cache/mlx-models/Qwen3.5-4B-q8 --push act --out runs/eq_q8_act.json
    uv run python sweep/check_equivalence_mlx.py --compare runs/eq_q8_out.json runs/eq_q8_act.json

The bf16 cases (--mlx-model Qwen/Qwen3.5-4B --allow-full, or --engine hf) load ~9.3 GB of weights. On a 16 GB Mac run
them alone, last, and under sweep/memguard.py.
"""
import argparse
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(pa, pb, tok):
    A, B = json.load(open(pa)), json.load(open(pb))
    print(f"A = {pa} ({A['engine']})\nB = {pb} ({B['engine']})\n"
          f"agree = shared greedy prefix; dlp = max |chosen-token logprob A - B| over the shared prefix")
    rows = []
    for a, b in zip(A["cases"], B["cases"]):
        k = common_prefix(a["ids"], b["ids"])
        m = min(len(a["ids"]), len(b["ids"]))
        dlp = max((abs(x - y) for x, y in zip(a["lp"][:k], b["lp"][:k])), default=0.0)
        label = f"{a['prompt'][:10]!r} " + ("baseline" if a["neuron"] is None else f"n{a['neuron']} {a['dose']:+g}")
        print(f"  {label:28s} agree {k:3d}/{m}  dlp {dlp:.3f}  A: {tok.decode(a['ids'][:18])!r}")
        if k < m:
            print(f"  {'':28s}                     B: {tok.decode(b['ids'][:18])!r}")
        rows.append({"label": label, "agree": k, "of": m, "dlp": round(dlp, 4)})
    full = sum(r["agree"] == r["of"] for r in rows)
    print(f"{full}/{len(rows)} cases identical to the end; median agree {np.median([r['agree'] for r in rows]):.0f}")
    return rows


def generate_batch_act(model, act_add, prompt_ids, max_tokens, eos):
    """Greedy batch decode with the push in activation space; act_add["v"] [B, I] is filtered with the live rows."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_qwen import filter_cache

    B = act_add["v"].shape[0]
    cache = model.make_cache()
    rows = [{"ids": [], "lp": []} for _ in range(B)]
    live = list(range(B))
    logits = model(mx.array([prompt_ids] * B), cache=cache)[:, -1]
    for step in range(max_tokens):
        lp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        t = mx.argmax(lp, axis=-1)
        c = mx.take_along_axis(lp, t[:, None], axis=-1)[:, 0]
        mx.eval(t, c)
        t, c = np.array(t), np.array(c)
        keep = []
        for pos, r in enumerate(live):
            rows[r]["ids"].append(int(t[pos]))
            rows[r]["lp"].append(round(float(c[pos]), 4))
            if int(t[pos]) not in eos:
                keep.append(pos)
        if not keep or step == max_tokens - 1:
            break
        if len(keep) < len(live):
            filter_cache(cache, keep)
            act_add["v"] = act_add["v"][mx.array(keep)]
            live = [live[p] for p in keep]
        logits = model(mx.array([[rows[r]["ids"][-1]] for r in live]), cache=cache)[:, -1]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--mlx-model", default=None, help="MLX weights to run (default: the 8-bit conversion)")
    ap.add_argument("--allow-full", action="store_true", help="allow unquantized weights on a Mac under 24 GB")
    ap.add_argument("--mem-gb", type=float, default=9.0, help="MLX memory limit")
    ap.add_argument("--stats")
    ap.add_argument("--engine", default="mlx", choices=["mlx", "hf"])
    ap.add_argument("--push", default="out", choices=["out", "act"])
    ap.add_argument("--neurons", default="2742,17,5000")
    ap.add_argument("--doses", default="-0.7,0.8")
    ap.add_argument("--prompts", default="Who are you?|Describe the color orange.")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="runs/eq.json")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.compare:
        compare(*a.compare, tok)
        return

    st = np.load(a.stats)
    layer, w_down = int(st["layer"]), st["w_down"].astype(np.float32)
    resid_p50 = float(st["resid_norm_q10_50_90"][1])
    down_norm = np.linalg.norm(w_down, axis=1)
    groups = []  # per prompt: (prompt, ids, [(neuron, dose, add)])
    for p in a.prompts.split("|"):
        enc = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True)
        ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
        cs = [(None, 0.0, 0.0)] + [(n, d, float(d * resid_p50 / down_norm[n]))
                                   for n in map(int, a.neurons.split(",")) for d in map(float, a.doses.split(","))]
        groups.append((p, ids, cs))

    cases = []
    if a.engine == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        sys.path.insert(0, os.path.dirname(__file__))
        from mlx_qwen import DEFAULT_MLX_MODEL, eos_ids, generate_batch, install_steering, limit_memory, load_model, text_layers

        limit_memory(a.mem_gb)
        a.mlx_model = a.mlx_model or DEFAULT_MLX_MODEL
        model = load_model(a.mlx_model, a.dtype, a.allow_full)
        eos = eos_ids(a.model, tok)
        steer = install_steering(model, layer)
        act_add = {"v": None}
        if a.push == "act":  # reference definition: push the neuron's activation and let down_proj carry it
            mlp = steer.inner

            class ActPush(nn.Module):
                def __call__(self, x):
                    act = nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
                    return mlp.down_proj(act + act_add["v"][:, None, :].astype(act.dtype))

            text_layers(model)[layer].mlp = ActPush()
        for p, ids, cs in groups:
            if a.push == "out":
                pushes = np.stack([np.zeros(w_down.shape[1], np.float32) if n is None else w_down[n] * add
                                   for n, _, add in cs])
                rows = generate_batch(model, steer, ids, pushes, a.new, 5, eos)
            else:
                v = np.zeros((len(cs), w_down.shape[0]), np.float32)
                for i, (n, _, add) in enumerate(cs):
                    if n is not None:
                        v[i, n] = add
                act_add["v"] = mx.array(v)
                rows = generate_batch_act(model, act_add, ids, a.new, eos)
            for (n, d, add), r in zip(cs, rows):
                cases.append({"prompt": p, "neuron": n, "dose": d, "add": add, "ids": r["ids"], "lp": r["lp"]})
        engine = f"mlx:{a.mlx_model}/{a.dtype}/push-{a.push}"
    else:
        import torch
        from transformers import Qwen3_5ForConditionalGeneration

        # straight onto the GPU: loading on the CPU and then .to("mps") holds two copies in the same unified memory
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="mps").eval()
        mlp = model.model.language_model.layers[layer].mlp
        push = {"n": None, "v": 0.0}

        def hook(m, inp):
            if push["n"] is None:
                return None
            h = inp[0].clone()
            h[..., push["n"]] += push["v"]
            return (h,)

        mlp.down_proj.register_forward_pre_hook(hook)
        for p, ids, cs in groups:
            for n, d, add in cs:
                push.update(n=n, v=add)
                with torch.inference_mode():
                    g = model.generate(torch.tensor([ids], device="mps"), max_new_tokens=a.new, do_sample=False,
                                       output_logits=True, return_dict_in_generate=True)
                hi = g.sequences[0, len(ids):].tolist()
                hlp = [round(float(torch.log_softmax(l[0].float(), -1)[t]), 4) for l, t in zip(g.logits, hi)]
                cases.append({"prompt": p, "neuron": n, "dose": d, "add": add, "ids": hi, "lp": hlp})
        engine = "hf/mps/bfloat16/act"
    json.dump({"engine": engine, "layer": layer, "cases": cases}, open(a.out, "w"))
    print(f"saved {len(cases)} rollouts ({engine}) -> {a.out}")


if __name__ == "__main__":
    main()
