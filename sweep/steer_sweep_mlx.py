"""steer_sweep.py for Apple Silicon: the same jobs, controls, resume and record schema, run through MLX instead of the
vLLM steering fork. Activation captures are not supported (capture_id is still written).

Built to survive a 16 GB Mac: an 8-bit model by default, an MLX memory cap, results written after every batch, a
torn last line (from a kill mid-write) trimmed on resume, and a resume refused if the run's settings changed. Run
long sweeps under sweep/memguard.py, and keep test runs out of runs/ (the explorer shows every run there).

Steering neuron n adds `add * W_down[:, n]` to layer L's MLP output at every position, prompt included, which is what
the vLLM fork does at its `mlp_out` hook. Rows of a batch share a prompt and differ only in their push.

    uv run python sweep/memguard.py --max-gb 10 -- uv run python sweep/steer_sweep_mlx.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons random:384 --random-controls 64 --run q35-4b-L22-mlx

Rerunning the same command resumes: rollouts already in the jsonl are skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from mlx_qwen import DEFAULT_MLX_MODEL, eos_ids, generate_batch, install_steering, limit_memory, load_model  # noqa: E402

# copied from steer_sweep.py, which imports vllm at the top
PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
LADDER = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DESIGNS = {
    "main": [(p, d) for p in ("who", "snake", "orange") for d in (-2.0, 2.0, 10.0)] + [("free", 10.0)],
    "calib": [(p, s * d) for p in ("who", "orange", "free") for d in LADDER for s in (-1.0, 1.0)],
    "main_rel": [(p, d) for p in ("who", "snake", "orange") for d in (-0.7, 0.5, 0.8)] + [("free", 3.0)],
}


# settings that change what a rollout is; a resume must keep them (meta.json written before these existed lacks some)
RESUME_KEYS = ("model", "layer", "stats", "design", "dose_unit", "max_tokens", "top_k", "engine", "prompt_ids",
               "resid_norm_p50", "seed", "random_controls")


def load_done(path, key):
    """Keys of the rollouts already in the jsonl. A kill mid-write leaves a torn last line; it is cut off here, since
    the explorer reads the file line by line too."""
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as f:
        data = f.read()
    keep = data.rfind(b"\n") + 1
    if keep < len(data):
        print(f"trimming a torn last line ({len(data) - keep} bytes) from {path}", flush=True)
        with open(path, "r+b") as f:
            f.truncate(keep)
    return {key(json.loads(line)) for line in data[:keep].splitlines() if line.strip()}


def parse_neurons(spec, n, seed):
    if spec == "all":
        return list(range(n))
    if spec.startswith("random:"):
        return sorted(np.random.default_rng(seed + 1).choice(n, int(spec.split(":")[1]), replace=False).tolist())
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--stats", required=True, help="npz from neuron_stats(_mlx).py for the layer to steer")
    ap.add_argument("--neurons", default="all", help='"all", "random:128", "2742", "0-99", or "1,5,9-12"')
    ap.add_argument("--design", default="main", choices=sorted(DESIGNS))
    ap.add_argument("--dose-unit", default="absmax", choices=["absmax", "rel"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--random-controls", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32, help="rows decoded together (all on one prompt)")
    ap.add_argument("--chunk", type=int, default=512, help="jobs grouped by prompt together; results are written per batch")
    ap.add_argument("--mlx-model", default=DEFAULT_MLX_MODEL, help="MLX weights to run (default: the 8-bit conversion)")
    ap.add_argument("--allow-full", action="store_true", help="allow unquantized weights on a Mac under 24 GB")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--mem-gb", type=float, default=9.0, help="MLX memory limit")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many jobs (benchmarks)")
    a = ap.parse_args()
    limit_memory(a.mem_gb)

    st = np.load(a.stats)
    layer, abs_max, w_down = int(st["layer"]), st["abs_max"], st["w_down"]
    neurons = parse_neurons(a.neurons, len(abs_max), a.seed)
    design = DESIGNS[a.design]
    resid_p50 = float(st["resid_norm_q10_50_90"][1])
    down_norm = np.linalg.norm(w_down.astype(np.float32), axis=1)
    text_cfg = AutoConfig.from_pretrained(a.model).get_text_config()
    assert not getattr(text_cfg, "layer_scale", False)

    tok = AutoTokenizer.from_pretrained(a.model)
    prompt_ids = {
        k: tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=True)
        for k, v in PROMPTS.items()
    }
    prompt_ids = {k: list(v["input_ids"] if hasattr(v, "keys") else v) for k, v in prompt_ids.items()}

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{a.run}.jsonl")
    meta = {"model": a.model, "layer": layer, "stats": os.path.abspath(a.stats), "prompts": PROMPTS,
            "design_name": a.design, "design": design, "resid_norm_p50": resid_p50, "dose_unit": a.dose_unit,
            "hook": "mlp_out", "max_tokens": a.max_tokens, "top_k": a.top_k, "capture_layers": [],
            "capture_hook": "mlp_in", "capture_positions": "last_prompt", "capture_root": None,
            "capture_tag": a.run, "prompt_ids": prompt_ids, "engine": f"mlx:{a.mlx_model}/{a.dtype}",
            "seed": a.seed, "random_controls": a.random_controls, "batch": a.batch}  # batch: rows decode together, which shifts low-margin tokens
    meta = json.loads(json.dumps(meta))  # tuples -> lists, as they read back
    meta_path = os.path.join(a.out_dir, f"{a.run}.meta.json")
    if os.path.exists(out_path) and os.path.exists(meta_path):
        old = json.load(open(meta_path))
        changed = sorted(k for k in set(meta) | set(old) if k in RESUME_KEYS and k in old and old[k] != meta.get(k))
        if changed:
            raise SystemExit(f"{out_path} was made with different {', '.join(changed)}; resuming would mix two "
                             f"setups in one run. Use a new --run name.")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    rng = np.random.default_rng(a.seed)
    ctrl_ref = rng.integers(len(abs_max), size=a.random_controls)
    ctrl_vecs = rng.standard_normal((a.random_controls, w_down.shape[1])).astype(np.float32)
    ctrl_vecs *= (np.linalg.norm(w_down[ctrl_ref].astype(np.float32), axis=1)
                  / np.linalg.norm(ctrl_vecs, axis=1))[:, None]

    units = [("neuron", n) for n in neurons]
    for k in range(a.random_controls):
        units.insert(round((k + 1) * len(units) / (a.random_controls + 1)), ("random", k))

    key = lambda j: (j["kind"], j["neuron"], j["prompt"], j["dose"])
    done = load_done(out_path, key)

    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in PROMPTS]
    jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": d} for kind, n in units for p, d in design]
    jobs = [j for j in jobs if key(j) not in done]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"L{layer}: {len(neurons)} neurons + {a.random_controls} random controls, {len(jobs)} rollouts to run "
          f"({len(done)} already done)", flush=True)
    if not jobs:
        return

    def prepare(job):
        kind, n, p, d = job["kind"], job["neuron"], job["prompt"], job["dose"]
        ref = n if kind == "neuron" else int(ctrl_ref[n]) if kind == "random" else None
        job["ref_neuron"] = ref
        if ref is None:
            job["add"] = 0.0
        elif a.dose_unit == "rel":
            job["add"] = float(d * resid_p50 / down_norm[ref])
        else:
            job["add"] = float(d * abs_max[ref])
        job["rel_push"] = abs(job["add"]) * float(down_norm[ref]) / resid_p50 if ref is not None else 0.0
        job["capture_id"] = f"L{layer}_{kind}{n}_{p}_{d:+g}".replace("+", "p").replace("-", "m")
        if ref is None:
            return np.zeros(w_down.shape[1], np.float32)
        direction = w_down[n] if kind == "neuron" else ctrl_vecs[n]
        return direction.astype(np.float32) * job["add"]

    model = load_model(a.mlx_model, a.dtype, a.allow_full)
    steer = install_steering(model, layer)
    eos = eos_ids(a.model, tok)

    t_start, tok_total = time.time(), 0
    with open(out_path, "a") as f:
        for c in range(0, len(jobs), a.chunk):
            chunk = jobs[c : c + a.chunk]
            t0 = time.time()
            pushes = [prepare(j) for j in chunk]
            by_prompt = {}
            for i, j in enumerate(chunk):
                by_prompt.setdefault(j["prompt"], []).append(i)
            n_tok = 0
            for p, idx in by_prompt.items():
                for b in range(0, len(idx), a.batch):
                    sel = idx[b : b + a.batch]
                    rows = generate_batch(model, steer, prompt_ids[p], np.stack([pushes[i] for i in sel]),
                                          a.max_tokens, a.top_k, eos)
                    lines = []
                    for i, r in zip(sel, rows):
                        job = chunk[i]
                        lines.append(json.dumps({
                            "layer": layer, "kind": job["kind"], "neuron": job["neuron"],
                            "ref_neuron": job["ref_neuron"], "prompt": job["prompt"], "dose": job["dose"],
                            "add": job["add"], "rel_push": job["rel_push"],
                            "text": tok.decode(r["ids"], skip_special_tokens=True), "finish": r["finish"],
                            "n_gen": len(r["ids"]), "token_ids": r["ids"], "logprob": r["lp"],
                            "top_ids": r["top_ids"], "top_lps": r["top_lps"], "capture_id": job["capture_id"],
                        }) + "\n")
                        n_tok += len(r["ids"])
                    f.write("".join(lines))  # one write per batch, so a kill tears at most the last line
                    f.flush()
            dt = time.time() - t0
            tok_total += n_tok
            print(f"{min(c + a.chunk, len(jobs))}/{len(jobs)} rollouts, {n_tok / dt:,.0f} gen tok/s, "
                  f"{len(chunk) / dt:.2f} rollouts/s, {n_tok / len(chunk):.0f} tok/rollout", flush=True)
    dt = time.time() - t_start
    print(f"done: {len(jobs)} rollouts, {tok_total:,} tokens in {dt:.0f}s = {tok_total / dt:,.0f} gen tok/s, "
          f"{len(jobs) / dt:.2f} rollouts/s", flush=True)


if __name__ == "__main__":
    main()
