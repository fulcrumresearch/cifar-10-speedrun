#!/usr/bin/env python3
"""Empirical verification of the runners/ harness — pro's first-GPU-run checklist.

This is NOT a measurement run. It exercises the correctness/fairness invariants
that static review can't prove, on real A100-SXM4 hardware, and writes one
structured report. Run from the repo root:

  uv run modal run runners/verify.py::verify
  uv run modal run runners/verify.py::verify --targets-csv hiverge   # one target

Per target it spawns TWO containers (X, Y); each builds once, records env/dataset/
static-cache fingerprints + GPU telemetry, then runs trial ids in two orders
([0,1] then [1,0]) so we can check trial-order independence within a container and
determinism across containers. Local-only checks (certification edge cases,
incomplete-run mirror guard) run on the client instantly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

# Reuse run.py's baked image + volume (same image hash -> no rebuild).
from runners.run import CACHE, DATA, image, vol

app = modal.App("cifar-runners-verify")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sig(t):
    """Cheap content signature for a (possibly huge) tensor: shape/dtype + two
    moments. Content-sensitive enough to detect cross-container differences."""
    tf = t.detach().float()
    return [list(t.shape), str(t.dtype),
            round(float(tf.sum().item()), 3), round(float((tf * tf).sum().item()), 3)]


def _e1_epoch_data_sig(state):
    """Signature of e1's graph-bound epoch_data buffers (first + last step's image
    batch). Used to prove charged prepare OVERWRITES the build-time precompute, so
    no build-time data feeds a measured trial (fairness check, pro #6/#12)."""
    try:
        ed = state.recipe._CACHE.get("epoch_data")
        if not ed:
            return None
        return {"n_steps": len(ed), "first": _sig(ed[0][0]), "last": _sig(ed[-1][0])}
    except Exception as e:
        return {"error": repr(e)}


def _e1_graph_state(state):
    """Best-effort introspection of e1's graph-bound state right after prepare
    (before replay): momentum buffers should be zero, step count reset."""
    try:
        C = state.recipe._CACHE
        opt2, loader = C.get("opt2"), C.get("train_loader")
        mb_absmax = 0.0
        nbuf = 0
        for st in opt2.state.values():
            buf = st.get("momentum_buffer")
            if buf is not None:
                nbuf += 1
                mb_absmax = max(mb_absmax, float(buf.abs().max().item()))
        return {"opt2_step_count": getattr(opt2, "step_count", None),
                "loader_epoch": getattr(loader, "epoch", None),
                "n_momentum_buffers": nbuf, "momentum_absmax_before_train": mb_absmax}
    except Exception as e:
        return {"error": repr(e)}


@app.function(image=image, gpu="A100-80GB", cpu=8, timeout=60 * 40,
              retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=1.0),
              volumes={CACHE: vol})
def _probe(target_name: str, orders: list[list[int]], seed_base: int) -> dict:
    import os
    import sys
    import types

    os.environ.setdefault("CIFAR_DATA", DATA)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{CACHE}/inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{CACHE}/triton")
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

    import torch
    import torchvision

    from runners import harness, targets, validate

    out: dict = {"target": target_name}

    # --- check 1: import/env paths ---
    import runners
    out["env"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "runners_file": runners.__file__, "cifar_data_env": os.environ.get("CIFAR_DATA"),
        "device": torch.cuda.get_device_name(0),
    }

    # --- check 2: dataset invariants ---
    try:
        test = torch.load(f"{DATA}/test.pt", map_location="cpu", weights_only=True)
        labels = test["labels"]
        out["dataset"] = {
            "test_sha16": _file_sha256(f"{DATA}/test.pt"),
            "train_sha16": _file_sha256(f"{DATA}/train.pt"),
            "test_images_shape": list(test["images"].shape),
            "test_label_counts": torch.bincount(labels).tolist(),
            "classes": list(test.get("classes", [])),
        }
    except Exception as e:
        out["dataset"] = {"error": repr(e)}

    # --- GPU guard + build (untimed) with telemetry around it (check 8) ---
    out["gpu_guard"] = harness.sxm_single_gpu_or_exit()
    target = targets.get(target_name).cls()
    harness.seed_everything(harness.BUILD_SEED)
    telem_pre = harness.gpu_telemetry()
    with harness.timed() as bt:
        state = target.build(harness.BuildContext(DATA, {}))
    telem_post = harness.gpu_telemetry()
    out["build"] = {"build_s": round(bt["s"], 3), "ok": True,
                    "telem_before": telem_pre, "telem_after": telem_post}
    if target_name == "e1":
        out["env"]["e1_recipe_file"] = state.recipe.__file__
        out["env"]["e1_DATA_PATH"] = state.recipe.DATA_PATH
        out["epoch_data_after_build"] = _e1_epoch_data_sig(state)

    # --- check 3: static proc_images cache signatures ---
    try:
        loader = (state.obj["loader"] if target_name == "hiverge"
                  else state.recipe._CACHE["train_loader"])
        out["static_cache_sig"] = {k: _sig(v) for k, v in loader.proc_images.items()}
    except Exception as e:
        out["static_cache_sig"] = {"error": repr(e)}

    # --- trials in each order (checks 4, 6, 7, 10) ---
    passes = []
    for order in orders:
        rows = []
        for li, tid in enumerate(order):
            seed = seed_base + tid
            harness.seed_everything(seed)
            sw = harness.Stopwatch()
            rc = harness.RunContext(tid, seed, DATA, sw)
            with harness.timed() as pt:
                run = target.prepare(state, rc)
            run.prepared_token = tid
            gstate = _e1_graph_state(state) if target_name == "e1" else {}
            edsig = _e1_epoch_data_sig(state) if target_name == "e1" else None
            with harness.timed() as tt:
                target.train(state, run, rc)
            model = run.model
            model.eval_forward = types.MethodType(lambda self, x: self.forward(x), model)
            model_fp = harness.param_fingerprint(model)        # post-train, pre-eval
            with harness.timed() as et:
                res = validate.run_validation(model, DATA)
            fp_after_eval = harness.param_fingerprint(model)
            rows.append({
                "tid": tid, "local_idx": li,
                "prepare_s": round(pt["s"], 5), "train_loop_s": round(tt["s"], 5),
                "compute_honest_s": round(pt["s"] + tt["s"], 5),
                "graded": round(res["graded"], 6), "model_fp": model_fp,
                "eval_s": round(et["s"], 4),
                "eval_mutated_weights": model_fp != fp_after_eval,
                "graph_state": gstate, "epoch_data_after_prepare": edsig,
            })
        passes.append({"order": order, "rows": rows})
    out["passes"] = passes
    try:
        vol.commit()
    except Exception:
        pass
    # Return a JSON string, not the raw dict: the dict can contain torch-typed
    # values (e.g. torch.Size/dtype) that the torch-free client can't unpickle
    # (Modal DeserializationError). default=str stringifies anything exotic.
    return json.dumps(out, default=str)


# ---------------- local-only checks (run on the client) ----------------
def _local_checks() -> dict:
    """check 12 (certification edge cases) + check 11 (incomplete-run mirror guard),
    exercised against the real run.py code paths without a GPU."""
    import tempfile

    from runners import run as R

    # check 12: certification on synthetic trial sets
    def trials(accs):
        return [{"train_loop_s": 1.8, "prepare_s": 0.03, "compute_honest_s": 1.83,
                 "graded": a, "container_id": i % 2, "local_trial_idx": i // 2,
                 "validate_mutated_weights": False, "prepare_breakdown_unsynced_s": {}}
                for i, a in enumerate(accs)]
    cert = {}
    for name, accs in [("all_pass_0.95", [0.95] * 8), ("all_fail_0.93", [0.93] * 8),
                       ("zerovar_pass", [0.9500] * 8), ("zerovar_fail", [0.9300] * 8),
                       ("mixed", [0.939, 0.941, 0.942, 0.940, 0.943, 0.938, 0.941, 0.944])]:
        s = R._aggregate(trials(accs), name, {}, n_req=len(accs))
        cert[name] = {"p": s["p_value_vs_0.94_one_sided"], "certifies": s["certifies"],
                      "acc_mean": s["acc_mean"]}

    # check 11: incomplete / mutated runs must NOT write the runners/ flat mirror
    mirror = {}
    orig_repo = R.REPO
    try:
        with tempfile.TemporaryDirectory() as td:
            R.REPO = Path(td)
            for tag, summ in [("incomplete", {"complete": False, "valid": False}),
                              ("mutated", {"complete": True, "valid": False}),
                              ("valid", {"complete": True, "valid": True})]:
                R._write("S", f"probe_{tag}", 2, {"label": f"probe_{tag}"}, [], summ)
                flat = Path(td) / "runners" / "results" / f"runner_probe_{tag}.json"
                mirror[tag] = {"mirror_written": flat.exists(),
                               "expected": summ["valid"]}
    finally:
        R.REPO = orig_repo
    return {"certification_edge_cases": cert, "flat_mirror_guard": mirror}


def _eq(a, b):
    return "MATCH" if a == b else "DIFFER"


def _overwritten(build_sig, prep_sig):
    # We WANT build != prepare: charged prepare must overwrite the build-time data.
    return "overwritten (good)" if build_sig != prep_sig else "REUSED BUILD DATA (BAD)"


def _analyze(name: str, X: dict, Y: dict) -> dict:
    a = {"build_s": {"X": X["build"]["build_s"], "Y": Y["build"]["build_s"]},
         "telemetry": {"X_build": [X["build"]["telem_before"], X["build"]["telem_after"]]},
         "dataset_sha_match_across_containers":
             _eq(X.get("dataset", {}).get("test_sha16"), Y.get("dataset", {}).get("test_sha16")),
         "static_cache_match_across_containers":
             _eq(X.get("static_cache_sig"), Y.get("static_cache_sig")),
         "static_cache_sig": X.get("static_cache_sig"),
         "env": X.get("env"), "gpu": [X.get("gpu_guard"), Y.get("gpu_guard")]}

    # trial-order independence WITHIN container X: pass0 vs pass1, per global id
    p0 = {r["tid"]: r for r in X["passes"][0]["rows"]}
    p1 = {r["tid"]: r for r in X["passes"][1]["rows"]}
    order_ind = {}
    for tid in sorted(p0):
        order_ind[tid] = {
            "model_fp": _eq(p0[tid]["model_fp"], p1[tid]["model_fp"]),
            "acc": _eq(p0[tid]["graded"], p1[tid]["graded"]),
            "acc_p0": p0[tid]["graded"], "acc_p1": p1[tid]["graded"],
        }
    a["order_independence_within_X"] = order_ind

    # determinism ACROSS containers: X.pass0 vs Y.pass0 per id
    y0 = {r["tid"]: r for r in Y["passes"][0]["rows"]}
    a["determinism_across_containers"] = {
        tid: {"model_fp": _eq(p0[tid]["model_fp"], y0[tid]["model_fp"]),
              "acc": _eq(p0[tid]["graded"], y0[tid]["graded"])}
        for tid in sorted(p0) if tid in y0}

    # timing + accuracy summary + first-trial-vs-rest (thermal/build-heat)
    rows = [r for P in X["passes"] for r in P["rows"]] + [r for P in Y["passes"] for r in P["rows"]]
    tl = [r["train_loop_s"] for r in rows]
    ch = [r["compute_honest_s"] for r in rows]
    accs = [r["graded"] for r in rows]
    first = [r["train_loop_s"] for r in rows if r["local_idx"] == 0]
    rest = [r["train_loop_s"] for r in rows if r["local_idx"] != 0]
    import statistics
    a["timing"] = {
        "train_loop_mean_s": round(statistics.fmean(tl), 4),
        "compute_honest_mean_s": round(statistics.fmean(ch), 4),
        "acc_mean": round(statistics.fmean(accs), 5),
        "acc_range": [round(min(accs), 5), round(max(accs), 5)],
        "first_trial_train_mean_s": round(statistics.fmean(first), 4) if first else None,
        "rest_trial_train_mean_s": round(statistics.fmean(rest), 4) if rest else None,
        "eval_s_mean": round(statistics.fmean([r["eval_s"] for r in rows]), 4),
        "any_eval_mutated_weights": any(r["eval_mutated_weights"] for r in rows),
    }
    if name == "e1":
        a["e1_graph_state_sample"] = X["passes"][0]["rows"][0].get("graph_state")
        # Fairness: build-time precompute must be OVERWRITTEN by charged prepare
        # (so no build-time data feeds a measured trial), and the per-seed data
        # must be reproducible regardless of trial order.
        bsig = X.get("epoch_data_after_build")
        p0d = {r["tid"]: r.get("epoch_data_after_prepare") for r in X["passes"][0]["rows"]}
        p1d = {r["tid"]: r.get("epoch_data_after_prepare") for r in X["passes"][1]["rows"]}
        a["epoch_data_build_vs_prepare"] = {tid: _overwritten(bsig, p0d[tid]) for tid in sorted(p0d)}
        a["epoch_data_order_independent"] = {tid: _eq(p0d[tid], p1d[tid]) for tid in sorted(p0d)}
    return a


@app.local_entrypoint()
def verify(targets_csv: str = "hiverge,e1", seed: int = 0):
    from datetime import datetime, timezone

    names = [n.strip() for n in targets_csv.split(",") if n.strip()]
    orders = [[0, 1], [1, 0]]

    print("=== local checks (certification + mirror guard) ===", flush=True)
    local = _local_checks()
    print(json.dumps(local, indent=2), flush=True)

    print(f"\n=== spawning 2 probes x {len(names)} target(s) on A100-SXM4 ===", flush=True)
    handles = {nm: [_probe.spawn(nm, orders, seed), _probe.spawn(nm, orders, seed)]
               for nm in names}
    report = {"local": local, "targets": {}}
    for nm, (hX, hY) in handles.items():
        try:
            X, Y = json.loads(hX.get()), json.loads(hY.get())
            report["targets"][nm] = _analyze(nm, X, Y)
            report["targets"][nm]["_raw"] = {"X": X, "Y": Y}
        except Exception as e:
            report["targets"][nm] = {"error": repr(e)}
        print(f"\n=== {nm} ===\n" +
              json.dumps({k: v for k, v in report["targets"][nm].items() if k != "_raw"}, indent=2),
              flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).resolve().parents[1] / "runners" / "results" / f"verify_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
