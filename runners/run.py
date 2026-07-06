#!/usr/bin/env python3
"""Single Modal entrypoint for the production measurement harness.

Every recipe is graded under ONE centrally-enforced boundary (see harness.py):
build is one-time & untimed, prepare is per-run & charged, train is timed. Two
metrics are reported per recipe — train_loop_s (primary, lineage-comparable) and
compute_honest_s (charged prepare + train). Trials are pooled across <=24
A100-SXM4 containers, exact-allocated and topped up to hit n.

Run from the repo root (`uv run` provides modal):

  uv run modal run runners/run.py::trial --hiverge --n 200
  uv run modal run runners/run.py::trial --e1 --n 200
  uv run modal run runners/run.py::trial --target hiverge --n 16 --overrides '{"train_epochs":7.75}'

This stays one file (like bench/run.py) because Modal loads the entrypoint file
specially; the pure logic lives in harness.py / targets.py / validate.py (mounted
via add_local_python_source). Aggregation + result writing live here since this
is the client/launcher.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
DATA = "/cifar/data"
CACHE = "/cache/kernel"
GATE = 0.94
MAX_CONTAINERS = 24  # Modal silently drops results above ~24

# ---- self-contained image: CUDA + torch + baked CIFAR-10 + mounted runners ----
_BAKE = (
    "import torch, torchvision, os; "
    f"os.makedirs('{DATA}', exist_ok=True); "
    "tr=torchvision.datasets.CIFAR10('/tmp/cifar', download=True, train=True); "
    "torch.save({'images': torch.tensor(tr.data), 'labels': torch.tensor(tr.targets), "
    f"'classes': tr.classes}}, '{DATA}/train.pt'); "
    "te=torchvision.datasets.CIFAR10('/tmp/cifar', download=True, train=False); "
    "torch.save({'images': torch.tensor(te.data), 'labels': torch.tensor(te.targets), "
    f"'classes': te.classes}}, '{DATA}/test.pt')"
)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .pip_install("numpy")  # from the default index (not the PyTorch CUDA wheel index)
    .pip_install("torch==2.4.0", "torchvision==0.19.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .run_commands(f'python -c "{_BAKE}"')
    .add_local_python_source("runners")
)
app = modal.App("cifar-runners")
vol = modal.Volume.from_name("cifar-speedrun-kernel-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", cpu=8, timeout=60 * 25,
              # Many retries: each non-SXM draw exits to reroll, and SXM hosts can be
              # preempted mid-build — a container needs several shots to land + finish
              # on a stable SXM host when the pool is scarce.
              retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=1.0),
              volumes={CACHE: vol})
def _trial(name: str, overrides: dict, trial_ids: list[int],
           seed_base: int, container_id: int, plain_eval: bool) -> dict:
    """Build once on this container, then run the assigned trial ids."""
    import os

    os.environ.setdefault("CIFAR_DATA", DATA)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{CACHE}/inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{CACHE}/triton")
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

    from runners import harness, targets

    target = targets.get(name).cls()  # adapter imports its own vendored recipe in build()
    out = harness.run_trials(
        target, data_path=os.environ["CIFAR_DATA"], overrides=overrides,
        trial_ids=trial_ids, seed_base=seed_base, container_id=container_id,
        plain_eval=plain_eval,
    )
    try:
        vol.commit()
    except Exception:
        pass
    return out


# ---------------- local helpers (client side) ----------------
def _chunk(ids: list[int], containers: int) -> list[list[int]]:
    """Round-robin `ids` into at most `containers` (<=24) balanced groups."""
    containers = max(1, min(containers, MAX_CONTAINERS))
    groups: list[list[int]] = [[] for _ in range(containers)]
    for i, gid in enumerate(ids):
        groups[i % containers].append(gid)
    return [g for g in groups if g]


def _spawn_collect(name, id_groups, seed, ov, plain_eval, cid_base=0):
    calls = [_trial.spawn(name, ov, g, seed, cid_base + i, plain_eval)
             for i, g in enumerate(id_groups)]
    trials, gpus = [], []
    for c in calls:
        try:
            r = c.get(); trials.extend(r["trials"]); gpus.append(r["gpu"])
        except Exception as e:  # a lost container drops its trials, doesn't kill the run
            print(f"  (container failed: {e!r})", flush=True)
    return trials, gpus


def _ms(xs):
    return {"mean_s": round(statistics.fmean(xs), 4),
            "std_s": round(statistics.pstdev(xs), 4) if len(xs) >= 2 else 0.0,
            "min_s": round(min(xs), 4), "n": len(xs)}


def _aggregate(trials, label, ov, n_req):
    """Dual-metric summary + significance + cluster/drop-first diagnostics."""
    from collections import defaultdict
    train = [t["train_loop_s"] for t in trials]
    honest = [t["compute_honest_s"] for t in trials]
    prep = [t["prepare_s"] for t in trials]
    accs = [t["graded"] for t in trials]
    n = len(accs)

    p = None
    if n >= 2:
        if statistics.pstdev(accs) == 0:  # zero-variance special case (audit #22)
            p = 0.0 if statistics.fmean(accs) > GATE else 1.0
        else:
            try:
                import scipy.stats
                p = float(scipy.stats.ttest_1samp(accs, GATE, alternative="greater").pvalue)
            except Exception:
                p = None

    by_cont = defaultdict(list)
    for t in trials:
        by_cont[t["container_id"]].append(t)
    cont_means = [statistics.fmean([x["compute_honest_s"] for x in ts]) for ts in by_cont.values()]
    df_honest = [x["compute_honest_s"] for ts in by_cont.values() for x in ts if x["local_trial_idx"] != 0]

    bd_sum, bd_cnt = defaultdict(float), defaultdict(int)
    for t in trials:
        for k, v in (t.get("prepare_breakdown_unsynced_s") or {}).items():
            bd_sum[k] += v; bd_cnt[k] += 1

    return {
        "label": label, "overrides": ov, "n_requested": n_req, "n_returned": n,
        "complete": n_req is None or n >= n_req,
        # compute-honest is THE official measurement (charged per-run prepare + train).
        "metric": "compute_honest_s",
        "time": _ms(honest),
        "acc_mean": round(statistics.fmean(accs), 5),
        "acc_std": round(statistics.pstdev(accs), 5) if n >= 2 else 0.0,
        "acc_min": round(min(accs), 5),
        "n_pass_ge_0.94": sum(1 for a in accs if a >= GATE),
        "pass_rate": round(sum(1 for a in accs if a >= GATE) / n, 3),
        "p_value_vs_0.94_one_sided": p, "certifies": (p is not None and p < 0.05),
        "cluster": {"containers": len(by_cont),
                    "time_mean_by_container_std_s": round(statistics.pstdev(cont_means), 4) if len(cont_means) >= 2 else 0.0},
        "drop_first_trial": {"time_mean_s": round(statistics.fmean(df_honest), 4) if df_honest else None,
                             "n": len(df_honest)},
        # secondary, for transparency / decomposition only (raw values in trials.jsonl):
        "diagnostics": {
            "train_loop": _ms(train),
            "prepare": {**_ms(prep), "breakdown_mean_s": {k: round(bd_sum[k] / bd_cnt[k], 4) for k in bd_sum}},
        },
        "validate_mutations": sum(1 for t in trials if t.get("validate_mutated_weights")),
    }


def _write(stamp, label, n, config, trials, summary):
    d = REPO / "runners" / "results" / f"{stamp}_{label}_{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config, indent=2))
    with (d / "trials.jsonl").open("w") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    # Mirror a flat runner_<label>.json (the plot source) only for VALID runs
    # (complete + no eval weight mutations); bad/partial runs stay sandboxed in
    # their timestamped dir so they can't poison downstream plots (audit #8/#11/#36).
    if summary.get("valid"):
        flat = REPO / "runners" / "results" / f"runner_{label}.json"
        flat.parent.mkdir(parents=True, exist_ok=True)
        flat.write_text(json.dumps(summary, indent=2))
    return d


@app.local_entrypoint()
def trial(target: str = "", hiverge: bool = False, e1: bool = False,
          n: int = 200, containers: int = 24, seed: int = 0,
          overrides: str = "{}", plain_eval: bool = True, top_up: bool = True,
          label: str = ""):
    """Grade a target over n pooled trials; write the dual-metric summary."""
    from datetime import datetime, timezone

    from runners import targets

    if sum(bool(x) for x in (hiverge, e1, target)) > 1:
        raise SystemExit("specify exactly one target (--hiverge | --e1 | --target NAME)")
    name = "hiverge" if hiverge else "e1" if e1 else target
    if not name:
        raise SystemExit("specify a target: --hiverge | --e1 | --target NAME")
    spec = targets.get(name)
    ov = json.loads(overrides)
    if not isinstance(ov, dict):
        raise SystemExit("--overrides must be a JSON object, e.g. '{\"train_epochs\":7.75}'")
    pe = plain_eval or spec.plain_eval
    import re
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", label or name)

    trials, gpus = _spawn_collect(name, _chunk(list(range(n)), containers), seed, ov, pe)
    returned = {t["global_trial_id"] for t in trials}
    rnd = 0
    while top_up and rnd < 4:
        missing = [i for i in range(n) if i not in returned]
        if not missing:
            break
        rnd += 1
        print(f"top-up round {rnd}: {len(missing)} missing trial(s)", flush=True)
        t2, g2 = _spawn_collect(name, _chunk(missing, containers), seed, ov, pe,
                                cid_base=MAX_CONTAINERS * rnd)
        trials.extend(t2); gpus.extend(g2)
        returned |= {t["global_trial_id"] for t in t2}

    # dedupe by global_trial_id — retried/topped-up containers can return the same
    # id more than once; keep the first result per id so the pool is exactly the
    # set of distinct trial ids, never double-counted (audit finding #14).
    dedup = {}
    for t in trials:
        dedup.setdefault(t["global_trial_id"], t)
    trials = sorted(dedup.values(), key=lambda t: t["global_trial_id"])

    if not trials:
        print("no trials returned"); return

    summary = _aggregate(trials, label, ov, n_req=n)
    summary["gpus"] = sorted(set(gpus))
    summary["valid"] = bool(summary["complete"]) and summary.get("validate_mutations", 0) == 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {"label": label, "target": name, "n": n, "containers": min(containers, MAX_CONTAINERS),
              "seed": seed, "overrides": ov, "plain_eval": pe}
    d = _write(stamp, label, n, config, trials, summary)

    t, dl = summary["time"], summary["diagnostics"]["train_loop"]
    print(f"\n{label}: compute-honest {t['mean_s']:.4f}±{t['std_s']:.4f}s "
          f"(train-loop diag {dl['mean_s']:.4f}s) | "
          f"acc {summary['acc_mean']:.5f}±{summary['acc_std']:.5f} | "
          f"{summary['n_pass_ge_0.94']}/{summary['n_returned']} pass | "
          f"p={summary['p_value_vs_0.94_one_sided']} | valid={summary['valid']} | "
          f"(±=per-trial std) wrote {d}")
