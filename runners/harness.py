"""The measurement boundary + engine — the single source of truth for how every
recipe is timed.

Why this file exists
--------------------
In the old `bench/` harness each recipe owned its own `setup()` and the runner
timed only `train()`. Whatever a recipe chose to do in `setup()` was untimed, so
work drifted across the timed boundary inconsistently: e1 pre-materialises its
entire augmented batch stream and inits whitening in untimed `setup()`, while
hiverge/e2/e4/e5 do that same work *inside* the timed loop. The comparison was
silently apples-to-oranges.

The fix is to make the boundary a property of the RUNNER, not the recipe. A
recipe ("target") implements three phases with fixed, enforced semantics; this
engine decides what is timed, what is charged, and what is free:

  build(ctx, recipe)  -> BuildState   ONCE per worker process. Untimed, amortised
                                      like compile: allocate model/optimisers/
                                      loaders, run compile warmups, capture CUDA
                                      graphs. MUST NOT do per-run work that
                                      defines a measured trial.
  prepare(state, run) -> RunState     PER-RUN. The WHOLE call is CHARGED to the
                                      compute-honest metric (never the train-loop
                                      metric): per-run whitening init, aug
                                      materialisation, model/optimiser reset.
  train(state, run, ctx)              The TIMED training loop (primary,
                                      lineage-comparable metric).

Dual metric (both reported):
  train_loop_s     = wall-clock of train()              (primary)
  compute_honest_s = charged prepare() + train_loop_s   (honest total)
One-time build is free for BOTH (paid once, like everyone's compile).

This is apples-to-apples: e1 pays augmentation in CHARGED prepare; eager recipes
pay it in TIMED train; nobody gets per-run augmentation off the clock. Only
genuinely one-time build/compile/graph-capture is ever free.

This is a measurement harness for TRUSTED, curated recipes, not a security
sandbox — a malicious file can always smuggle work through globals. The goal is
to make boundary mistakes hard to write and easy to detect.

The file has four sections: (1) contract types, (2) timing, (3) guards,
(4) the orchestrator.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Accuracy gate: TTA top-1 on the held-out graded split must clear this.
GATE = 0.94
# Fixed seed for the one-time build, so the static (build-time) augmentation cache
# — including the loader's one-time random pre-flip — is identical across
# containers, not a per-host roll (audit finding #11).
BUILD_SEED = 1234


# ======================================================================
# 1. Contract types
# ======================================================================
@dataclass
class BuildContext:
    """Inputs to the one-time, untimed build phase."""
    data_path: str                 # dir holding train.pt / test.pt
    overrides: dict                # hyp overrides patched onto the recipe
    device: str = "cuda"


@dataclass
class RunContext:
    """Per-trial context handed to prepare() and train().

    `charge` labels charged-prepare sub-steps for the report breakdown
    (`with run.charge.section("whiten"): ...`). The authoritative charged TOTAL
    is the whole prepare() call, timed by the orchestrator, so a target can't
    dodge a charge by leaving work unlabelled — the breakdown is diagnostic only.
    """
    trial_id: int
    seed: int
    data_path: str
    charge: "Stopwatch"


@dataclass
class BuildState:
    """Opaque per-worker build artefacts (model, optimisers, loader, graphs)."""
    obj: Any
    recipe: Any = None
    meta: dict = field(default_factory=dict)


@dataclass
class RunState:
    """Opaque per-trial state from prepare(), consumed by train(). `model` is
    what the orchestrator evaluates. The orchestrator only ever calls train() on
    a RunState returned by prepare(), so build() cannot hand back a train-ready
    run."""
    model: Any
    obj: Any = None
    prepared_token: int = 0


@runtime_checkable
class Target(Protocol):
    name: str
    recipe: str        # path to the vendored recipe the adapter drives (provenance)
    plain_eval: bool   # force eval_forward -> forward (eval-TTA neutralised)

    def build(self, ctx: BuildContext) -> BuildState:
        """One-time, untimed. Import the vendored recipe, allocate + compile +
        capture. NOT train-ready."""
        ...

    def prepare(self, state: BuildState, run: RunContext) -> RunState:
        """Per-run, charged. Reset to untrained, init whitening, materialise the
        per-run augmented stream."""
        ...

    def train(self, state: BuildState, run: RunState, ctx: RunContext) -> None:
        """The timed training loop. Pure compute: no reset, whitening, aug
        materialisation, final eval, or sleep."""
        ...


# ======================================================================
# 2. Timing  (perf_counter + cuda.synchronize is the official metric; CUDA
#    events are diagnostic only. Graph replay is async, so every region syncs
#    AFTER the work before reading the clock.)
# ======================================================================
def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed():
    """Time a region in seconds, CUDA-synced both ends. Yields {'s': elapsed}."""
    _sync()
    box = {"s": None}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        _sync()
        box["s"] = time.perf_counter() - t0


class Stopwatch:
    """Accumulates labelled sub-steps for the prepare breakdown (report only).

    Deliberately does NOT cuda.synchronize per section: the authoritative charged
    number is the whole prepare() call, synced once by the orchestrator. Syncing
    here would serialise async work and inflate the official prepare_s (audit
    finding #3). The breakdown is therefore approximate (CPU/launch time per
    section), diagnostic only.
    """

    def __init__(self):
        self.total = 0.0
        self.breakdown: dict[str, float] = {}

    @contextmanager
    def section(self, label: str = "_"):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.total += dt
            self.breakdown[label] = self.breakdown.get(label, 0.0) + dt


# ======================================================================
# 3. Guards  (cheap, high-value enforcement; not a security sandbox)
# ======================================================================
def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def sxm_single_gpu_or_exit() -> str:
    """Pin the timing population to one full A100-SXM4-80GB. Reject PCIe, MIG,
    multi-GPU, or wrong-memory hosts. Exit (don't raise) so Modal's retry rerolls
    onto a fresh host (audit finding #16)."""
    import torch
    name = gpu_name()
    if "SXM" not in name or "A100" not in name:
        print(f"[guard] not an A100-SXM host ({name!r}); exiting to reroll", flush=True)
        os._exit(13)
    if torch.cuda.device_count() != 1:
        print(f"[guard] expected 1 GPU, saw {torch.cuda.device_count()}; exiting", flush=True)
        os._exit(13)
    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / 1e9
    if "MIG" in props.name or mem_gb < 79:
        print(f"[guard] unexpected device {props.name!r} mem={mem_gb:.0f}GB; exiting", flush=True)
        os._exit(13)
    return name


def gpu_telemetry() -> dict:
    """Best-effort GPU temperature / power / SM clock, to diagnose thermal drift
    and build-heat bias across trials (audit findings #5/#8/#17). Never raises."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        keys = ("temp_c", "power_w", "sm_mhz")
        return {k: (float(v) if v.replace(".", "", 1).isdigit() else v)
                for k, v in zip(keys, parts)}
    except Exception:
        return {}


def seed_everything(seed: int):
    """Reset CPU+CUDA RNG before each trial's prepare so aug draws and init are
    reproducible per trial and independent of trial order."""
    import random

    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def no_sleep(threshold_s: float = 0.05, enabled: bool = True):
    """Trip on any wall-clock sleep over `threshold_s` inside build/prepare/train
    — the old e1 time.sleep(60) thermal-cooldown boundary exploit. (torch.cuda._sleep
    spins the GPU and would only hurt a cheater, so we leave it.)"""
    if not enabled:
        yield
        return
    real = time.sleep

    def guarded(secs, *a, **k):
        if secs and secs > threshold_s:
            raise RuntimeError(f"time.sleep({secs}) in a measured phase is a "
                               f"timing-boundary exploit; disallowed in official runs.")
        return real(secs, *a, **k)

    time.sleep = guarded
    try:
        yield
    finally:
        time.sleep = real


def param_fingerprint(model) -> str:
    """Hash of ALL params + buffers (incl. integer buffers like BN counters), to
    assert eval/validate didn't mutate the model — weights must come back
    bit-identical (audit finding #16)."""
    import torch
    h = hashlib.blake2b(digest_size=16)
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            h.update(str((tuple(t.shape), t.dtype)).encode())
            h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


# ======================================================================
# 4. Orchestrator  (runs IN the container; owns the boundary)
# ======================================================================
def run_trials(target, *, data_path: str, overrides: dict,
               trial_ids: list[int], seed_base: int, container_id: int,
               plain_eval: bool = True, sleep_guard: bool = True) -> dict:
    """Build once, then run the assigned trials. Returns {gpu, build_s, trials}.
    The target imports its own vendored recipe in build()."""
    import types

    from . import validate

    gpu = sxm_single_gpu_or_exit()

    # ---- one-time build (untimed; build_s diagnostic only). Seeded with a fixed
    #      BUILD_SEED so the static build-time augmentation cache is identical
    #      across containers. ----
    seed_everything(BUILD_SEED)
    with no_sleep(enabled=sleep_guard):
        with timed() as bt:
            state = target.build(BuildContext(data_path, overrides))
    build_s = round(bt["s"], 3)
    print(f"[build] {target.name} on {gpu} in {build_s}s", flush=True)

    records = []
    for local_idx, tid in enumerate(trial_ids):
        seed = seed_base + tid
        seed_everything(seed)
        sw = Stopwatch()
        runctx = RunContext(tid, seed, data_path, sw)

        # ---- charged per-run prepare (whole call timed -> compute_honest) ----
        with no_sleep(enabled=sleep_guard):
            with timed() as pt:
                run = target.prepare(state, runctx)
        prepare_s = pt["s"]
        run.prepared_token = tid

        # ---- timed train loop (primary metric; timed() syncs after replay) ----
        with no_sleep(enabled=sleep_guard):
            with timed() as tt:
                target.train(state, run, runctx)
        train_loop_s = tt["s"]

        # ---- telemetry snapshot right after the timed region (thermal diag) ----
        telem = gpu_telemetry()

        # ---- untimed eval, eval-TTA neutralised, weight-mutation checked ----
        model = run.model
        if plain_eval or getattr(target, "plain_eval", True):
            model.eval_forward = types.MethodType(lambda self, x: self.forward(x), model)
        with timed() as ft0:
            fp_before = param_fingerprint(model)
        with timed() as et:
            res = validate.run_validation(model, data_path)
        with timed() as ft1:
            fp_after = param_fingerprint(model)

        records.append({
            "global_trial_id": tid, "container_id": container_id,
            "local_trial_idx": local_idx, "seed": seed, "gpu": gpu,
            "train_loop_s": round(train_loop_s, 6),
            "prepare_s": round(prepare_s, 6),
            "compute_honest_s": round(prepare_s + train_loop_s, 6),
            # diagnostic only + not CUDA-synced per section, so approximate:
            "prepare_breakdown_unsynced_s": {k: round(v, 6) for k, v in sw.breakdown.items()},
            "graded": round(res["graded"], 6), "feedback": round(res["feedback"], 6),
            "validate_mutated_weights": fp_before != fp_after,
            "eval_s": round(et["s"], 4), "fingerprint_s": round(ft0["s"] + ft1["s"], 4),
            "telemetry": telem,
            "build_s": build_s if local_idx == 0 else None,
        })
        print(f"[trial {local_idx+1}/{len(trial_ids)} id={tid}] "
              f"train={train_loop_s:.4f}s honest={records[-1]['compute_honest_s']:.4f}s "
              f"acc={res['graded']:.4f}", flush=True)

    return {"gpu": gpu, "build_s": build_s, "trials": records}
