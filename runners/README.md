# `runners/` — the production measurement harness

One harness that produces every timing + accuracy number in the root README, with a
**single, centrally-enforced setup-vs-train boundary** so no recipe can game
*where* its work lives on the clock. Run any recipe the same way:

```bash
uv run modal run runners/run.py::trial --hiverge --n 200
uv run modal run runners/run.py::trial --e1 --n 200
```

It is self-sustaining: the recipe code it runs is vendored into
`runners/recipes/` (byte-exact copies), so the harness never reaches into other
dirs at runtime. The adapters only relocate *where on the clock* each piece of
that code runs.

## Why this exists (the boundary problem)

This project's earlier harness timed only each recipe's `train()` and let
`setup()` be whatever the recipe file chose. Because every recipe owned its own
`setup()`, there was no central policy on what is untimed:

- **base/cifar10_speedrun.py** (genuine hiverge, 7.65 ep) times `init_whiten +
  train loop + TTA-eval` with CUDA events.
- **e1** moves whitening, the *entire* augmented+shuffled batch stream
  (`precompute_epochs`, ~31 ms), and eval into untimed `setup()`; its timed loop
  is just a CUDA mega-graph replay.
- **hiverge / e2 / e4 / e5** augment *inside* the timed loop; e3 uses per-step
  graphs; only e1 pre-materialises.

So the comparison silently drifted apples-to-oranges, and there was **no hiverge
at the correct 7.65-epoch schedule** in the timed-loop contract at all (the
earlier harness's baseline was the padded 8.5-epoch recipe, ~2.17 s).

## The contract (enforced by `harness.py`, not by recipes)

A recipe ("target") implements three phases; the **runner** decides what is timed:

| phase | when | timed? | what belongs here |
| :-- | :-- | :-- | :-- |
| `build(ctx, recipe)` | once per worker process | **free** (untimed, like compile) | allocate model/optimisers/loaders, compile warmups, CUDA-graph capture, static deterministic preprocessing. **Must not** leave a train-ready run or do per-run work that defines a measured trial. |
| `prepare(state, run)` | every trial | **charged** | per-run whitening init, augmentation materialisation, model/optimiser reset to untrained. The *whole call* is timed by the runner. |
| `train(state, run, ctx)` | every trial | **timed** | the training loop. Pure compute: no reset/whiten/aug-materialisation/eval/sleep. |

Final eval is untimed and **eval-TTA-neutralised** (every `eval_forward` forced to
plain `forward`, so only the grader's standard 6-view TTA counts).

### The metric: compute-honest (the single official number)

```
compute_honest_s = charged prepare() + wall-clock of train()   # THE official metric
```

`prepare` (per-run reset + whitening + augmentation-materialisation) and `train`
are both counted; one-time `build` (compile + CUDA-graph capture) is free (paid
once, like everyone's compile). The bare `train_loop_s` is still recorded per
trial as a **diagnostic** (it is the lineage-comparable airbench/hiverge "loop
only" number, useful for decomposition) but it is **not** the reported metric:
train-loop alone is gameable by relocating per-run work out of the timed loop —
exactly what e1 does with augmentation — so compute-honest is the single number we
certify and report.

### Why this is fair (apples-to-apples)

e1 pays its augmentation in **charged `prepare`**; eager recipes pay theirs
**inside timed `train`**. Nobody gets per-run augmentation off the clock; the only
thing ever free is genuinely one-time build/compile/graph-capture.

`compute_honest_s` charges the **whole** per-run `prepare` (reset + whitening +
aug materialisation) for *every* recipe, so cross-recipe **differences** in
`compute_honest_s` isolate the relocated work. (e1's relocated augmentation slice,
`precompute_epochs`, is ~31 ms in isolation; `compute_honest_s` is the stricter
anti-gaming number — both targets charge reset+whiten too — not literally
"train-loop + 31 ms".) The per-section `prepare` breakdown in the summary
is diagnostic only (it is not CUDA-synced, to avoid perturbing the official total).

One subtlety we keep consistent across all recipes: the loader's **static
`proc_images` cache** (deterministic normalise + pad, plus a one-time random
pre-flip) is part of `build` for everyone — it is one-time, not per-run, and the
build is run under a fixed `BUILD_SEED` so that pre-flip is identical across
containers rather than a per-host roll. The **per-epoch** crop / flip-toggle /
jitter / shuffle is the per-run augmentation, charged (e1) or timed (eager). The
runner reports `compute_honest_s` as this full per-run `prepare` + train (measured
n=200: e1 1.828 s vs hiverge 1.978 s → 7.6%).

### Is it fair that e1's build does far more than hiverge's?

Yes, for the *compute* metrics. e1's build is heavy (multi-resolution compile
warmup, per-step + whole-run CUDA-graph capture, a `_mega_check` self-test, and a
throwaway real-data precompute to capture against); hiverge's is light. But the
boundary, not the weight, is what matters:

- **Free is reserved for one-time, amortised work available to any recipe** —
  compile and graph *capture*. Capture is a *tracing* cost paid once, exactly like
  `torch.compile`; the actual training FLOPs still execute in the **timed** replay
  on fresh per-trial data. e1's win is "paid once to remove per-step launch
  overhead," which hiverge could adopt too — a real efficiency difference, not
  hidden compute.
- **Per-run work is charged.** The one per-run thing e1 relocates —
  augmentation materialisation — is charged in `prepare`; that is what
  compute-honest exists for.
- **Build-time real-data work is throwaway.** e1's build-time `precompute_epochs`
  and `_mega_check` never feed a measured trial: charged `prepare` re-does reset +
  whitening + `precompute_epochs(out=...)` into the same buffers before every
  trial's train. (`verify.py` checks this empirically — it hashes the graph-bound
  `epoch_data` after build and after each prepare and asserts prepare overwrites
  it, so no build-time data reaches a measured trial.)

So the `build` vs charged-`prepare` split *is* the fairness mechanism: it draws the
line between legitimate one-time amortisation (capture/compile) and relocating
per-run work (gaming). Build's only un-charged real cost is thermal — see below.

## Timing & 200-run methodology

- **Timer:** `time.perf_counter()` bracketed by `torch.cuda.synchronize()` on both
  ends (single-GPU wall-clock, the airbench/hiverge convention). CUDA-graph
  replay is async, so the timer always syncs *after* the work. CUDA events would
  be diagnostic only; we don't use them as the official number.
- **Pooled across containers:** `n` trials are exact-allocated round-robin over
  ≤24 A100-SXM4 containers (above ~24 Modal silently drops results). Each
  container builds once, then runs its assigned global trial ids. A lost
  container drops only its trials; missing ids are **topped up** in a second
  wave so the run hits `n`.
- **Host hygiene:** every container rejects non-SXM / multi-GPU hosts and rerolls
  (Modal retry). No `time.sleep` cooldown is allowed in any measured phase (the
  old e1 exploit). Per-trial RNG is reset so trial order doesn't matter.
- **Diagnostics:** the summary reports per-container train-loop means + their
  spread, per-trial GPU telemetry (temp/power/clock), and a drop-first-trial view.

**First trial / build heat.** Build is free for the metric but not thermally
free: e1's build (compile + mega-graph capture + `_mega_check`) is heavier than
hiverge's, so e1 enters its first measured trial slightly warmer. This is a
**conservative** asymmetry — heat can only slow a GPU, never speed it, so a
heavier build can only make e1's *own* early trials slower, never faster; it can
only *understate* e1's advantage, never inflate it. Empirically it is ~9 ms
(~0.5%) on e1's first trial and ~0 on hiverge, at low absolute temps (32–41 °C,
far from the throttle wall). We therefore (a) report **all trials** as the
headline (no data dropped, so no selection effects), and (b) carry
**drop-first-per-container** + the telemetry as sensitivities. We keep
`_mega_check` (e1's correctness self-test) rather than disabling it for build
speed — its only cost is this tiny, conservative heat.

## Output

`runners/results/<UTCstamp>_<label>_<n>/` holds `config.json`, `trials.jsonl`
(one raw record per trial), and `summary.json` (the compute-honest aggregate with
significance + cluster/drop-first diagnostics). A flat copy is mirrored to
`runners/results/runner_<label>.json` for `figures/plot.py`.

Per-step accuracy-vs-walltime trajectory curves are **not** produced by this
runner: the timed train loop is pure compute, with no in-loop validation.

## Files

```
runners/
  harness.py        boundary policy + dataclasses + timing + guards + orchestrator
  validate.py       vendored grader (6-view TTA, no_grad-not-inference_mode)
  targets.py        recipe adapters + ablation cells + the named-target registry
  run.py            Modal image/app/worker + CLI + aggregation + result writing
  verify.py         empirical harness self-checks on real hardware (see its header)
  recipes/          vendored recipe code the adapters drive (self-sustaining)
    hiverge_base.py   copy of base/cifar10_speedrun.py (one line neutralised; see header)
    e1.py             fable's e1 champion + 2 runner deltas (see file header)
    e5.py             fable's e5 recipe (28→32 curriculum on plain torch.compile)
  results/          run outputs land here
```

(Consolidated from a finer-grained tree for cleanliness; the pure logic is still
modular. `run.py` is one file because Modal loads the entrypoint specially. The
copies in `recipes/` are vendored so the harness never reaches into other dirs at
runtime; see each file's header for provenance and the deltas, if any, vs its source.)

## The main targets

- **`hiverge`** — faithful 7.65-epoch port of `recipes/hiverge_base.py` (the
  genuine ~1.96 s baseline). Faithful means base's schedule semantics, **not**
  the 8.5-epoch recipe with a smaller epoch number: LR decays over
  `total_train_steps` to 0, **no EMA/lookahead tail**, whitening init from
  `images[:960]` (not `[:5000]`). Reuses base's own `CifarNet`/`Muon`/`CifarLoader`.
- **`e1`** — the champion. Reuses `recipes/e1.py`'s own `setup()` process-cache split:
  first `setup()` = one-time build; cache-hit `setup()` + `fs.prepare()` = charged
  prepare (the `fs.prepare()` graph-buffer rebind/zero was originally *inside* the
  timed `train()` and is relocated to charged prepare); `fs.mega.replay()` = timed.
- **`e5` / `e5_full32`** — fable's plain-`torch.compile` recipe (inductor
  auto-CUDA-graphs the step; no hand-rolled graphs), run identically except the
  resolution schedule (shipped 28→32 curriculum vs all-32). This is the README's
  single-variable curriculum pair; augmentation runs inside timed `train`, like
  hiverge.
- **`e1a_*` / `hiverge765_*` / `hiverge_nocg`** — the ablation cells behind the
  README's curriculum × graph-mode grid (plus pooling, augmentation placement,
  and the "hiverge isn't launch-bound" controls); see `targets.py`.

## Enforcement vs. trust

This is a harness for **trusted, curated recipes**, not a security sandbox — a
malicious file can always smuggle work through globals. We implement the cheap,
high-value guards (SXM/single-GPU pin, no-sleep in measured phases, per-trial RNG
reset, loader-state reset, eval weight-mutation fingerprint) and skip the
over-engineered ones (static proofs that `build` didn't touch train data). The
charged-`prepare` design is the structural protection: anything per-run a recipe
does is charged, so the only way to shrink a number is to make the work genuinely
cheaper, not to hide it.
