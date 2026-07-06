# What enables the 24px curriculum?

**Question.** Deepening the resolution curriculum from 28→32 to 24→28→32 is worth
−0.081 s inside e1 (1.903 → 1.823 s at matched accuracy, n=200). On the
plain-`torch.compile` e5 recipe the same move *loses* time: 24/25px starts miss
the accuracy gate at schedules where the 28px start passes, and the longer
gate-clearing schedule erases the cheaper steps. Which of e1's changes makes the
24px start profitable?

**Answer.** Two independent things, each worth roughly half; no other recipe
difference matters:

1. **Resize, don't crop** *(accuracy side)*. e1 builds low-res inputs by
   bilinearly downsampling full 32px crops; e5 random-crops a 24px *window*.
   Swapping only the input mechanism inside e5 reproduces e1's accuracy
   behaviour exactly — downsampling roughly halves the accuracy cost of a 24px
   start (−0.0012 vs −0.0019).
2. **An execution path that actually realizes the FLOPs saving** *(time side)*.
   Inside e1's whole-run CUDA graph a 24px epoch costs 0.575× a 32px epoch —
   the (24/32)² = 0.5625 FLOPs theory almost exactly. On plain compile the same
   epoch delivers only ~0.65× steady-state (per-step eager optimizer/loader
   overhead), and ~0.82× as charged (a per-container first-trial warm-up leak
   that appears only at the extra resolution).

All cells: `runners/` harness, compute-honest metric, A100-SXM4; labels map to
`runners/results/`.

## Context: the full decomposition ladder

The hiverge → e1 gap decomposes as follows. Every cell runs the shortest
schedule that still meets the accuracy gate, so cells are accuracy-matched
(means 0.9402–0.9405; n=200; all certified):

| change, at matched accuracy | time | Δ |
| :-- | :-- | :-- |
| hiverge (full-32) → e5's recipe, full-32 | 1.978 → 1.993 s | ≈ neutral |
| e5: full-32 → 28→32 curriculum *(single-variable)* | 1.993 → 1.865 s | **−0.128 s** |
| e5 → e1's full systems stack, same 28→32 curriculum | 1.865 → 1.903 s | +0.038 s |
| e1: 28→32 → 24→28→32 *(single-variable)* | 1.903 → 1.823 s | **−0.081 s** |

The two negative steps are single-variable: within one recipe, only the
resolution schedule changes. The recipe swaps are not single-variable, but both
point the same way — at a fixed schedule, no recipe or systems change helps.
This document explains the last row: why deepening the curriculum pays inside
e1 when the same move loses time in e5.

## Hypotheses (as registered before the runs)

- **H1 — input mechanism (accuracy side).** The recipes build low-res batches
  differently. e5 *crops*: a random `crop_size` window of the padded 32px image
  ([`e5.py:411-417`](../runners/recipes/e5.py)) — this is how all of e5's
  reduced-resolution training works, including the 28px epochs of its shipped
  28→32 curriculum; at 24px the network sees 75% of the linear field of view,
  at native scale. e1 *downsamples*: the loader
  always produces standard 32px crops and the model bilinearly interpolates
  them to `res` inside `forward`
  ([`e1.py:752-755`](../runners/recipes/e1.py)) — full scene, reduced detail;
  progressive *resizing* in the fast.ai sense. Prediction: cropping destroys
  more accuracy than downsampling at 24px.
- **H2 — other recipe differences (accuracy side).** e1's remaining training
  changes (EMA/tail details, GlobalAmaxPool head, aug interactions) make its
  accuracy generically more robust at low resolution.
- **H3 — step-cost dilution (time side).** Even at equal accuracy cost, the
  24px start pays less on plain compile: per-step, resolution-independent
  overhead (eager `_foreach` optimizer + loader work between the auto-graphed
  forward/backward) and warm-up effects dilute the (r/32)² saving. e1's
  whole-run graph contains everything, so its step time tracks FLOPs.

## Experiments

1. **Fixed-epoch accuracy contrast** (isolates the accuracy side; schedule held
   at 8.5 epochs in every cell, so no knee-tuning confound): 24→28→32 vs
   28→32 in each recipe, plus an e5 variant (`e5_ds`, in
   [`targets.py`](../runners/targets.py)) that keeps everything of e5 but
   replaces window-cropping with crop-at-32-then-bilinear-resize — the
   H1-vs-H2 discriminator.
2. **Per-resolution per-epoch cost fit** (isolates the time side; no new runs):
   regress each cell's train-loop diagnostic on its epochs-at-each-resolution
   (`train_loop ≈ a + Σ_r epochs_r · c_r`) per recipe over all n≥8 cells, and
   compare ĉ_r/ĉ_32 with the (r/32)² FLOPs theory.

## Results

### Accuracy side — the mechanism is the whole story

Fixed 8.5-epoch cells (± is the s.e. of the mean):

| recipe / low-res mechanism | 28-start acc | 24-start acc | Δacc (24 − 28) |
| :-- | --: | --: | --: |
| e1 (downsample) | 0.94059 ± 0.00018 (n=64) | 0.93935 ± 0.00017 (n=64) | **−0.00124 ± 0.00025** |
| e5 shipped (window-crop) | 0.94077 ± 0.00012 (n=128) | 0.93889 ± 0.00018 (n=64) | **−0.00188 ± 0.00022** |
| e5 + downsample (`e5_ds`) | — (same recipe) | 0.93957 ± 0.00016 (n=64) | **−0.00120 ± 0.00020** |

Swapping only the input mechanism moves e5's deepening penalty from −0.00188 to
−0.00120 — statistically identical to e1's −0.00124. So **H1 is confirmed and
H2 is rejected**: cropping to 24px costs ~0.0007 more accuracy than
downsampling (≈2.8σ on the crop-vs-downsample contrast at n=64), and e1's other
recipe changes buy nothing on this axis. Intuition: a 24px window discards 25%
of the linear field of view (and its random placement adds label-irrelevant
variability), while a 24px resize keeps the whole object at lower detail.

Even with downsampling, a 24px start costs ~0.0012 accuracy at fixed epochs —
the curriculum is never free; it has to be bought back with a longer schedule
(≈0.4–0.5 epochs at the ~0.0024–0.0030 acc/epoch slope near the gate).

*(The `e5_ds` cells run ~0.5 s slower than shipped e5 — an artifact of the
patch interpolating the full 50k-image tensor every epoch rather than
per-batch inside a compiled/graphed step, as e1 does. Those cells are used
only for accuracy; their times are excluded from every cost claim.)*

### Time side — only the whole-run graph delivers the FLOPs curve

Weighted least-squares fit of train-loop time vs epochs-at-resolution
(12 e5 cells, 12 e1 cells; e1 residuals ≤ 15 ms):

| | e1 (whole-run graph) | e5 (plain compile) | FLOPs theory (r/32)² |
| :-- | --: | --: | --: |
| c24/c32 | **0.575** | 0.82 as charged / ~0.65 steady-state | 0.562 |
| c28/c32 | **0.774** | 0.811 | 0.766 |
| per-step overhead | ~0.2 ms | ~1.5–4 ms | 0 |

Two separate leaks on plain compile:

- **Steady-state per-step overhead.** The eager `_foreach` optimizer and loader
  work between auto-graphed steps cost the same at every resolution, so the
  relative saving of a small step shrinks (c28 already 0.81× vs 0.77× theory).
- **Per-container warm-up at the extra resolution.** e5's first trial per
  container runs ~0.6 s slow *only in 24px cells* (drop-first mean 1.780 vs
  all-trials 1.856, n=64) — lazy compile/autotune work at the unwarmed size
  that survives the recipe's own warmup. The compute-honest metric charges all
  trials, so this leak (~+0.05–0.08 s at production container counts) is part
  of the price of an extra resolution on plain compile. e1's cells show no
  such gap — after graph capture there is nothing left to warm.

### Knee economics — putting both sides together

Value of deepening 28→32 → 24→28→32, at gate-clearing schedules:

| | savings per 8.5 ep | accuracy to repay | repay cost | net |
| :-- | --: | --: | --: | --: |
| e1 | 0.205 s | 0.0012 (≈ +0.45 ep) | ~0.11 s | **−0.08 s** *(measured: −0.081, n=200)* |
| e5 shipped | 0.10 s (0.18 spill-free) | 0.0019 (≈ +0.8 ep) | ~0.20 s | **+0.03…+0.10 s (loss)** |

The two effects compound: e5 pays ~0.09 s more to repay the crop-induced extra
accuracy loss *and* collects ~0.03–0.10 s less from the cheap steps. Fixing
either one alone leaves the 24px start ≈ break-even on plain compile; e1 has
both, which is what turns the deep start into its −0.081 s edge.

## Conclusion

Nothing exotic enables the 24px curriculum. It is (a) **progressive resizing
done right** — downsample the full image; never crop away field of view at low
res — and (b) **step execution clean enough that a 0.56× FLOPs step is a 0.56×
wall-clock step** (a whole-run graph achieves this; plain `torch.compile` with
an eager optimizer leaks a few ms per step plus per-size warm-up and only
delivers ~0.65–0.82×). The rest of e1's recipe contributes nothing measurable
to low-resolution accuracy (H2 rejected). Even done right, the 24px start is
bought, not free: ~0.0012 accuracy repaid with ~half an epoch, netting
≈ −0.08 s on an A100.

## Reproducing

```bash
# accuracy contrasts (fixed 8.5 epochs)
uv run modal run runners/run.py::trial --target e1a_prog_mega --n 64 \
  --overrides '{"train_epochs":8.5,"decay_epochs":8.125}' --label e1a_prog24_s85_n64
uv run modal run runners/run.py::trial --target e1a_prog_mega --n 64 \
  --overrides '{"res_schedule":[28,28,28,28,28,32,32,32,32],"train_epochs":8.5,"decay_epochs":8.125}' \
  --label e1a_prog28_s85_n64
uv run modal run runners/run.py::trial --target e5 --n 64 --containers 8 \
  --overrides '{"res_schedule":[24,24,24,28,28,28,32,32,32],"train_epochs":8.5,"decay_epochs":8.375}' \
  --label e5_prog24_s85_n64b
uv run modal run runners/run.py::trial --target e5_ds --n 64 --containers 8 \
  --overrides '{"res_schedule":[24,24,24,28,28,28,32,32,32],"train_epochs":8.5,"decay_epochs":8.375}' \
  --label e5_ds24_s85_n64
# (28-start e5 baseline: e5_mild_s850_n128, already in results/)
```

The cost fit ([`rescost.py`](rescost.py), run from the repo root) regresses
`diagnostics.train_loop` from every `runners/results/*/summary.json` on
epochs-at-resolution derived from each cell's `config.json`.
