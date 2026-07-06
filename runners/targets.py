"""Recipe adapters + the named-target registry.

Self-contained: the recipe code lives in `runners/recipes/` (vendored copies),
imported as a normal subpackage — the harness does NOT reach into base/ or any
external recipe dir at runtime. Each adapter fits its recipe into the central
build/charged-prepare/timed-train boundary (see harness.py) WITHOUT editing the
recipe: it only orchestrates the recipe's internals, conserving "what's loaded
where" (the recipe's own model build + data loading are unchanged; we only
relocate WHERE on the clock each piece runs).

  hiverge -> Hiverge765 : faithful 7.65-epoch port of recipes/hiverge_base.py
                          (the genuine ~1.96s baseline).
  e1      -> E1Target    : the champion (whole-run mega-graph + prog-res +
                          GlobalAmaxPool + fused-foreach).

The recipe modules import torch, so adapters import them LAZILY inside build()
(which runs in the GPU container), keeping this module importable torch-free.
e2..e5 get added here once core + these two are validated on Modal.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from math import ceil

from .harness import BuildContext, BuildState, RunContext, RunState

_E1_DEFAULT_HYP = None   # pristine snapshot of e1's hyp, captured on first build
_E1_DEFAULT_POOL = None  # pristine GlobalAmaxPool class (for pool-swap ablations)


# ======================================================================
# hiverge — faithful 7.65-epoch port of recipes/hiverge_base.py
# ======================================================================
# This is the GENUINE baseline (~1.96s train-only), NOT the padded 8.5-epoch
# bench/recipes/hiverge.py. A faithful port is NOT "the 8.5 recipe with
# train_epochs=7.65"; it uses base's schedule semantics:
#   * total_train_steps = ceil(7.65 * len(loader))   (base main())
#   * LR decays linearly over total_train_steps to 0 (no separate decay, no floor)
#   * NO EMA / lookahead tail                          (base has none)
#   * whitening init from train_loader.images[:960]    (base; NOT [:5000])
# We reuse base's own classes, so model + data loading are byte-identical to base;
# only the clock placement changes:
#   build   = compile + synthetic warmup + pre-build static proc_images cache
#   prepare = model.reset() + init_whiten([:960]) + fresh optimisers + epoch reset
#   train   = the 7.65-epoch loop (in-loop aug, fused SGD + Muon, linear LR decay)
_HV = dict(BS=1536, BIAS_LR=0.0573, HEAD_LR=0.5415, SGD_MOM=0.825,
           MUON_LR=0.205, MUON_MOM=0.655, LS=0.09, WB_EPOCHS=0.2)
_HV["WD"] = 1.0418e-06 * _HV["BS"]
_HV_AUG = {"flip": True, "translate": 2,
           "color_jitter": {"enabled": True, "brightness_range": 0.1399, "contrast_range": 0.1308}}


def _hv_make_optimizers(base, model, total_train_steps):
    import torch
    filt = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_b = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    cfgs = [
        dict(params=[model.whiten.bias], lr=_HV["BIAS_LR"], weight_decay=_HV["WD"] / _HV["BIAS_LR"]),
        dict(params=norm_b, lr=_HV["BIAS_LR"], weight_decay=_HV["WD"] / _HV["BIAS_LR"]),
        dict(params=[model.head.weight], lr=_HV["HEAD_LR"], weight_decay=_HV["WD"] / _HV["HEAD_LR"]),
    ]
    o1 = torch.optim.SGD(cfgs, momentum=_HV["SGD_MOM"], nesterov=True, fused=True)
    o2 = base.Muon(filt, lr=_HV["MUON_LR"], momentum=_HV["MUON_MOM"], nesterov=True,
                   norm_freq=4, total_train_steps=total_train_steps, weight_decay=_HV["WD"])
    o2.param_groups[0]["momentum_buffer_dtype"] = torch.half
    for o in (o1, o2):
        for g in o.param_groups:
            g["initial_lr"] = g["lr"]
    return o1, o2


def _hv_loop(model, loader, o1, o2, total_steps, wb_steps, forward_step):
    """recipes/hiverge_base.py main()'s training loop, verbatim in structure.
    In-loop augmentation (loader.__iter__) is therefore TIMED."""
    lr1_base = 1.0 / max(1, wb_steps)
    lr2_base = 1.0 / total_steps
    lr1_init = o1.param_groups[0]["initial_lr"]
    lr2_groups = o1.param_groups[1:] + o2.param_groups
    lr2_init = [g["initial_lr"] for g in lr2_groups]
    step = 0
    for _ in range(ceil(total_steps / len(loader))):
        model.train()
        for inputs, labels in loader:
            loss = forward_step(inputs, labels, step < wb_steps)
            loss.backward()
            o1.param_groups[0]["lr"] = lr1_init * (1 - step * lr1_base)
            f2 = 1 - step * lr2_base
            for g, init in zip(lr2_groups, lr2_init):
                g["lr"] = init * f2
            o1.step(); o1.zero_grad(set_to_none=True)
            o2.step(); o2.zero_grad(set_to_none=True)
            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break


class Hiverge765:
    name = "hiverge"
    recipe = "runners/recipes/hiverge_base.py"  # vendored source (provenance)
    plain_eval = True

    def build(self, ctx: BuildContext) -> BuildState:
        import torch
        import torch.nn.functional as F

        from .recipes import hiverge_base as base
        unknown = set(ctx.overrides) - {"train_epochs"}
        if unknown:
            raise ValueError(f"hiverge only supports the 'train_epochs' override; got {unknown}")
        epochs = float(ctx.overrides.get("train_epochs", 7.65))
        model = base.CifarNet().cuda().to(memory_format=torch.channels_last)
        model.compile(mode="max-autotune")
        loader = base.CifarLoader(ctx.data_path, train=True, batch_size=_HV["BS"], aug=_HV_AUG)
        total = ceil(epochs * len(loader))
        wb = ceil(_HV["WB_EPOCHS"] * len(loader))

        @torch.compile(mode="max-autotune", fullgraph=True)
        def forward_step(inputs, labels, whiten_bias_grad: bool):
            out = model(inputs, whiten_bias_grad=whiten_bias_grad)
            return F.cross_entropy(out, labels, label_smoothing=_HV["LS"], reduction="sum")

        # compile warmup on synthetic data (untimed)
        warm = base.CifarLoader(ctx.data_path, train=True, batch_size=_HV["BS"], aug=_HV_AUG)
        warm.images = torch.randn_like(warm.images)
        warm.labels = torch.randint_like(warm.labels, 0, 10)
        with torch.no_grad():
            model.init_whiten(warm.normalize(warm.images[:960]))
        wo1, wo2 = _hv_make_optimizers(base, model, total)
        _hv_loop(model, warm, wo1, wo2, max(2 * wb + 6, 16), wb, forward_step)
        del warm, wo1, wo2
        torch.cuda.synchronize()

        # pre-build the loader's static proc_images cache (untimed); the per-EPOCH
        # crop/flip/jitter/shuffle still runs inside the timed loop (see README).
        if not loader.proc_images:
            imgs = loader.proc_images["norm"] = loader.normalize(loader.images)
            if loader.aug.get("flip", False):
                imgs = loader.proc_images["flip"] = base.batch_flip_lr(imgs)
            pad = loader.aug.get("translate", 0)
            if pad > 0:
                loader.proc_images["pad"] = F.pad(imgs, (pad,) * 4, "reflect")

        return BuildState(obj={"model": model, "loader": loader, "forward_step": forward_step,
                               "total": total, "wb": wb}, recipe=base)

    def prepare(self, state: BuildState, run: RunContext) -> RunState:
        import torch
        base, o = state.recipe, state.obj
        model, loader = o["model"], o["loader"]
        with run.charge.section("reset"):
            model.reset()
            model.zero_grad(set_to_none=True)  # don't inherit stale grads from a prior errored trial
            # The vendored loader rebuilds its proc_images cache whenever epoch==0
            # (which would charge the static normalize/pre-flip/pad to the TIMED
            # loop and break the boundary — audit finding #1). Start at an even
            # epoch != 0: epoch%2 parity matches a fresh epoch-0 start, but the
            # build-time cache survives. (Also resets per-trial so flip-parity
            # doesn't leak across trials.)
            loader.epoch = 2
        with run.charge.section("whiten"):
            with torch.no_grad():
                model.init_whiten(loader.normalize(loader.images[:960]))
        with run.charge.section("optim"):
            o1, o2 = _hv_make_optimizers(base, model, o["total"])
        return RunState(model=model, obj={"o1": o1, "o2": o2})

    def train(self, state: BuildState, run: RunState, ctx: RunContext) -> None:
        o = state.obj
        _hv_loop(run.model, o["loader"], run.obj["o1"], run.obj["o2"],
                 o["total"], o["wb"], o["forward_step"])


class Hiverge765NoCG(Hiverge765):
    """Diagnostic: the hiverge baseline but compiled `max-autotune-no-cudagraphs`
    (inductor's automatic CUDA-graphing OFF) and run eagerly — a genuinely
    UN-graphed hiverge. Reuses Hiverge765.prepare/train; only build() differs.
    Lets us measure how much graphing is worth on hiverge's OWN net: compare
    vs `hiverge` (max-autotune, inductor-graphed fwd+bwd) and the per-step cells."""
    name = "hiverge_nocg"

    def build(self, ctx: BuildContext) -> BuildState:
        import torch
        import torch.nn.functional as F

        from .recipes import hiverge_base as base
        unknown = set(ctx.overrides) - {"train_epochs"}
        if unknown:
            raise ValueError(f"hiverge_nocg only supports the 'train_epochs' override; got {unknown}")
        epochs = float(ctx.overrides.get("train_epochs", 7.65))
        model = base.CifarNet().cuda().to(memory_format=torch.channels_last)
        model.compile(mode="max-autotune-no-cudagraphs")
        loader = base.CifarLoader(ctx.data_path, train=True, batch_size=_HV["BS"], aug=_HV_AUG)
        total = ceil(epochs * len(loader))
        wb = ceil(_HV["WB_EPOCHS"] * len(loader))

        @torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
        def forward_step(inputs, labels, whiten_bias_grad: bool):
            out = model(inputs, whiten_bias_grad=whiten_bias_grad)
            return F.cross_entropy(out, labels, label_smoothing=_HV["LS"], reduction="sum")

        warm = base.CifarLoader(ctx.data_path, train=True, batch_size=_HV["BS"], aug=_HV_AUG)
        warm.images = torch.randn_like(warm.images)
        warm.labels = torch.randint_like(warm.labels, 0, 10)
        with torch.no_grad():
            model.init_whiten(warm.normalize(warm.images[:960]))
        wo1, wo2 = _hv_make_optimizers(base, model, total)
        _hv_loop(model, warm, wo1, wo2, max(2 * wb + 6, 16), wb, forward_step)
        del warm, wo1, wo2
        torch.cuda.synchronize()

        if not loader.proc_images:
            imgs = loader.proc_images["norm"] = loader.normalize(loader.images)
            if loader.aug.get("flip", False):
                imgs = loader.proc_images["flip"] = base.batch_flip_lr(imgs)
            pad = loader.aug.get("translate", 0)
            if pad > 0:
                loader.proc_images["pad"] = F.pad(imgs, (pad,) * 4, "reflect")

        return BuildState(obj={"model": model, "loader": loader, "forward_step": forward_step,
                               "total": total, "wb": wb}, recipe=base)


# ======================================================================
# e1 — champion (reuse the recipe's own setup() cache split)
# ======================================================================
# recipes/e1.py already separates one-time build from per-run work via a process
# _CACHE: the FIRST setup() compiles + captures the mega-graph + allocs buffers
# (build); LATER setup() calls hit the cache and only re-init to untrained +
# re-materialise the augmented stream (reset_all + precompute_epochs(out=...)).
# So we reuse setup() and only move the boundary:
#   build   = first setup()                     -> untimed
#   prepare = cache-hit setup() + fs.prepare()  -> charged (precompute_epochs is the
#             ~31ms per-run aug; fs.prepare() rebinds/zeroes graph-bound optim
#             buffers — it sat INSIDE the timed train() originally, relocated here)
#   train   = fs.mega.replay()                  -> timed (one async replay; the
#             orchestrator syncs after)
class E1Target:
    name = "e1"
    recipe = "runners/recipes/e1.py"  # vendored source (provenance)
    plain_eval = True

    def build(self, ctx: BuildContext) -> BuildState:
        global _E1_DEFAULT_HYP
        from .recipes import e1 as e1mod
        if _E1_DEFAULT_HYP is None:                 # capture pristine defaults once
            _E1_DEFAULT_HYP = copy.deepcopy(e1mod.hyp)
        # Bind the data path explicitly rather than trusting an import-time global,
        # and reset hyp to defaults so a warm container can't carry stale overrides
        # or a stale graph cache (audit findings #12/#13/#4).
        e1mod.DATA_PATH = ctx.data_path
        e1mod._CACHE.clear()
        e1mod.hyp.clear(); e1mod.hyp.update(copy.deepcopy(_E1_DEFAULT_HYP))
        if ctx.overrides:
            e1mod.hyp.update(ctx.overrides)
        e1mod.setup()  # one-time compile + mega-graph capture + caches (untimed)
        return BuildState(obj={}, recipe=e1mod)

    def prepare(self, state: BuildState, run: RunContext) -> RunState:
        e1mod = state.recipe
        # e1's reset_all() does NOT reset loader.epoch, but precompute_epochs uses
        # epoch%2 for flip parity and increments it per epoch — so without this the
        # per-trial augmentation parity would depend on trial order (audit #2).
        e1mod._CACHE["train_loader"].epoch = 0
        with run.charge.section("reset+whiten+precompute"):
            model, extra = e1mod.setup()         # cache hit: reset_all + precompute(out=)
        with run.charge.section("graph_prepare"):
            extra["fullstep"].prepare()          # rebind + zero graph-bound optim buffers
        return RunState(model=model, obj={"extra": extra})

    def train(self, state: BuildState, run: RunState, ctx: RunContext) -> None:
        assert run.prepared_token == ctx.trial_id, "train() got an unprepared run"
        run.obj["extra"]["fullstep"].mega.replay()  # timed; orchestrator syncs after


# ======================================================================
# e1 ablation cells (Pro's design) — single-lever toggles on the e1 model:
#   curriculum {prog | full32} x graph {mega | perstep | eager} x aug {precomputed
#   | loader} x pool {global | maxpool3}. Build always runs e1.setup() (free); only
#   prepare/train orchestration changes. Graph modes require aug=precomputed; the
#   pool swap is legal only at full-32 (24px breaks MaxPool2d(3) geometry).
# ======================================================================
class _NoValCtx:
    """Preserve run_steps()'s EMA/lookahead behaviour without timed validation
    (EMA creation/updates are gated on `ctx is not None`)."""
    def validate(self, *args, **kwargs):
        return None


class _PrecomputedStepLoader:
    """Feed e1.precompute_epochs() step buffers through the eager run_steps() loop,
    so eager-vs-graph differs only in graphing, not augmentation location.
    run_steps() expects len(loader)==steps_per_epoch and calls iter() once per epoch;
    precompute_epochs() returns a flat per-step list, so yield the next epoch slice."""
    def __init__(self, step_data, steps_per_epoch):
        self.step_data = step_data
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        start = self.epoch * self.steps_per_epoch
        end = min(start + self.steps_per_epoch, len(self.step_data))
        self.epoch += 1
        for i in range(start, end):
            yield self.step_data[i]


class E1AblationTarget:
    recipe = "runners/recipes/e1.py"
    plain_eval = True
    # subclasses set these:
    name = "e1a"
    curriculum = "prog"        # "prog" | "full32"
    graph_mode = "mega"        # "mega" | "perstep" | "eager"
    aug_mode = "precomputed"   # "precomputed" | "loader"
    pool = "global"            # "global" | "maxpool3"

    def build(self, ctx: BuildContext) -> BuildState:
        global _E1_DEFAULT_HYP, _E1_DEFAULT_POOL
        if self.graph_mode == "autocg":
            # Flip the compiled step to plain max-autotune (inductor auto-cudagraphs it)
            # and skip e1's manual graph capture — set BEFORE importing e1 so the
            # @torch.compile decorator + setup() see them. The eager run_steps loop then
            # replays inductor's own cudagraphs. Default (other cells) is unchanged.
            import os
            os.environ["E1_STEP_COMPILE_MODE"] = "max-autotune"
            os.environ["E1_SKIP_CAPTURE"] = "1"
        from .recipes import e1 as e1mod

        if _E1_DEFAULT_HYP is None:
            _E1_DEFAULT_HYP = copy.deepcopy(e1mod.hyp)
        if _E1_DEFAULT_POOL is None:
            _E1_DEFAULT_POOL = e1mod.GlobalAmaxPool

        # Reset all mutable module state so warm containers can't leak config.
        e1mod.DATA_PATH = ctx.data_path
        e1mod._CACHE.clear()
        e1mod.GlobalAmaxPool = _E1_DEFAULT_POOL
        e1mod.hyp.clear()
        e1mod.hyp.update(copy.deepcopy(_E1_DEFAULT_HYP))
        if ctx.overrides:
            e1mod.hyp.update(ctx.overrides)

        if self.curriculum == "full32":
            e1mod.hyp["res_schedule"] = [32] * len(_E1_DEFAULT_HYP["res_schedule"])
        elif self.curriculum != "prog":
            raise ValueError(f"bad curriculum={self.curriculum!r}")

        if self.pool == "maxpool3":
            if self.curriculum != "full32":
                raise ValueError("MaxPool2d(3) pool swap is legal only with curriculum='full32'")
            e1mod.GlobalAmaxPool = lambda: e1mod.nn.MaxPool2d(3)
        elif self.pool != "global":
            raise ValueError(f"bad pool={self.pool!r}")

        if self.graph_mode in ("mega", "perstep") and self.aug_mode != "precomputed":
            raise ValueError("graph modes require aug_mode='precomputed'")
        if self.graph_mode not in ("mega", "perstep", "eager", "autocg"):
            raise ValueError(f"bad graph_mode={self.graph_mode!r}")
        if self.aug_mode not in ("precomputed", "loader"):
            raise ValueError(f"bad aug_mode={self.aug_mode!r}")

        # One-time build (free). For eager cells this captures graphs we won't use,
        # deliberately: build is free and this reuses e1's tested setup path rather
        # than a partial reimplementation.
        e1mod.setup()
        return BuildState(obj={"graph_mode": self.graph_mode, "aug_mode": self.aug_mode}, recipe=e1mod)

    @staticmethod
    def _extra_from_cache(e1mod):
        c = e1mod._CACHE
        return {"optimizers": [c["opt1"], c["opt2"]], "fullstep": c["fs"],
                "epoch_data": c["epoch_data"], "train_loader": c["train_loader"],
                "total_train_steps": c["total_train_steps"], "decay_steps": c["decay_steps"],
                "whiten_bias_train_steps": c["whiten_bias_train_steps"]}

    def prepare(self, state: BuildState, run: RunContext) -> RunState:
        e1mod = state.recipe
        mode, aug = state.obj["graph_mode"], state.obj["aug_mode"]
        e1mod._CACHE["train_loader"].epoch = 0  # reset per-trial flip parity

        if aug == "precomputed":
            with run.charge.section("reset+whiten+precompute"):
                model, extra = e1mod.setup()  # cache hit: reset_all + precompute_epochs(out=...)
            if mode in ("mega", "perstep"):
                with run.charge.section("graph_prepare"):
                    extra["fullstep"].prepare()
            return RunState(model=model, obj={"extra": extra})

        # eager + loader: no precompute (in-loop aug is timed in train()), no fs.prepare
        c = e1mod._CACHE
        model, loader = c["model"], c["train_loader"]
        with run.charge.section("reset+whiten"):
            e1mod.reset_all(model, loader, c["opt1"], c["opt2"])
            loader.epoch = 0
        return RunState(model=model, obj={"extra": self._extra_from_cache(e1mod)})

    def train(self, state: BuildState, run: RunState, ctx: RunContext) -> None:
        assert run.prepared_token == ctx.trial_id, "train() got an unprepared run"
        e1mod = state.recipe
        mode, aug = state.obj["graph_mode"], state.obj["aug_mode"]
        extra = run.obj["extra"]

        if mode == "mega":
            extra["fullstep"].mega.replay()
            return
        if mode == "perstep":
            e1mod.run_steps_stepdata(run.model, extra["epoch_data"], extra["fullstep"],
                                     extra["total_train_steps"], extra["decay_steps"])
            return
        # eager
        opt1, opt2 = extra["optimizers"]
        loader = (_PrecomputedStepLoader(extra["epoch_data"], extra["fullstep"].steps_per_epoch)
                  if aug == "precomputed" else extra["train_loader"])
        e1mod.run_steps(run.model, loader, opt1, opt2, extra["decay_steps"],
                        extra["whiten_bias_train_steps"], extra["total_train_steps"],
                        ctx=_NoValCtx(), lr_floor=e1mod.hyp["lr_floor"])


class E1AProgMega(E1AblationTarget):
    name = "e1a_prog_mega"; curriculum = "prog"; graph_mode = "mega"; aug_mode = "precomputed"


class E1AFull32Mega(E1AblationTarget):
    name = "e1a_full32_mega"; curriculum = "full32"; graph_mode = "mega"; aug_mode = "precomputed"


class E1AProgEagerPrecomp(E1AblationTarget):
    name = "e1a_prog_eager_precomp"; curriculum = "prog"; graph_mode = "eager"; aug_mode = "precomputed"


class E1AFull32EagerPrecomp(E1AblationTarget):
    name = "e1a_full32_eager_precomp"; curriculum = "full32"; graph_mode = "eager"; aug_mode = "precomputed"


class E1AProgPerstep(E1AblationTarget):
    name = "e1a_prog_perstep"; curriculum = "prog"; graph_mode = "perstep"; aug_mode = "precomputed"


class E1AFull32Perstep(E1AblationTarget):
    name = "e1a_full32_perstep"; curriculum = "full32"; graph_mode = "perstep"; aug_mode = "precomputed"


class E1AProgEagerLoader(E1AblationTarget):
    name = "e1a_prog_eager_loader"; curriculum = "prog"; graph_mode = "eager"; aug_mode = "loader"


class E1AFull32MaxPool3Mega(E1AblationTarget):
    name = "e1a_full32_maxpool3_mega"; curriculum = "full32"; graph_mode = "mega"; aug_mode = "precomputed"; pool = "maxpool3"


# autocg: e1's net on PLAIN torch.compile (max-autotune auto-cudagraph, no hand-rolled
# graph) — tests whether the aggressive 24->28->32 curriculum's matched-accuracy win
# survives without the manual mega-graph. fwd+bwd auto-cudagraphed; optimizer stays eager.
class E1AProgAutocg(E1AblationTarget):
    name = "e1a_prog_autocg"; curriculum = "prog"; graph_mode = "autocg"; aug_mode = "precomputed"


class E1AFull32Autocg(E1AblationTarget):
    name = "e1a_full32_autocg"; curriculum = "full32"; graph_mode = "autocg"; aug_mode = "precomputed"


# ======================================================================
# Registry
# ======================================================================
@dataclass(frozen=True)
class TargetSpec:
    name: str
    cls: type
    plain_eval: bool = True


TARGETS: dict[str, TargetSpec] = {
    "hiverge": TargetSpec("hiverge", Hiverge765),
    "e1": TargetSpec("e1", E1Target),
}

# e1 ablation cells
for _c in (E1AProgMega, E1AFull32Mega, E1AProgEagerPrecomp, E1AFull32EagerPrecomp,
           E1AProgPerstep, E1AFull32Perstep, E1AProgEagerLoader, E1AFull32MaxPool3Mega,
           E1AProgAutocg, E1AFull32Autocg):
    TARGETS[_c.name] = TargetSpec(_c.name, _c)


def get(name: str) -> TargetSpec:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {sorted(TARGETS)}")
    return TARGETS[name]


# ======================================================================
# hiverge — 7.65-epoch per-step CUDA-graph adapters
# ======================================================================
# Answers: "does adding per-step CUDA graphs to the stock hiverge baseline — and
# nothing else — get it to e1's frontier?" These keep hiverge's exact net /
# optimizers / 7.65-ep schedule / whiten-bias warmup / in-loop-aug semantics / NO
# EMA, and ONLY change eager Python-loop -> per-step CUDA-graph replay. The
# capturable orthogonalizing optimizer is e1.OrthoMomentum, which is hiverge's
# Muon refactored for capture (same half-momentum, nesterov, periodic do_norm
# param-scaling, and Newton-Schulz constants/magnitude); a build-time selftest
# asserts one graphed step matches one eager base.Muon step.
def _hv_make_graph_optimizers(e1mod, model, total_train_steps):
    """HIVERGE optimizer split, but using e1.OrthoMomentum for capture safety."""
    import torch

    conv_filters = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=_HV["BIAS_LR"], weight_decay=_HV["WD"] / _HV["BIAS_LR"]),
        dict(params=norm_biases, lr=_HV["BIAS_LR"], weight_decay=_HV["WD"] / _HV["BIAS_LR"]),
        dict(params=[model.head.weight], lr=_HV["HEAD_LR"], weight_decay=_HV["WD"] / _HV["HEAD_LR"]),
    ]
    opt1 = torch.optim.SGD(param_configs, momentum=_HV["SGD_MOM"], nesterov=True, fused=True)
    opt2 = e1mod.OrthoMomentum(
        conv_filters,
        lr=_HV["MUON_LR"],
        momentum=_HV["MUON_MOM"],
        nesterov=True,
        norm_freq=4,
        total_train_steps=total_train_steps,
        weight_decay=_HV["WD"],
    )
    opt2.param_groups[0]["momentum_buffer_dtype"] = torch.half
    for opt in (opt1, opt2):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return opt1, opt2


def _hv_prebuild_proc_cache(base, loader):
    """Build the static normalize/preflip/pad cache outside the timed loop."""
    import torch.nn.functional as F

    if loader.proc_images:
        return
    images = loader.proc_images["norm"] = loader.normalize(loader.images)
    if loader.aug.get("flip", False):
        images = loader.proc_images["flip"] = base.batch_flip_lr(images)
    pad = loader.aug.get("translate", 0)
    if pad > 0:
        loader.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")


def _hv_materialize_augmented_steps(loader, total_steps):
    """Materialize exactly the batch stream CifarLoader.__iter__ would yield."""
    import torch

    out = []
    step = 0
    for _ in range(ceil(total_steps / len(loader))):
        for inputs, labels in loader:
            out.append((
                inputs.contiguous(memory_format=torch.channels_last),
                labels.contiguous(),
            ))
            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    return out


class _HivergePerStepGraph:
    """One captured full training step per (whiten_bias_grad, do_norm).

    The HIVERGE/no-resolution/no-EMA analogue of e1.FullStepGraph: it captures
    forward + backward + fused SGD + OrthoMomentum's capturable update, with
    per-step LR/NS scalars fed through persistent CUDA 0-dim tensors.
    """

    def __init__(self, model, opt1, opt2, forward_step, total_train_steps, whiten_bias_train_steps):
        import torch

        self.model = model
        self.opt1 = opt1
        self.opt2 = opt2
        self.forward_step = forward_step
        self.total = total_train_steps
        self.wbts = whiten_bias_train_steps
        self.lr1_base = 1.0 / max(1, whiten_bias_train_steps)
        self.lr2_base = 1.0 / total_train_steps
        self.bias_lr = _HV["BIAS_LR"]
        self.head_lr = _HV["HEAD_LR"]
        self.muon_lr = _HV["MUON_LR"]
        self.wd2 = opt2.param_groups[0]["weight_decay"]
        zt = lambda: torch.zeros((), device="cuda", dtype=torch.float32)
        self.lr0 = zt()  # whiten.bias SGD group lr
        self.lr1 = zt()  # norm-bias SGD group lr
        self.lrh = zt()  # head.weight SGD group lr
        self.lro = zt()  # OrthoMomentum lr
        self.tm = zt()   # Newton-Schulz target magnitude
        self.graphs = {}
        self.pool = None

    def _region(self, whiten_bias_grad, do_norm, x=None, y=None, lro=None, tm=None):
        import torch

        model, opt1, opt2 = self.model, self.opt1, self.opt2
        x = self.static_x if x is None else x
        y = self.static_y if y is None else y
        lro = self.lro if lro is None else lro
        tm = self.tm if tm is None else tm

        loss = self.forward_step(x, y, whiten_bias_grad)
        if whiten_bias_grad:
            grads = torch.autograd.grad(loss, [self.wb] + self.plist_rest)
            self.wb.grad = grads[0]
            rest = list(grads[1:])
        else:
            # Eager HIVERGE has wb.grad=None here (the bias is detached). In the
            # captured graph we use a persistent zero grad and force lr0=0 for
            # these variants, so the whiten-bias trajectory matches exactly.
            self.wb.grad = self.wbzero
            rest = list(torch.autograd.grad(loss, self.plist_rest))

        for p, grad in zip(self.plist_rest, rest):
            p.grad = grad

        opt1.step()
        params, outs = opt2._compute_update(do_norm, tm)
        with torch.no_grad():
            for p, update in zip(params, outs):
                p.addcmul_(update, lro, value=-1.0)
            for p in params:
                p.mul_(1 - lro * self.wd2)

    def capture(self, batch):
        import torch

        model, opt1, opt2 = self.model, self.opt1, self.opt2
        x, y = batch
        self.static_x = x.clone().contiguous(memory_format=torch.channels_last)
        self.static_y = y.clone().contiguous()
        self.wb = opt1.param_groups[0]["params"][0]
        p1_rest = [p for group in opt1.param_groups[1:] for p in group["params"]]
        p2 = [meta["param"] for meta in opt2.filter_params_meta]
        self.plist_rest = p2 + p1_rest
        self.wbzero = torch.zeros_like(self.wb)

        for group, lr_tensor in zip(opt1.param_groups, (self.lr0, self.lr1, self.lrh)):
            group["lr"] = lr_tensor
        self.lr0.fill_(self.bias_lr)
        self.lr1.fill_(self.bias_lr)
        self.lrh.fill_(self.head_lr)
        self.lro.fill_(self.muon_lr)
        self.tm.fill_(0.3)

        model.train()
        variants = ((True, False), (True, True), (False, False), (False, True))
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for whiten_bias_grad, do_norm in variants:
                for _ in range(2):
                    self._region(whiten_bias_grad, do_norm)
        torch.cuda.current_stream().wait_stream(warm_stream)
        torch.cuda.synchronize()

        self.params1 = [self.wb] + p1_rest
        self.bufs1 = [opt1.state[p]["momentum_buffer"] for p in self.params1]
        self.params2 = p2
        self.bufs2 = [opt2.state[p]["momentum_buffer"] for p in p2]

        for whiten_bias_grad, do_norm in variants:
            graph = torch.cuda.CUDAGraph()
            if self.pool is None:
                with torch.cuda.graph(graph):
                    self._region(whiten_bias_grad, do_norm)
                self.pool = graph.pool()
            else:
                with torch.cuda.graph(graph, pool=self.pool):
                    self._region(whiten_bias_grad, do_norm)
            self.graphs[(whiten_bias_grad, do_norm)] = graph
        torch.cuda.synchronize()

    def prepare(self):
        """Rebind graph-captured optimizer buffers and reset per-run state."""
        import torch

        with torch.no_grad():
            for p, buf in zip(self.params1, self.bufs1):
                self.opt1.state[p]["momentum_buffer"] = buf
            for p, buf in zip(self.params2, self.bufs2):
                self.opt2.state[p]["momentum_buffer"] = buf
            torch._foreach_zero_(self.bufs1)
            torch._foreach_zero_(self.bufs2)
            self.wbzero.zero_()
            self.opt2.step_count = 0
            self.opt2.last_norm_step = 0
            self.opt2.current_grad_norms = None
            self.model.zero_grad(set_to_none=True)

    def step(self, s, inputs, labels):
        import torch

        with torch.no_grad():
            do_norm, tm_val = self.opt2._advance_schedule()
            f2 = 1.0 - s * self.lr2_base
            self.lr0.fill_(self.bias_lr * (1.0 - s * self.lr1_base) if s < self.wbts else 0.0)
            self.lr1.fill_(self.bias_lr * f2)
            self.lrh.fill_(self.head_lr * f2)
            self.lro.fill_(self.muon_lr * f2)
            self.tm.fill_(tm_val)
            self.static_x.copy_(inputs, non_blocking=True)
            self.static_y.copy_(labels, non_blocking=True)
            self.graphs[(s < self.wbts, do_norm)].replay()
            return do_norm


def _hv_model_snapshot(model):
    import torch

    params = list(model.parameters())
    buffers = [b for b in model.buffers() if b.dtype in (torch.half, torch.float)]
    return params, buffers, [p.detach().clone() for p in params], [b.clone() for b in buffers]


def _hv_model_restore(snapshot):
    import torch

    params, buffers, param_values, buffer_values = snapshot
    with torch.no_grad():
        for p, value in zip(params, param_values):
            p.copy_(value)
        for b, value in zip(buffers, buffer_values):
            b.copy_(value)
    for p in params:
        p.grad = None


def _hv_reset_graph_untrained(model, loader, graph):
    import torch

    model.reset()
    with torch.no_grad():
        model.init_whiten(loader.normalize(loader.images[:960]))
    model.zero_grad(set_to_none=True)
    graph.prepare()
    loader.epoch = 2


def _hv_run_one_eager_base_step(base, model, forward_step, batch, total, wb_steps, s,
                                opt2_step_count=0, opt2_last_norm_step=0):
    opt1, opt2 = _hv_make_optimizers(base, model, total)
    opt2.step_count = opt2_step_count
    opt2.last_norm_step = opt2_last_norm_step

    inputs, labels = batch
    loss = forward_step(inputs, labels, s < wb_steps)
    loss.backward()

    f2 = 1.0 - s * (1.0 / total)
    opt1.param_groups[0]["lr"] = _HV["BIAS_LR"] * (1.0 - s * (1.0 / max(1, wb_steps))) if s < wb_steps else 0.0
    opt1.param_groups[1]["lr"] = _HV["BIAS_LR"] * f2
    opt1.param_groups[2]["lr"] = _HV["HEAD_LR"] * f2
    opt2.param_groups[0]["lr"] = _HV["MUON_LR"] * f2

    opt1.step()
    opt1.zero_grad(set_to_none=True)
    opt2.step()
    opt2.zero_grad(set_to_none=True)
    return opt1, opt2


def _hv_selftest_graph_vs_base_muon(base, model, loader, forward_step, graph, batch, total, wb_steps):
    """Build-time guard: one graph replay must match one eager base.Muon step."""
    import torch

    variants = (
        # (schedule step s, desired do_norm, opt2.step_count before this one step)
        (0, False, 0),
        (0, True, 1),
        (wb_steps, False, 0),
        (wb_steps, True, 1),
    )
    max_param_diff = 0.0
    max_buffer_diff = 0.0
    for s, expected_do_norm, pre_step_count in variants:
        _hv_reset_graph_untrained(model, loader, graph)
        snap = _hv_model_snapshot(model)

        graph.prepare()
        graph.opt2.step_count = pre_step_count
        graph.opt2.last_norm_step = 0
        got_do_norm = graph.step(s, batch[0], batch[1])
        assert got_do_norm is expected_do_norm, (s, expected_do_norm, got_do_norm)
        torch.cuda.synchronize()
        graph_params = [p.detach().clone() for p in snap[0]]
        graph_buffers = [b.clone() for b in snap[1]]

        _hv_model_restore(snap)
        _hv_run_one_eager_base_step(
            base, model, forward_step, batch, total, wb_steps, s,
            opt2_step_count=pre_step_count, opt2_last_norm_step=0)
        torch.cuda.synchronize()

        if graph_params:
            max_param_diff = max(
                max_param_diff,
                max((a - p.detach()).float().abs().max().item() for a, p in zip(graph_params, snap[0])))
        if graph_buffers:
            max_buffer_diff = max(
                max_buffer_diff,
                max((a - b).float().abs().max().item() for a, b in zip(graph_buffers, snap[1])))

    # The graph path is expected to be near bit-identical on A100; the tolerance
    # only guards harmless fp16/cuBLAS rounding drift between base.Muon's padded
    # NS stack and e1.OrthoMomentum's grouped-shape NS refactor. Measured ~2.9e-3
    # on a smoke build; 5e-3 leaves headroom for per-host cuBLAS jitter so an
    # n=200 fan-out doesn't spuriously fail a container build on a benign rounding.
    tol = 5.0e-3
    print(f"DIAG hiverge765-perstep graph-vs-baseMuon one-step maxdiff "
          f"params {max_param_diff:.3e} bufs {max_buffer_diff:.3e}", flush=True)
    assert max_param_diff <= tol and max_buffer_diff <= tol, (
        f"hiverge765 per-step graph selftest failed: param diff={max_param_diff:.3e}, "
        f"buffer diff={max_buffer_diff:.3e}, tol={tol:.1e}")


class _Hiverge765PerstepBase:
    recipe = "runners/recipes/hiverge_base.py"
    plain_eval = True
    prematerialize_aug = False

    def build(self, ctx: BuildContext) -> BuildState:
        import torch
        import torch.nn.functional as F

        from .recipes import e1 as e1mod
        from .recipes import hiverge_base as base

        unknown = set(ctx.overrides) - {"train_epochs"}
        if unknown:
            raise ValueError(f"hiverge765 per-step only supports the 'train_epochs' override; got {unknown}")
        epochs = float(ctx.overrides.get("train_epochs", 7.65))

        model = base.CifarNet().cuda().to(memory_format=torch.channels_last)
        # no-cudagraphs: inductor's own cudagraphs (default under max-autotune)
        # collide with our manual per-step torch.cuda.graph capture ("Cannot
        # prepare for replay during capturing stage"). Same mode e1 uses.
        model.compile(mode="max-autotune-no-cudagraphs")
        loader = base.CifarLoader(ctx.data_path, train=True, batch_size=_HV["BS"], aug=_HV_AUG)
        total = ceil(epochs * len(loader))
        wb_steps = ceil(_HV["WB_EPOCHS"] * len(loader))

        @torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
        def forward_step(inputs, labels, whiten_bias_grad: bool):
            out = model(inputs, whiten_bias_grad=whiten_bias_grad)
            return F.cross_entropy(out, labels, label_smoothing=_HV["LS"], reduction="sum")

        warm = base.CifarLoader(ctx.data_path, train=True, batch_size=_HV["BS"], aug=_HV_AUG)
        warm.images = torch.randn_like(warm.images)
        warm.labels = torch.randint_like(warm.labels, 0, 10)
        with torch.no_grad():
            model.init_whiten(warm.normalize(warm.images[:960]))
        warm_batch = next(iter(warm))

        opt1, opt2 = _hv_make_graph_optimizers(e1mod, model, total)
        graph = _HivergePerStepGraph(model, opt1, opt2, forward_step, total, wb_steps)
        graph.capture(warm_batch)
        _hv_prebuild_proc_cache(base, loader)
        loader.epoch = 2
        selftest_batch = next(iter(loader))
        _hv_selftest_graph_vs_base_muon(base, model, loader, forward_step, graph, selftest_batch, total, wb_steps)

        _hv_reset_graph_untrained(model, loader, graph)
        loader.epoch = 2
        del warm
        torch.cuda.synchronize()

        return BuildState(obj={"model": model, "loader": loader, "graph": graph,
                               "total": total, "wb": wb_steps}, recipe=base)

    def prepare(self, state: BuildState, run: RunContext) -> RunState:
        import torch

        o = state.obj
        model, loader, graph = o["model"], o["loader"], o["graph"]

        with run.charge.section("reset"):
            model.reset()
            model.zero_grad(set_to_none=True)
            loader.epoch = 2
        with run.charge.section("whiten"):
            with torch.no_grad():
                model.init_whiten(loader.normalize(loader.images[:960]))
        with run.charge.section("graph_prepare"):
            graph.prepare()

        step_data = None
        if self.prematerialize_aug:
            with run.charge.section("premat_aug"):
                # loader.epoch==2 preserves fresh epoch-0 flip parity while
                # preventing CifarLoader from rebuilding its proc_images cache.
                step_data = _hv_materialize_augmented_steps(loader, o["total"])

        return RunState(model=model, obj={"step_data": step_data})

    def train(self, state: BuildState, run: RunState, ctx: RunContext) -> None:
        assert run.prepared_token == ctx.trial_id, "train() got an unprepared run"
        o = state.obj
        model, loader, graph, total = o["model"], o["loader"], o["graph"], o["total"]

        model.train()
        if self.prematerialize_aug:
            for s, (inputs, labels) in enumerate(run.obj["step_data"]):
                graph.step(s, inputs, labels)
            return

        step = 0
        for _ in range(ceil(total / len(loader))):
            model.train()
            for inputs, labels in loader:
                graph.step(step, inputs, labels)
                step += 1
                if step >= total:
                    break
            if step >= total:
                break


class Hiverge765PerstepAugEager(_Hiverge765PerstepBase):
    name = "hiverge765_perstep_aug_eager"
    prematerialize_aug = False


class Hiverge765PerstepAugPremat(_Hiverge765PerstepBase):
    name = "hiverge765_perstep_aug_premat"
    prematerialize_aug = True


TARGETS[Hiverge765PerstepAugEager.name] = TargetSpec(Hiverge765PerstepAugEager.name, Hiverge765PerstepAugEager)
TARGETS[Hiverge765PerstepAugPremat.name] = TargetSpec(Hiverge765PerstepAugPremat.name, Hiverge765PerstepAugPremat)
TARGETS[Hiverge765NoCG.name] = TargetSpec(Hiverge765NoCG.name, Hiverge765NoCG)


# ======================================================================
# e5 — "curriculum on plain torch.compile" (auto-cudagraph, NO manual graph)
# ======================================================================
# e5 reaches ~94% with a 28->32 resolution curriculum compiled `max-autotune`
# (inductor auto-cudagraphs the step) and ZERO hand-rolled CUDA graphs. We run it
# two ways, IDENTICAL except res_schedule, to isolate the curriculum WITHIN one
# clean auto-cudagraph recipe (no FullStepGraph, no e1-style surgery):
#   e5         : the shipped 28->32 curriculum
#   e5_full32  : res_schedule forced all-32 — the matched-accuracy control
# aug runs INSIDE timed train (like hiverge), so compute-honest = charged
# reset+whiten prepare + timed loop; the EMA/lookahead tail is gated on a non-None
# ctx, so train() passes _NoValCtx (tail runs, validation is a no-op).
_E5_DEFAULT_HYP = None  # pristine snapshot of e5's hyp, captured on first build


class _E5Base:
    recipe = "runners/recipes/e5.py"
    plain_eval = True
    name = "e5"
    res_schedule = None  # None -> e5's shipped 28->32 curriculum; else override

    def build(self, ctx: BuildContext) -> BuildState:
        global _E5_DEFAULT_HYP
        from .recipes import e5 as e5mod

        if _E5_DEFAULT_HYP is None:
            _E5_DEFAULT_HYP = copy.deepcopy(e5mod.hyp)

        # Reset mutable module state so warm containers can't leak config.
        e5mod.DATA_PATH = ctx.data_path
        e5mod._FWD_CACHE.clear()                     # drop compiled steps from a prior build
        e5mod.hyp.clear()
        e5mod.hyp.update(copy.deepcopy(_E5_DEFAULT_HYP))
        if ctx.overrides:
            e5mod.hyp.update(ctx.overrides)
        if self.res_schedule is not None:
            e5mod.hyp["res_schedule"] = list(self.res_schedule)

        model, extra = e5mod.setup()                 # compile-warm each res + reset (untimed)
        return BuildState(obj={"model": model, "extra": extra}, recipe=e5mod)

    def prepare(self, state: BuildState, run: RunContext) -> RunState:
        e5mod = state.recipe
        model, extra = state.obj["model"], state.obj["extra"]
        opt1, opt2 = extra["optimizers"]
        with run.charge.section("reset+whiten"):
            e5mod.reset_all(model, extra["train_loader"], opt1, opt2)
            extra["train_loader"].epoch = 0          # per-trial flip parity; proc_images cache (built in setup) survives
        return RunState(model=model, obj={"extra": extra})

    def train(self, state: BuildState, run: RunState, ctx: RunContext) -> None:
        assert run.prepared_token == ctx.trial_id, "train() got an unprepared run"
        e5mod = state.recipe
        e5mod.train(run.model, run.obj["extra"], _NoValCtx())  # EMA tail runs; validate is a no-op


class E5Prog(_E5Base):
    name = "e5"                                       # shipped 28->32 curriculum


class E5Full32(_E5Base):
    name = "e5_full32"
    res_schedule = [32] * 9                           # matched-accuracy control: same recipe, no curriculum


class E5Downsample(_E5Base):
    """e5 with e1's low-res input mechanism: each epoch builds standard full-32px
    translated crops (±2, flip, jitter — identical aug), then bilinearly resizes the
    whole batch tensor to the epoch's res. The shipped e5 instead random-crops a
    res-px WINDOW of the image (smaller field of view, native scale). Same recipe,
    same schedule semantics — isolates crop-vs-downsample for low-res accuracy.
    Implemented by swapping CifarLoader.__iter__ for a copy whose only change is
    crop-at-32 + interpolate-to-res (class-level patch; one target per container)."""
    name = "e5_ds"

    def build(self, ctx: BuildContext) -> BuildState:
        import torch
        import torch.nn.functional as _F

        from .recipes import e5 as e5mod

        def _iter_downsample(self):
            full = self.images.shape[-2]
            if not self.proc_images:
                images = self.proc_images["norm"] = self.normalize(self.images)
                if self.aug.get("flip", False):
                    images = self.proc_images["flip"] = e5mod.batch_flip_lr(images)
                pad = self.aug.get("translate", 0)
                if pad > 0:
                    self.proc_images["pad"] = _F.pad(images, (pad,) * 4, "reflect")
            if self.aug.get("translate", 0) > 0:
                images = e5mod.batch_crop(self.proc_images["pad"], full)  # always full-res crop
            elif self.aug.get("flip", False):
                images = self.proc_images["flip"]
            else:
                images = self.proc_images["norm"]
            if self.aug.get("flip", False):
                if self.epoch % 2 == 1:
                    images = images.flip(-1)
            cj = self.aug.get("color_jitter", {"enabled": False})
            if cj.get("enabled", False):
                images = e5mod.batch_color_jitter(
                    images, cj.get("brightness_range", 0.1), cj.get("contrast_range", 0.1))
            if self.crop_size != full:  # THE change: downsample, don't window-crop
                images = _F.interpolate(images, size=(self.crop_size,) * 2,
                                        mode="bilinear", align_corners=False)
                images = images.contiguous(memory_format=torch.channels_last)
            self.epoch += 1
            if self.shuffle:
                torch.randperm(len(self._indices), out=self._indices)
                indices = self._indices
            else:
                indices = torch.arange(len(self.images), device=self.images.device)
            for i in range(len(self)):
                idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
                yield (images[idxs], self.labels[idxs])

        e5mod.CifarLoader.__iter__ = _iter_downsample
        return super().build(ctx)


TARGETS[E5Prog.name] = TargetSpec(E5Prog.name, E5Prog)
TARGETS[E5Full32.name] = TargetSpec(E5Full32.name, E5Full32)
TARGETS[E5Downsample.name] = TargetSpec(E5Downsample.name, E5Downsample)
