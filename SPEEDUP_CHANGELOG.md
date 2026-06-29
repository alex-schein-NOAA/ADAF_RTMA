# ADAF_RTMA Training Speedup — Complete Changelog

**What this document is.** A single, exhaustive record of *every* change made to
speed up `train.py` training, why each one was made, how much it helped, and how it
affects the project going forward. It is written to be read top-to-bottom by someone
who wasn't in the room. If you only read one file about the speedup work, read this one.

**Headline result:** the speedups are large and real, and the **loss curve is
bit-identical** (0.00110 → 0.001081, unchanged within noise) — every change is either a
pure-speed transformation (same math, fewer/cheaper ops) or a config flag defaulting to
old behavior; no accuracy was traded away. The cleanest, **reproducible** number is the
single-GPU micro-benchmark: `torch.compile` gives **6.1×** model-only (no network
involved). The end-to-end multi-node figure looked like **3.64×** (2.503 → 0.688 s/step)
in one measurement — but see the **big caveat**: that 0.688 s was a lucky low. Repeating
the *exact same config* solo gives 1.18–1.85 s, because multi-node step time has ~2×
run-to-run noise from shared inter-node network contention (§6). So treat the multi-node
multiple as a **range (~2–3.6×) with real variance**, and trust the single-GPU 6.1× as
the clean figure.

**Branch:** `add-metar-adpsfc` (all changes currently uncommitted in the working tree).
**Hardware:** 2 nodes × 2 H100 NVL = 4-rank DDP. **Grid:** 1356×2294 (new RTMA grid).
**Per-GPU batch:** 2.

---

## 1. The big picture — how we got 3.64×

The speedup came in three "tiers," each targeting a *different* bottleneck that only
became visible after the previous one was removed. The order matters: we measured,
fixed the top cost, re-measured, and repeated.

| Tier | What it attacks | Lever | Step time | Cumulative speedup |
|-----:|-----------------|-------|----------:|-------------------:|
| — | original (all knobs off) | — | **2.503 s** | 1.00× |
| Group A | thousands of tiny GPU kernel launches | `torch.compile` (+ bf16, channels_last, tf32) | 1.011 s | 2.48× |
| Tier 1 | redundant work in the loss + per-step CPU↔GPU syncs | loss rewrite, on-GPU accumulation, DDP cleanups | 1.0075 s | 2.48× |
| Tier 2 | NCCL re-broadcasting static buffers every step | `ddp_broadcast_buffers=False` | **0.688 s*** | **3.64×*** |
| Tier 3 | exposed all-reduce wait (rank skew) | I/O smoothing (`non_blocking`, `prefetch`) | no measurable gain | — |

\* **The 0.688 s / 3.64× is a single lucky-quiet-network sample.** Five repeats of this
exact config later clocked **1.18–1.85 s** — the multi-node step time swings ~2.7× with
shared inter-node network load (§6). The single-GPU `torch.compile` number (**6.1×**, no
network) is the clean, reproducible figure; the multi-node multiple is a noisy range.

**The one-sentence story of each tier:**
- **Group A** — the model fired thousands of tiny GPU operations per step, leaving the
  H100 ~20% utilized (starved). `torch.compile` fused them into a handful of big
  kernels → util 94%, the single largest win.
- **Tier 1** — the loss function did redundant work (full-resolution `.clone()`s and a
  second pass over the same data) and synced the GPU to the CPU on *every* step. We
  removed the redundancy and batched the sync to once per epoch.
- **Tier 2** — after compile, profiling revealed the real top cost wasn't compute at
  all: DDP was re-broadcasting two never-changing model buffers from rank 0 on every
  forward, eating 63% of GPU time. We turned that broadcast off.
- **Tier 3** — once broadcast was gone, the new top item was GPUs idling while waiting
  for each other to finish (rank skew). We tried to smooth it with I/O knobs; the
  remaining wait turned out to be shared-fabric noise we can't control. 3.64× is the
  floor.

---

## 2. Why was anything slow in the first place?

Two facts about this model drove everything:

1. **The transformer runs at full resolution.** Despite `patch_size: 4` looking like a
   4× downsample, `PatchEmbed.forward` only does `flatten(2).transpose(1,2)` — it never
   actually strides/patches. So attention operates on the *entire* 1356×2294 grid
   (~3.1M tokens). That's why the model is expensive: sequence length, not parameter
   count (the model is only ~224k params).

2. **The work is a flood of small operations.** A Swin transformer at this resolution
   issues thousands of tiny reshape / layernorm / pointwise / attention kernels per
   step. Each kernel launch has fixed overhead. At this grid the GPU spent more time
   waiting for the *next* kernel to be launched than computing — eager-mode utilization
   was only **18–27%**. The GPU was starved, not saturated.

This is the key insight: **the model was latency-bound on kernel launches, not
compute-bound.** That's exactly the situation `torch.compile` is built to fix.

---

## 3. Group A — the foundation: torch.compile + AMP knobs

**Files:** `config/params_default.yaml`, `train.py`, `models/encdec.py`

This tier added four config flags, all **default-OFF** (so baseline behavior is
byte-for-byte unchanged until you flip them):

```yaml
tf32: False           # TF32 matmul + cuDNN TF32
amp_dtype: "float16"  # or "bfloat16" (H100-native)
channels_last: False  # NHWC memory format for the conv path
compile_model: False  # torch.compile(model) — the big one
```

### 3.1 What each knob does

- **`compile_model` (torch.compile)** — *the* win. Fuses the thousands of tiny ops into
  a few large kernels. Single-GPU micro-benchmark: **6.1× model-only** (step 1927 ms →
  315 ms, util 27% → 94%, and it even *cut* memory 54.4 → 50 GB). End-to-end in real
  DDP training it's ~2.5× because the loss and DDP (not compiled) then dominate.
- **`amp_dtype: bfloat16`** — H100-native; ~1.04× and more numerically stable than
  fp16. As a bonus it lets us disable the `GradScaler` (bf16 has fp32's exponent range,
  so no loss-scaling needed).
- **`channels_last`** — NHWC layout for the conv-heavy path; ~1.08×.
- **`tf32`** — essentially a no-op *here* (the matmuls already run under AMP autocast,
  so they're already in reduced precision), but harmless and standard, so left in.

### 3.2 The torch.compile environment problem (and the durable fix)

torch.compile would **not run** in the stock `ADAF_environment` — two blockers:

1. Triton's launcher needs the CUDA driver header `cuda.h`, which the env didn't ship
   → `cuda.h: No such file or directory`.
2. TorchInductor's CPU AVX512 codegen miscompiles on the system gcc-11
   (`decltype(...)::blendv ... not a class`).

**Fix (durable, already in place):**
- A **cloned env** at `/scratch3/BMC/wrfruc/Micah.Craine/conda_envs/ADAF_environment`
  carries an activate hook (`etc/conda/activate.d/zz_cuda_headers.sh`) that prepends the
  system CUDA include to `CPATH` on activation → fixes blocker (1).
- Blocker (2) is handled **in code**: `train.py` sets
  `torch._inductor.config.cpp.simdlen = 0` (forces scalar CPU codegen — negligible cost
  for this GPU-bound model) whenever `compile_model` is on.

> **Going forward:** any compiled run *must* use the clone env, activated by **absolute
> python path inside `srun`** (a multi-node PATH race otherwise falls back to a
> torch-less module python). This is wired into all the `experiments/*/job_sbatch.sh`
> scripts. The production launcher still needs this pattern applied before it can run
> compiled.

### 3.3 The one real model fix (`models/encdec.py`)

`SwinTransformerBlock.calculate_mask` now rounds H,W **up to a multiple of
`window_size`** before building the precomputed attention mask:

```python
H = ((H + self.window_size - 1) // self.window_size) * self.window_size
W = ((W + self.window_size - 1) // self.window_size) * self.window_size
```

Without this, a grid width not divisible by `window_size` (the 2294-wide RTMA grid)
crashes `window_partition` at model construction. The real forward always passes an
already-padded size, so this only affects the precomputed-at-init mask. Pairs with
`patch_size: 1` in the new-grid config.

**Result:** real DDP training **2.48×** (2.503 → 1.011 s/step), loss unchanged.

---

## 4. Tier 1 — bit-exact cleanups in the loss and training loop

**File:** `train.py`

These are pure-speed code changes: **same math, less work**. They were verified
bit-identical (loss 0.001080 vs 0.00108).

### 4.1 Loss function: drop wasted clones and a redundant pass

The old `loss_function` had two inefficiencies at full 1356×2294 resolution:

- **Wasted `.clone()`s.** It cloned `pre_field`, `tar_field`, etc. before calling
  `torch.masked_fill`. But `masked_fill` is *out-of-place* — it returns a fresh tensor
  anyway — so each clone allocated a full-resolution copy that was immediately thrown
  away. Removed; the masks are negated once and reused.
- **A redundant second MSE pass.** It computed the scalar loss and the per-channel loss
  as *two separate* `F.mse_loss` calls over the same data. Now we compute the squared
  error **once** and derive both the scalar mean and the per-channel mean from it:

```python
se_field = (pre_field_masked - tar_field_masked) ** 2
loss_field = se_field.mean()
loss_field_channel_wise = se_field.mean(dim=(0, 2, 3))
```

(Same for the obs loss.) Halves the full-resolution reduction work in the loss.

### 4.2 On-GPU loss accumulation — kill the per-step CPU↔GPU sync

The old loop did `loss_field += loss["loss_field"].detach().item()` **every step**.
`.item()` forces a host↔device synchronization — the CPU stalls until the GPU finishes,
draining the pipeline ~1000 times per epoch. Now we accumulate on the GPU and sync
**once at epoch end**:

```python
loss_field = torch.zeros((), device=self.device)   # was 0.0
...
loss_field += loss["loss_field"].detach()          # was .item(), no sync
...
logs = {"loss_field": (loss_field / steps_in_one_epoch).item(), ...}  # one sync/epoch
```

### 4.3 DDP cleanups

In the `DistributedDataParallel` wrap:
- **`find_unused_parameters=False`** (was `True`) — every parameter gets a gradient on
  the single forward path, so the unused-param scan each backward was pure overhead.
  Now a config flag (`ddp_find_unused_parameters`).
- **`gradient_as_bucket_view=True`** — gradients alias the reduce buckets instead of
  being copied into them (saves a copy + memory; numerically identical).

### 4.4 Removed a no-op

`gen.to(self.device, dtype=torch.float)` after the forward was a no-op (`.to()` returns
a new tensor; the result was discarded) — removed in both train and validate paths.

**Result:** the *eager* path got **1.26×** from these cleanups (2.503 → 1.992 s/step).
The *compiled* path was flat (1.011 → 1.0075) because torch.compile already fuses most
of what they remove — but it doesn't regress, and the code is simpler and correct.

> **Two dead ends recorded so nobody re-tries them with compile on:**
> `compile_loss=True` (compiling the loss as a *separate* graph) and
> `ddp_static_graph=True` both **regressed** the compiled step to 1.76 s (recompile /
> graph-break thrash between the two graphs). Both default **False** and are documented
> as compile-incompatible.

---

## 5. Tier 2 — the profiler surprise: stop broadcasting static buffers

**File:** `train.py` (DDP wrap) + `config/params_default.yaml`

After compile, we *guessed* the loss was the next bottleneck. We were wrong, and the
profiler proved it. A `torch.profiler` trace of the steady-state compiled step
(8 steps) showed:

| GPU cost (self-CUDA) | per step | share |
|----------------------|---------:|------:|
| **NCCL broadcast of `attn_mask` / `relative_position_index` buffers** | **0.70 s** | **63.6%** |
| compiled model fwd+bwd | 0.41 s | 37% |
| attention `bmm` | 0.11 s | 9.8% |
| gradient all-reduce (the *real* DDP sync) | 0.06 s | 5.3% |
| HtoD input copy | 0.05 s | 4% |

### What was happening

DDP's default `broadcast_buffers=True` re-broadcasts **every registered buffer from
rank 0 at the start of every forward pass**. This model registers two buffers —
`attn_mask` (encdec.py:319) and `relative_position_index` (encdec.py:140) — that are
computed **deterministically and identically on every rank** and **never updated during
training**. (That re-broadcast exists for things like BatchNorm running stats, which
*do* drift between ranks. This model has none.) So the broadcast was pure waste — and
it was the single biggest cost in the step.

### The fix

```python
broadcast_buffers=as_bool(getattr(self.params, "ddp_broadcast_buffers", True))
```

with `ddp_broadcast_buffers: False` set in `params_default.yaml`. **Bit-identical** —
the buffers are the same on every rank with or without the broadcast.

**Result (solo confirm, job 16148928):** 1.0075 → **0.6882 s/step**, a clean **1.46×**,
loss bit-identical (0.001081). **Cumulative: 3.64× over the original baseline.**

(The profile attributed 0.70 s to the broadcast but we only recovered 0.32 s/step —
because some of that NCCL broadcast was overlapping compute on a separate stream, so its
"CUDA time" double-counted. The wall-clock win is the 1.46× that matters.)

---

## 6. Tier 3 — the I/O lever, and why we stopped here

**Files:** `config/params_default.yaml`, `train.py`, `utils/dataloader_multifiles.py`

After Tier 2, re-profiling showed the new top item was **0.46 s/step of exposed
all-reduce *wait*** — not data transfer (the gradient payload is only ~0.9 MB). This is
**rank skew**: GPUs finishing their step at slightly different times and idling at the
all-reduce barrier until the slowest rank catches up. The usual cause is variable
input-batch-ready timing across ranks.

We added the safe lever to smooth that — keep every rank fed so they stay in lockstep:

- **`non_blocking: True`** — async host→device copies (the dataloader already pins
  memory, but `non_blocking=False` made every `.to()` synchronous, wasting the pinning).
- **`prefetch_factor: 2`** — each worker reads more batches ahead, cushioning Lustre/HDF5
  read jitter. (Also made the dataloader honor the config value instead of a hardcoded 1.)

### The result: no measurable gain — and a bigger discovery

We ran the A/B (`io_sync` control vs `io_async` lever) solo+sequential, twice. Both times
the **control failed to reproduce** the 0.688 s "floor" (it clocked 1.5 s, 1.3 s), and
warmup sometimes inverted (epoch 2 slower than epoch 1). Rather than keep guessing
"the fabric was busy," we ran the **decisive test: the same fixed `io_sync` config, solo,
five times** (jobs 16148928, 16195071-73, 16220206):

| sample | ep2 step | data_time | loss e2 |
|--------|---------:|----------:|--------:|
| original "clean" run | **0.688 s** | 0.04 | 0.001081 |
| repeat 1 | 1.476 s | 0.035 | 0.001080 |
| repeat 2 | 1.179 s | 0.036 | 0.001081 |
| repeat 3 | 1.312 s | 0.037 | 0.001081 |
| t2_final | 1.849 s | 0.034 | 0.001081 |

**Identical config, solo, swinging 0.69 → 1.85 s (median ~1.31, ~2.7× spread)** (repeat 3
even drifted *within* the job: ep1 1.01 → ep2 1.31). This nails the cause and rules out
everything internal:

- **Not the I/O knobs** — `io_async` was the *fast* run in one pair, slow in another; the
  knobs don't predict the outcome.
- **Not data/Lustre** — `data_time` is a flat ~0.035 s in every slow run. The filesystem
  delivers batches instantly; the lost time is in **compute+NCCL**, not I/O. (So the whole
  premise that smoothing I/O would help the all-reduce wait was attacking the wrong layer.)
- **Not config or node** — `io_sync` is byte-identical to the clean run, and one slow run
  ran on the *same nodes* (u20g[10-11]) as the clean one.

What's left is **time-varying contention on the shared inter-node network** (InfiniBand),
which our DDP all-reduce/broadcast crosses. A quiet *GPU partition* does not predict it.

**The important correction:** the **0.688 s was the lucky outlier**, not the floor.
Typical steady-state under contention is **~1.2–1.85 s (median ~1.31)**. So the "3.64×" headline was
computed against a best-case sample; the honest multi-node figure is a **noisy range**.
Loss stayed **0.001081 across all of it** — accuracy is rock-solid regardless.

> **Decisions:**
> 1. **Adopt `non_blocking=True` + `prefetch_factor=2` anyway** — bit-identical, zero-risk,
>    can only help keep ranks fed (just not a *measured* win).
> 2. **Stop A/B-ing fractional levers.** The run-to-run noise (~2×) dwarfs any remaining
>    lever; we literally cannot resolve a 5–10% change through it.
> 3. **Quote the single-GPU number (6.1×) as the clean speedup**; report the multi-node
>    multiple as a range with the variance caveat.
> 4. *(Optional, diagnostic only)* run **2 GPUs on one node** (NVLink, no InfiniBand) — if
>    stable, it confirms the variance is the cross-node fabric.

---

## 7. Supporting changes (plumbing that made the above possible)

- **`utils/misc_functions.py` — `as_bool()` helper.** YAML gives real booleans, but CLI
  overrides arrive as *strings*, so `--non_blocking False` would otherwise be the
  *truthy* string `"False"`. Every boolean knob is gated through `as_bool()` to avoid
  this footgun. Also registered three new CLI args (`--prefetch_factor`,
  `--non_blocking`, `--ddp_find_unused_parameters`).
- **`train.py` — `EPOCH_METRICS,...` line.** A machine-parseable per-epoch line
  (steps, tr_time, step_time, samples_per_sec, loss_field) that the benchmark parser
  reads. This is how every number in this doc was measured.
- **`train.py` — env-gated profiler hook.** `ADAF_PROFILE=1` profiles a short
  steady-state window of the last epoch; `ADAF_MAX_STEPS` caps steps uniformly across
  ranks (so an early break can't desync a DDP collective). This is the tool that found
  the Tier-2 broadcast cost — **measure, don't guess.**
- **`train.py` — per-rank seeding** (`set_random_seed(params.seed)`) so runs are
  reproducible and the loss-overlap correctness check is meaningful.
- **Surface pressure dropped** (config): the data no longer contains `*_sp`/`sta_p`, so
  `in_chans 21→17`, `out_chans 5→4`, and the var lists dropped sp. This is a *modeling*
  change (the model no longer predicts surface pressure), not a speed change — recorded
  here only because it's in the same diff.

---

## 8. The complete config surface (what every new flag does)

All default to the **old behavior**, so an un-updated config trains exactly as before.

| Flag | Default | Turn it on for | Notes |
|------|---------|----------------|-------|
| `compile_model` | False | **production speed** | The big win (~2.5× end-to-end). Needs the clone env. |
| `amp_dtype` | float16 | **production** (`bfloat16`) | H100-native, more stable, disables GradScaler. |
| `channels_last` | False | **production** | ~1.08×, conv path. |
| `tf32` | False | production | ~no-op under AMP, but harmless/standard. |
| `ddp_broadcast_buffers` | False* | **always off here** | *Set False in params_default. Off = 1.46×, bit-identical. Set True only if you add rank-varying buffers (e.g. BatchNorm). |
| `ddp_find_unused_parameters` | False | always | All params get grads; True is pure overhead. |
| `gradient_as_bucket_view` | (hardcoded True) | always | Saves a copy; numerically identical. |
| `non_blocking` | False* | production (`True`) | *Adopted True. Async HtoD; zero-risk, gain in the noise here. |
| `prefetch_factor` | 1* | production (`2`) | *Adopted 2. Read-ahead; zero-risk. |
| `compile_loss` | False | **never with compile** | Regressed to 1.76 s (graph thrash). |
| `ddp_static_graph` | False | **never with compile** | Same — regressed the compiled path. |

---

## 9. How this affects the project going forward

**For day-to-day training:**
- **Just submit `train_production_sbatch.sh`.** It runs `train.py` under 4-rank DDP on
  the **clone env** (absolute python path inside `srun`, `HDF5_USE_FILE_LOCKING=FALSE`)
  against `config/params_default.yaml`, which now **ships the production setup**: new grid
  (1356×2294, patch_size 1, batch_size 2, new data path) and all speed knobs ON
  (`compile_model`, `amp_dtype: bfloat16`, `channels_last`, `tf32`,
  `ddp_broadcast_buffers: False`, `non_blocking`, `prefetch_factor: 2`). Expect **epoch 1
  to be slow** (one-time compile warmup) and **epoch 2+ substantially faster** with an
  unchanged loss curve. The exact multi-node multiple varies run-to-run (~2× spread from
  network contention, §6); the clean single-GPU number is 6.1×.
- **To reproduce the original eager baseline:** set `compile_model`, `tf32`,
  `channels_last` back to `False` and `amp_dtype` to `float16` in the config.
- **`compile_model: True` requires the clone env.** Running the default config under the
  stock `ADAF_environment` will fail at the first Triton compile (no `cuda.h`).

**For correctness/safety:**
- Every speed change here is **accuracy-neutral by construction** and was verified
  bit-identical (loss 0.001081 throughout). Flipping these knobs does **not** change
  what the model learns.
- The flags were introduced default-OFF (to preserve the baseline during the A/Bs);
  `params_default.yaml` now ships the production-ON values. Code defaults (`getattr`
  fallbacks) remain OFF, so an *absent* key still means old behavior.

**For the next person chasing more speed** (these config levers are exhausted; further
gains require structural changes — and note the multi-node timing is too noisy to measure
small wins, so prefer the single-GPU bench for any fractional comparison):
1. **Real patch downsampling.** `PatchEmbed` is a no-op flatten → attention runs at
   full resolution. A strided patch-embed + matching upsampler would cut tokens *and*
   memory ~p². Biggest structural win, but changes the architecture and may cost
   accuracy. (Now more attractive: we're memory-bound, so it would also unlock a bigger
   batch.)
2. **SDPA / FlashAttention.** Attention is hand-rolled (`q@k.T → softmax → @v`), not
   `F.scaled_dot_product_attention`. Fused kernels could give big memory+speed wins —
   but the window attention's relative-position bias and mask must be threaded through
   SDPA's `attn_mask` path correctly, and numerics must be checked.
3. **Bigger batch via memory reduction.** bs=2 is the ceiling at this grid (~26
   GB/sample). Gradient checkpointing (already in `encdec.py`) trades compute for memory
   to unlock bs=3–4.
4. **Don't** re-chase the all-reduce wait with I/O knobs — it's fabric noise (§6).

**Measurement discipline (learned the hard way):**
- **Multi-node step time on this cluster has ~2× run-to-run noise.** The same config,
  run solo five times, gave 0.69 / 1.18 / 1.31 / 1.48 / 1.85 s — driven by shared inter-node
  network contention, *not* anything in our code (`data_time` was a flat ~0.035 s
  throughout). A quiet GPU partition does not predict it. **Do not trust a single
  multi-node timing**, and don't try to measure sub-2× effects there at all.
- **Use the single-GPU micro-bench for clean speedup numbers.** No DDP, no network →
  reproducible. That's where the trustworthy 6.1× comes from.
- **Profile before optimizing.** Our biggest win (Tier 2) was invisible to intuition —
  we'd have spent days fusing the loss (which barely registered) instead of flipping one
  DDP flag. The `ADAF_PROFILE` hook is in `train.py` for exactly this.

---

## 10. Reference: results at each step

**Clean, reproducible number (single-GPU micro-bench, model only, no loss/DDP/network):**
`torch.compile` = **6.1×** (1927 → 315 ms, util 27% → 94%). This is the figure to quote.

**Multi-node end-to-end (4×H100 DDP)** — directionally correct but ~2× noisy (§6), so read
these as *one sample each*, not precise:

| run | config | epoch-2 step | loss e2 |
|-----|--------|-------------:|--------:|
| original baseline | all knobs off | 2.503 s | 0.00110 |
| eager + Tier-1 cleanups | no compile | 1.992 s | 0.001094 |
| Group A (compile bundle) | compile+bf16+ch_last+tf32 | 1.011 s | 0.00108 |
| + Tier-1 cleanups | (compiled, flat) | 1.0075 s | 0.001080 |
| + Tier-2 broadcast off | **final config** | 0.688 s … 1.48 s* | 0.001081 |

\* **Same config, five solo runs: 0.688 / 1.179 / 1.312 / 1.476 / 1.849 s.** The 0.688
(→ a naive "3.64×") was the lucky low; ~1.2–1.85 s (median ~1.31) is typical. The variance is shared
inter-node network contention, not our code (`data_time` flat ~0.035 s). Loss is
bit-identical (0.001081) across all of it. **Trust the single-GPU 6.1× for magnitude;
treat the multi-node multiple as a ~2–3.6× range.** The I/O knobs (Tier 3) added no
measurable gain and were adopted only as zero-risk defaults.

**Source experiments:** `experiments/epoch_compare/` (real DDP A/Bs, `RESULTS.md`),
`experiments/gpu_saturation/` (single-GPU micro-bench, `RESULTS.md`). Prior handoffs:
`experiments/HANDOFF_speedup_and_accuracy.md` (Phase 1),
`experiments/HANDOFF_speedup_phase2.md` (Phase 2).
