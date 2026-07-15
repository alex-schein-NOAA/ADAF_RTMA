import os
import time
# import wandb
import random
import datetime
import argparse
import contextlib
import numpy as np

# from str2bool import str2bool
# from icecream import ic
from shutil import copyfile
# from apex import optimizers
from collections import OrderedDict

import torch
# import torch.cuda.amp as amp
import torch.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from utils.dataloader_multifiles import get_data_loader
# from utils.logging_utils import log_to_file
from utils.YParams import YParams
from utils.misc_functions import set_user_params, as_bool

#################################

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:
    def count_parameters(self):
        count_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                count_params += p.numel()
        return count_params
    
    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

    def __init__(self, params):
        self.params = params
        # self.set_device() #Should this be here when we set the device just below?

        # Set up local node
        torch.cuda.set_device(self.params.local_rank)
        self.device = torch.device("cuda", self.params.local_rank)
        print(f"world_rank: {self.params.world_rank} | local_rank: {self.params.local_rank} | device: {self.device} | num_data_workers={self.params.num_data_workers}")

        # --- low-risk throughput knobs (defaults below reproduce the baseline exactly) ---
        if as_bool(getattr(self.params, "tf32", False)):
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        _amp_dtype = str(getattr(self.params, "amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if _amp_dtype in ("bf16", "bfloat16") else torch.float16
        self.channels_last = as_bool(getattr(self.params, "channels_last", False))

        # Load model
        from models.encdec import build_model as model # dispatches on params.arch: flat SwinIR (default) or the 1/4-res lowres body
        self.model = model(self.params).to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # Load training and validation data
        print(f"[world_rank: {self.params.world_rank}] Begin data loading \n") #may need to be changed to rank 0 only
        (self.train_data_loader, self.train_dataset, self.train_sampler) = get_data_loader(self.params,
                                                                                           self.params.train_data_path,
                                                                                           dist.is_initialized(),
                                                                                           train=True)
        
        # train=False for the valid split: it selects the deterministic fixed-rate obs
        # dropout (comparable across epochs/runs) instead of the randomized training
        # regime, and an unshuffled sampler.
        (self.valid_data_loader, self.valid_dataset, self.valid_sampler) = get_data_loader(self.params,
                                                                                           self.params.valid_data_path,
                                                                                           dist.is_initialized(),
                                                                                           train=False)
        print(f"[world_rank: {self.params.world_rank}] Data loaded \n") #may need to be changed to rank 0 only or removed

        # Set up optimizer
        if self.params.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        elif self.params.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr)
        else:
            raise Exception("Only Adam and AdamW optimizers implemented")
        
        # GradScaler is only meaningful for fp16; for bf16 it is created disabled
        # (scale/step/update become pass-throughs, so the branch logic below is unchanged).
        if self.params.enable_amp:
            self.gscaler = amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        # Set up distributed training
        if dist.is_initialized():
            # gradient_as_bucket_view: grads alias the reduce buckets (saves a copy +
            # memory; numerically identical). static_graph: opt-in -- lets DDP assume a
            # fixed graph each step for better all-reduce/backward overlap, valid here
            # because every step takes the same path and all params get grads.
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.params.local_rank],
                                                 output_device=[self.params.local_rank],
                                                 find_unused_parameters=as_bool(self.params.ddp_find_unused_parameters),
                                                 gradient_as_bucket_view=True,
                                                 # broadcast_buffers default True re-broadcasts attn_mask/relative_position_index
                                                 # from rank 0 every forward. Those buffers are computed deterministically and
                                                 # identically on every rank (no BN running stats) -> the broadcast is pure waste
                                                 # and was 63% of GPU time in the profile. Disabling is bit-identical.
                                                 broadcast_buffers=as_bool(getattr(self.params, "ddp_broadcast_buffers", True)),
                                                 static_graph=as_bool(getattr(self.params, "ddp_static_graph", False)))
        # Post-Local-SGD: capture the DDP module BEFORE torch.compile wraps it, so we can
        # call no_sync() to skip the per-step gradient all-reduce during the local phase
        # (the compiled wrapper doesn't expose no_sync cleanly). For the first
        # ADAF_LOCALSGD_WARMUP steps we train with normal every-step DDP; after that each
        # rank steps on its own local gradients and we average the model *weights* across
        # ranks every ADAF_LOCALSGD_H steps. H=0 disables it (default -> plain DDP). This
        # touches the shared inter-node fabric H x less often, trading exact-SGD
        # equivalence for far fewer cross-node collectives.
        self._ddp_module = self.model if dist.is_initialized() else None
        self._localsgd_h = max(0, int(os.environ.get("ADAF_LOCALSGD_H", "0") or 0))
        self._localsgd_warmup = max(0, int(os.environ.get("ADAF_LOCALSGD_WARMUP", "0") or 0))
        self._loss_fn = self.loss_function
        # Global grad-norm clip. 0 (default) = off, so the flat-model runs are unchanged.
        # A guard against a loss blow-up 12 h into a 25 h run, useful for the deeper
        # lowres body; there is no other protection in this loop.
        self._grad_clip = float(getattr(self.params, "grad_clip", 0) or 0)
        # Linear LR warmup over the first N optimizer steps. 0 (default) = off, so every
        # run before this one is unchanged. A transformer trained from scratch takes its
        # largest, worst-conditioned steps in the first few hundred iterations, before
        # Adam's moment estimates have settled.
        self._warmup_iters = int(getattr(self.params, "warmup_iters", 0) or 0)
        if as_bool(getattr(self.params, "compile_model", False)):
            # Inductor's CPU-side AVX512 codegen miscompiles on the system gcc-11
            # ("decltype(...)::blendv ... not a class"); force scalar CPU codegen
            # (those CPU glue kernels are negligible for this GPU-bound model).
            # Requires cuda.h on CPATH for the Triton launcher -- the cloned env's
            # activate.d hook provides it. Measured ~2x at batch_size=12.
            import torch._inductor.config as _ind
            _ind.cpp.simdlen = 0
            self.model = torch.compile(self.model)
        # compile_loss: OFF by default. Compiling the loss as a separate graph regressed
        # the fast step in the epoch_compare A/B (1.01 -> 1.76 s/step, ep2 slower than ep1
        # -- recompile/graph-break thrash between the model and loss graphs). Left as an
        # opt-in flag for future fusing-into-the-model-graph work, not the separate compile.
        if as_bool(getattr(self.params, "compile_loss", False)):
            import torch._inductor.config as _ind
            _ind.cpp.simdlen = 0
            self._loss_fn = torch.compile(self.loss_function)
        # Per-variable loss weights, in target_vars order (repo config: q,t,u10,v10).
        # Renormalized to mean 1 so the gradient scale -- and therefore the usable LR --
        # is unchanged by reweighting. Default [1,1,1,1] = the historical uniform-pixel
        # loss, whose *effective* per-variable contributions under min-max normalization
        # are t 1.0 / q 2.64 / u 0.31 / v 0.42 (measured; see grid_distribution_analysis).
        w = getattr(self.params, "loss_channel_weights", None)
        if w is None:
            w = [1.0] * len(self.params.target_vars)
        w = torch.tensor([float(x) for x in w], dtype=torch.float32, device=self.device)
        if len(w) != len(self.params.target_vars):
            raise ValueError(f"loss_channel_weights has {len(w)} entries, "
                             f"expected {len(self.params.target_vars)} (target_vars order)")
        self.loss_weights = (w / w.mean()).view(1, -1, 1, 1)

        self.iters = 0
        self.startEpoch = 0
        # Best held-out metric seen so far -- persisted in the checkpoint and restored on
        # resume, so best_ckpt is not clobbered by the first (possibly worse) resumed epoch.
        self.best_valid_metric = 1.0e6
        #plotting stuff left out for now

        # Set up dynamical learning rate, if requested
        if self.params.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        factor=self.params.lr_reduce_factor,
                                                                        patience=self.params.scheduler_patience,
                                                                        mode="min")
        elif self.params.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.params.max_epochs,
                                                                        last_epoch=self.startEpoch - 1)
        else:
            self.scheduler = None

        # %% Resume train
        if self.params.resuming:
            if self.params.log_to_screen and self.params.world_rank==0:
                print(f"Loading checkpoint from {self.params.best_checkpoint_path}")
            self.restore_checkpoint(self.params.best_checkpoint_path)
        
        self.epoch = self.startEpoch

        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print(f"Number of trainable model parameters: {self.count_parameters()}")
            if self._localsgd_h > 0:
                print(f"Post-Local-SGD: averaging weights every {self._localsgd_h} steps "
                      f"after {self._localsgd_warmup} warmup steps")

    ##########
    
    def loss_function(self,
                      pre_field,
                      tar_field,
                      tar_obs,
                      tar_field_obs,
                      field_mask=None,
                      obs_tar_mask=None,
                      mask_out_of_range=True):
        """
        pre_field: model's output
        tar_field: label, after normalization
        """
        
        # (2026-06-05) note these are still attached, i.e. differentiable for gradient flow.
        # (2026-06-27) masked_fill is out-of-place (returns a fresh tensor), so the earlier
        # .clone() of each tensor only allocated a full-res copy that was immediately thrown
        # away by the masked_fill below -- dropped. Negate each mask once and reuse.
        if mask_out_of_range: # fill input with 0 where field_mask is False
            not_field_mask = ~field_mask
            pre_field_masked = pre_field.masked_fill(not_field_mask, 0)
            tar_field_masked = tar_field.masked_fill(not_field_mask, 0)
            tar_field_obs_masked = tar_field_obs.masked_fill(not_field_mask, 0)
        else:
            pre_field_masked = pre_field
            tar_field_masked = tar_field
            tar_field_obs_masked = tar_field_obs

        # Per-channel weights (mean 1), kept in fp32 -- the squared errors below are
        # already fp32 (bf16 prediction minus fp32 target promotes). With the default
        # [1,1,1,1] this is an exact no-op: the weighted mean reduces to the plain mean.
        wts = self.loss_weights

        # type 1 loss -- compute the squared error once and derive both the scalar mean
        # and the per-channel mean from it (was two separate full-res mse passes).
        se_field = (pre_field_masked - tar_field_masked) ** 2
        loss_field = (se_field * wts).mean()
        loss_field_channel_wise = se_field.mean(dim=(0, 2, 3))  # unweighted: a diagnostic

        # type 2 loss -- the one that is actually backwarded (target: analysis_obs)
        se_field_obs = (pre_field_masked - tar_field_obs_masked) ** 2
        loss_field_obs = (se_field_obs * wts).mean()

        # type 3 loss - use fresh masks for obs loss
        not_obs_tar_mask = ~obs_tar_mask  # fill input with 0 where obs_tar_mask is False.
        pre_field_obs_masked = pre_field.masked_fill(not_obs_tar_mask, 0)
        tar_obs_masked = tar_obs.masked_fill(not_obs_tar_mask, 0)
        se_obs = (pre_field_obs_masked - tar_obs_masked) ** 2
        loss_obs = se_obs.mean()
        loss_obs_channel_wise = se_obs.mean(dim=(0, 2, 3))

        return {"loss_field": loss_field,
                "loss_field_channel_wise": loss_field_channel_wise,
                "loss_obs": loss_obs,
                "loss_obs_channel_wise": loss_obs_channel_wise,
                "loss_field_obs": loss_field_obs}

    ##########

    def _average_model_params(self):
        """Post-Local-SGD reconciliation: replace each rank's weights with the cross-rank
        mean. Averages parameters only (optimizer moments stay local, per the standard
        post-local-SGD recipe). Flattens into ONE all-reduce so the whole payload costs a
        single collective every H steps, not one per parameter tensor."""
        ws = dist.get_world_size()
        params = list(self._ddp_module.module.parameters())
        with torch.no_grad():
            flat = torch._utils._flatten_dense_tensors([p.data for p in params])
            dist.all_reduce(flat)
            flat /= ws
            for p, val in zip(params, torch._utils._unflatten_dense_tensors(
                    flat, [p.data for p in params])):
                p.data.copy_(val)

    ##########

    def _prepare_batch(self, data):
        """Move a batch to device; assemble derived tensors on GPU when enabled.

        With params.gpu_assemble the dataloader returns *raw* components and the
        heavy arithmetic (field_obs_tar / residual / concatenate) runs here on the
        GPU instead of in the CPU workers -- lighter workers, fewer needed, less
        core contention on the training rank. Bit-faithful to the CPU path.
        Returns: inp, inp_hrrr, target_field, target_obs, target_field_obs,
                 field_mask, obs_tar_mask (all on device).
        """
        nb = as_bool(self.params.non_blocking)
        to_dev = lambda t: t.to(self.device, dtype=torch.float, non_blocking=nb)
        to_bool = lambda t: t.to(self.device, dtype=torch.bool, non_blocking=nb)

        if as_bool(getattr(self.params, "gpu_assemble", False)):
            (inp_hrrr, inp_obs, topo, field_tar, obs_tar,
             field_mask, obs_tar_mask, heldout_mask, obs_source, _, _) = data
            inp_hrrr = to_dev(inp_hrrr)
            inp_obs = to_dev(inp_obs)
            topo = to_dev(topo)
            field_tar = to_dev(field_tar)
            obs_tar = to_dev(obs_tar)
            obs_tar_mask = to_bool(obs_tar_mask)
            field_mask = to_bool(field_mask)
            heldout_mask = to_bool(heldout_mask).unsqueeze(1)  # (B,1,H,W): broadcasts over vars
            # obs_source (1=mesonet, 2=METAR, 0=unset), (B,1,H,W) to broadcast over vars.
            obs_source = obs_source.to(self.device, dtype=torch.int8, non_blocking=nb).unsqueeze(1)

            # field_obs_tar: target field with obs substituted at observed
            # locations. Built from RAW field_tar/obs_tar, BEFORE any residual.
            field_obs_tar = field_tar.clone()
            field_obs_tar[obs_tar_mask] = 0
            field_obs_tar += obs_tar

            if as_bool(self.params.learn_residual):
                field_tar = field_tar - inp_hrrr
                obs_tar = obs_tar - inp_hrrr
                field_obs_tar = field_obs_tar - inp_hrrr

            inp = torch.cat((inp_hrrr, inp_obs, topo), dim=1)  # (B,C,H,W): channel dim
            if self.channels_last:
                inp = inp.contiguous(memory_format=torch.channels_last)
            return (inp, inp_hrrr, field_tar, obs_tar, field_obs_tar,
                    field_mask, obs_tar_mask, heldout_mask, obs_source)

        # --- legacy CPU-assembled path (unchanged behavior) ---
        (inp, field_tar, obs_tar, field_obs_tar, inp_hrrr, _, _,
         field_mask, obs_tar_mask, heldout_mask, obs_source) = data
        inp = to_dev(inp)
        if self.channels_last:
            inp = inp.contiguous(memory_format=torch.channels_last)
        inp_hrrr = to_dev(inp_hrrr)
        field_tar = to_dev(field_tar)
        obs_tar = to_dev(obs_tar)
        field_obs_tar = to_dev(field_obs_tar)
        field_mask = to_bool(field_mask)
        obs_tar_mask = to_bool(obs_tar_mask)
        heldout_mask = to_bool(heldout_mask).unsqueeze(1)  # (B,1,H,W): broadcasts over vars
        obs_source = obs_source.to(self.device, dtype=torch.int8, non_blocking=nb).unsqueeze(1)
        return (inp, inp_hrrr, field_tar, obs_tar, field_obs_tar,
                field_mask, obs_tar_mask, heldout_mask, obs_source)

    ##########

    def train_one_epoch(self):
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print(f"Training...")
        self.epoch += 1
        if self.params.resuming:
            self.resumeEpoch += 1
        tr_time = 0
        data_time = 0
        steps_in_one_epoch = 0
        # Accumulate scalar losses on-GPU and sync once at epoch end. Per-step .item()
        # forced a host<->device sync every step, stalling the pipeline.
        loss_field = torch.zeros((), device=self.device)
        loss_obs = torch.zeros((), device=self.device)
        loss_field_obs = torch.zeros((), device=self.device)
        loss_field_channel_wise = torch.zeros(len(self.params.target_vars), device=self.device, dtype=float)
        loss_obs_channel_wise = torch.zeros(len(self.params.target_vars), device=self.device, dtype=float)
        
        # --- optional profiling (env-gated, DDP-safe) ---------------------------
        # ADAF_PROFILE=1 profiles a short window of the LAST epoch (epoch 1 = compile
        # warmup, so the trace is steady-state). ADAF_MAX_STEPS caps steps per epoch
        # so the job is short; the cap is identical across ranks so no DDP collective
        # desync from an early break. Trace + key_averages dumped on rank 0 only.
        max_steps = int(os.environ.get("ADAF_MAX_STEPS", "0") or 0)
        prof = None
        if os.environ.get("ADAF_PROFILE", "") and self.epoch == self.params.max_epochs:
            from torch.profiler import profile, ProfilerActivity, schedule
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=2, active=8, repeat=1),
                record_shapes=True, with_stack=False, profile_memory=False,
            )
            prof.__enter__()

        self.model.train()
        # data_time must span the loader's blocking next() -- the wait for a batch, which
        # happens at the `for` line, NOT after `data` is already in hand. Start the clock
        # before the loop and reset it at the end of each iteration so each batch's real
        # I/O wait is captured. The old placement (data_start set AFTER the yield) timed
        # only the tuple unpack (~microseconds), making data_time read ~0 and hiding any
        # dataloader bottleneck.
        data_start = time.time()
        for i, data in enumerate(self.train_data_loader):
            data_time += time.time() - data_start
            self.iters += 1
            steps_in_one_epoch += 1

            # On resume self.iters is restored from the checkpoint and is past the warmup,
            # so this is a no-op -- a trained model should not be re-warmed. ReduceLROnPlateau
            # also writes param_groups[*]["lr"], but with scheduler_patience it cannot fire
            # inside a warmup that completes in epoch 1.
            if 0 < self.iters <= self._warmup_iters:
                warm_lr = self.params.lr * self.iters / self._warmup_iters
                for g in self.optimizer.param_groups:
                    g["lr"] = warm_lr

            tr_start = time.time()

            self.optimizer.zero_grad()
            # Post-Local-SGD: after warmup, skip DDP's gradient all-reduce (grads stay
            # local); weights are averaged across ranks every H steps below.
            in_local_phase = (self._localsgd_h > 0 and self._ddp_module is not None
                              and self.iters > self._localsgd_warmup)
            sync_ctx = (self._ddp_module.no_sync() if in_local_phase
                        else contextlib.nullcontext())
            with sync_ctx, amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                (inp, inp_hrrr, target_field, target_obs, target_field_obs,
                 field_mask, obs_tar_mask, _, _) = self._prepare_batch(data)

                # No EncDec_two_encoder code here either
                gen = self.model(inp)

                loss = self._loss_fn(pre_field=gen,
                                     tar_field=target_field,
                                     tar_obs=target_obs,
                                     tar_field_obs=target_field_obs,
                                     field_mask=field_mask,
                                     obs_tar_mask=obs_tar_mask)

                loss_field += loss["loss_field"].detach()
                loss_obs += loss["loss_obs"].detach()
                loss_field_obs += loss["loss_field_obs"].detach()
                loss_field_channel_wise += loss["loss_field_channel_wise"].detach()
                loss_obs_channel_wise += loss["loss_obs_channel_wise"].detach()

                # The single backwarded loss, selected by target. (Was three near-identical
                # if-blocks; folded so grad clipping has one insertion point.)
                if self.params.target == "obs": # target = sparse observations only
                    bwd_loss = loss["loss_obs"]
                elif self.params.target == "analysis": # target = gridded fields only, no obs
                    bwd_loss = loss["loss_field"]
                elif self.params.target == "analysis_obs": # gridded fields + sparse observations
                    bwd_loss = loss["loss_field_obs"]
                else:
                    raise ValueError(f"unknown params.target: {self.params.target}")

                if self.params.enable_amp:
                    self.gscaler.scale(bwd_loss).backward()
                    if self._grad_clip > 0:
                        # Grads are still scaled by the GradScaler here; unscale first or
                        # the clip threshold would be applied to the scaled values. No-op
                        # for bf16 (scaler disabled -> scale factor 1).
                        self.gscaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                else:
                    bwd_loss.backward()
                    if self._grad_clip > 0:
                        clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.optimizer.step()

                # Post-Local-SGD reconciliation: average weights across ranks every H steps.
                if in_local_phase and (self.iters % self._localsgd_h == 0):
                    self._average_model_params()

                tr_time += time.time() - tr_start

            if prof is not None:
                prof.step()
            # Uniform across ranks -> safe early break (no collective mismatch).
            if max_steps and steps_in_one_epoch >= max_steps:
                break
            data_start = time.time()  # start timing the wait for the NEXT batch

        if prof is not None:
            prof.__exit__(None, None, None)
            if self.params.world_rank == 0:
                trace_dir = os.environ.get("ADAF_PROFILE_DIR") or os.path.join(self.params.experiment_dir, "profile")
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = os.path.join(trace_dir, f"step_trace_ep{self.epoch}.json")
                prof.export_chrome_trace(trace_path)
                ka = prof.key_averages()
                print("==== PROFILE: top by self CUDA time ====", flush=True)
                print(ka.table(sort_by="self_cuda_time_total", row_limit=25), flush=True)
                print("==== PROFILE: top by self CPU time ====", flush=True)
                print(ka.table(sort_by="self_cpu_time_total", row_limit=20), flush=True)
                print(f"==== PROFILE: chrome trace -> {trace_path} ====", flush=True)

        # Single host<->device sync per epoch (was once per step via .item()).
        logs = {"loss_field": (loss_field / steps_in_one_epoch).item(),
                "loss_obs": (loss_obs / steps_in_one_epoch).item(),
                "loss_field_obs": (loss_field_obs / steps_in_one_epoch).item(),
                "steps": steps_in_one_epoch}
        
        #This might need a rewrite, but leave it for now
        for i_, var_ in enumerate(self.params.target_vars):
            tmp_var_1 = loss_obs_channel_wise[i_] / steps_in_one_epoch
            tmp_var_2 = loss_field_channel_wise[i_] / steps_in_one_epoch
            logs[f"loss_obs_{var_}"] = tmp_var_1
            logs[f"loss_field_{var_}"] = tmp_var_2

        # Calc and sync loss across all GPUs
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                logs[key] = torch.tensor(logs[key], device=self.device)
                dist.all_reduce(logs[key])
                logs[key] = float(logs[key] / dist.get_world_size()) #could be params.world_size, why the need for the separate call? But it's more robust, so leave for now

        step_time = tr_time / steps_in_one_epoch
        
        return tr_time, data_time, step_time, logs
    
    ##########

    def validate_one_epoch(self):
        """Validate, and compute the metric that now drives LR + checkpoint selection:
        MSE against the HELD-OUT stations -- the cells hidden from the model input by
        the dataloader's validation dropout (fixed 10%, deterministic per file, no
        block), scored against the obs that were kept in the target. That is exactly
        the deployed skill heldout_eval.py measures, computed for free here. The old
        selection keyed on TRAINING loss_field, a quantity the optimizer does not even
        minimize (it minimizes loss_field_obs)."""
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print("Validating...")
        self.model.eval()

        nvar = len(self.params.target_vars)
        valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)
        valid_loss_field = valid_buff[0].view(-1)
        valid_loss_obs = valid_buff[1].view(-1)
        valid_loss_field_obs = valid_buff[2].view(-1)
        valid_steps = valid_buff[3].view(-1)
        # Held-out SE and cell counts are summed (not averaged per step) so the pooled
        # metric weights every held-out station equally, regardless of how many a given
        # cycle happens to have.
        heldout_se = torch.zeros(nvar, dtype=torch.float32, device=self.device)
        heldout_n = torch.zeros(nvar, dtype=torch.float32, device=self.device)
        # Reference scores at the SAME held-out cells, free (both fields are already in
        # the batch): RTMA is the analysis we must beat, HRRR the background we must
        # improve on. Without these the log says "MSE went down" but not "down past
        # RTMA", which is the question the whole run is asking.
        heldout_se_rtma = torch.zeros(nvar, dtype=torch.float32, device=self.device)
        heldout_se_hrrr = torch.zeros(nvar, dtype=torch.float32, device=self.device)

        valid_start = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader):
                # No plotting code here
                # No EncDec_two_encoder code here
                (inp, inp_hrrr, target_field, target_obs, target_field_obs,
                 field_mask, obs_tar_mask, heldout_mask, obs_source) = self._prepare_batch(data)

                # No EncDec_two_encoder code here either
                gen = self.model(inp)

                loss = self.loss_function(pre_field=gen,
                                          tar_field=target_field,
                                          tar_obs=target_obs,
                                          tar_field_obs=target_field_obs,
                                          field_mask=field_mask,
                                          obs_tar_mask=obs_tar_mask)

                valid_steps += 1.0
                valid_loss_field += loss["loss_field"]
                valid_loss_obs += loss["loss_obs"]
                valid_loss_field_obs += loss["loss_field_obs"]

                # Held-out cells: dropped from the input AND carrying a real ob in the
                # target. target_field_obs holds the ob there (obs were substituted
                # before the residual, and gen is a residual too, so the difference is
                # the model-vs-ob error either way).
                # heldout_metric_source restricts the metric (and thus best_ckpt + LR
                # selection) to one obs network. "metar" scores clean METAR only -- the
                # same signal heldout_eval.py --source metar reports and what the final
                # A/B is judged on -- so selection tracks the deployed skill instead of
                # noisy pooled mesonet. "all" (default) keeps the legacy pooled behavior.
                sel = heldout_mask & obs_tar_mask
                _msrc = getattr(self.params, "heldout_metric_source", "all")
                if _msrc == "metar":
                    sel = sel & (obs_source == 2)
                elif _msrc == "mesonet":
                    sel = sel & (obs_source == 1)
                sel = sel.float()
                ob = target_field_obs.float()          # the ob at those cells (residual space)
                se = ((gen.float() - ob) ** 2) * sel
                heldout_se += se.sum(dim=(0, 2, 3))
                heldout_n += sel.sum(dim=(0, 2, 3))

                # Same cells, same units. Everything here is a residual vs HRRR, so
                # HRRR's own prediction is exactly 0 and RTMA's is target_field (the
                # RTMA analysis, which at these cells was NOT overwritten by the ob --
                # obs are substituted into target_field_obs only).
                heldout_se_rtma += (((target_field.float() - ob) ** 2) * sel).sum(dim=(0, 2, 3))
                heldout_se_hrrr += ((ob ** 2) * sel).sum(dim=(0, 2, 3))

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(heldout_se)
            dist.all_reduce(heldout_n)
            dist.all_reduce(heldout_se_rtma)
            dist.all_reduce(heldout_se_hrrr)

        # divide by number of steps
        valid_buff[0:3] = valid_buff[0:3] / valid_buff[3]
        valid_buff_cpu = valid_buff.detach().cpu().numpy()

        n = heldout_n.clamp(min=1.0)
        heldout_mse = (heldout_se / n).detach().cpu().numpy()
        heldout_mse_rtma = (heldout_se_rtma / n).detach().cpu().numpy()
        heldout_mse_hrrr = (heldout_se_hrrr / n).detach().cpu().numpy()

        logs = {"valid_loss_field": valid_buff_cpu[0],
                "valid_loss_obs": valid_buff_cpu[1],
                "valid_loss_field_obs": valid_buff_cpu[2],
                "valid_heldout_mse": float(heldout_mse.mean()),
                "valid_heldout_mse_rtma": float(heldout_mse_rtma.mean()),
                "valid_heldout_mse_hrrr": float(heldout_mse_hrrr.mean()),
                "valid_heldout_n": float(heldout_n.sum().item())}
        for i_, var_ in enumerate(self.params.target_vars):
            logs[f"valid_heldout_mse_{var_}"] = float(heldout_mse[i_])
            logs[f"valid_heldout_mse_rtma_{var_}"] = float(heldout_mse_rtma[i_])
            logs[f"valid_heldout_mse_hrrr_{var_}"] = float(heldout_mse_hrrr[i_])

        valid_time = time.time() - valid_start

        return valid_time, logs

    ##########

    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model

        print(f"Saving model to {checkpoint_path}")
        torch.save({"iters": self.iters,
                    "epoch": self.epoch,
                    "best_valid_metric": self.best_valid_metric,
                    "model_state": model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                    checkpoint_path)
        
    ##########

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.params.local_rank}")
        try:
            self.model.load_state_dict(checkpoint["model_state"]) #Works if model was trained/saved without DDP
        except ValueError: # model was stored using DDP, which prepends "module."
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"]
        # .get(): checkpoints written before this key existed resume with a fresh 1e6.
        self.best_valid_metric = checkpoint.get("best_valid_metric", 1.0e6)
        self.resumeEpoch = 0
        if self.params.resuming: # restore checkpoint is used for finetuning as well as resuming.
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for g in self.optimizer.param_groups: # uses config specified lr
                g["lr"] = self.params.lr

    ##########

    # (2026-06-05) Not used in this script; should be used externally for inference, though maybe this is better suited to be spun off into its own thing, not dependent on Trainer params
    def load_model(self, model_path): 
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print(f"Loading the model weights from {model_path}")

        checkpoint = torch.load(model_path, map_location=f"cuda:{self.params.local_rank}")

        if dist.is_initialized():
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            new_model_state = OrderedDict()
            if "model_state" in checkpoint:
                model_key = "model_state"
            else:
                model_key = "state_dict"

            for key in checkpoint[model_key].keys():
                if "module." in key: # model was stored using DDP which prepends "module."
                    name = str(key[7:])
                    new_model_state[name] = checkpoint[model_key][key]
                else:
                    new_model_state[key] = checkpoint[model_key][key]
            self.model.load_state_dict(new_model_state)
            self.model.eval()

    ##########

    def train(self):
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print("Starting the main training loop...")

        for epoch in range(self.startEpoch, self.params.max_epochs):
            epoch_wall_start = time.time()
            if dist.is_initialized(): # Sync epochs across GPUs
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)
            # start = time.time() #Not needed given timing in the *_one_epoch functions?

            # Train one epoch
            tr_time, data_time, step_time, train_logs = self.train_one_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]
            # No plotting code here

            if self.params.log_to_screen and self.params.world_rank==0: #only print once
                print(f"Epoch: {epoch + 1}")
                print(f"Training epoch time={tr_time: .2f} seconds")
                print(f"Training data load time={data_time: .2f} seconds")
                print(f"Training per-step time={step_time: .2f} seconds")
                print(f"Training loss: {train_logs['loss_field']}")
                print(f"Learning rate: {current_lr}")
                # Machine-parseable line for the throughput-sweep parser (parse_sweep.py)
                steps = train_logs["steps"]
                samples_per_sec = (steps * self.params.batch_size) / tr_time if tr_time > 0 else 0.0
                print(f"EPOCH_METRICS,epoch={epoch + 1},steps={steps},"
                      f"tr_time={tr_time:.4f},data_time={data_time:.4f},step_time={step_time:.4f},"
                      f"samples_per_sec={samples_per_sec:.4f},loss_field={train_logs['loss_field']:.6f}")

            # validate one epoch
            valid_time = 0.0
            valid_logs = None
            if epoch % self.params.valid_frequency == 0:
                valid_time, valid_logs = self.validate_one_epoch()

                if self.params.log_to_screen and self.params.world_rank==0: #only print once
                    print(f"Valid time={valid_time: .2f} seconds")
                    print(f"Valid loss={valid_logs['valid_loss_field']}")
                    # Scorecard at the held-out stations. skill = model MSE / RTMA MSE:
                    # < 1.0 means the model beats RTMA at stations it never saw -- the
                    # whole point of the run. e358 (no dropout) sat at ~1.6 for t.
                    _msrc = getattr(self.params, "heldout_metric_source", "all")
                    print(f"Valid held-out MSE [{_msrc}] over "
                          f"{int(valid_logs['valid_heldout_n'])} cells "
                          f"(model | rtma | hrrr | model/rtma):")
                    for v in self.params.target_vars:
                        m = valid_logs[f"valid_heldout_mse_{v}"]
                        r = valid_logs[f"valid_heldout_mse_rtma_{v}"]
                        h = valid_logs[f"valid_heldout_mse_hrrr_{v}"]
                        print(f"    {v:4s} {m:.6f} | {r:.6f} | {h:.6f} | {m / max(r, 1e-12):.3f}")
                    print(f"HELDOUT_METRICS,epoch={epoch + 1},"
                          f"model={valid_logs['valid_heldout_mse']:.6f},"
                          f"rtma={valid_logs['valid_heldout_mse_rtma']:.6f},"
                          f"hrrr={valid_logs['valid_heldout_mse_hrrr']:.6f},"
                          + ",".join(f"{v}={valid_logs[f'valid_heldout_mse_{v}']:.6f}"
                                     for v in self.params.target_vars))

            # LR scheduler + checkpoint selection now key on the held-out-station metric
            # (the deployed skill), not on the training loss.
            valid_metric = valid_logs["valid_heldout_mse"] if valid_logs else None

            if self.params.scheduler == "ReduceLROnPlateau" and valid_metric is not None:
                self.scheduler.step(valid_metric)

            # Save model checkpoint
            ckpt_time = 0.0
            if (self.params.world_rank == 0 and epoch % self.params.save_model_freq == 0 and self.params.save_checkpoint):
                _ck = time.time()
                self.save_checkpoint(self.params.checkpoint_path)
                ckpt_time += time.time() - _ck

            # If model is the best yet (regardless of save_model_freq), save to the best checkpoint path
            # !! This will wipe out the previous best model !! Needs modification for that case
            if (self.params.world_rank == 0 and self.params.save_checkpoint
                    and valid_metric is not None):
                if valid_metric <= self.best_valid_metric:
                    print(f"Held-out MSE improved from {self.best_valid_metric} to {valid_metric}")
                    self.best_valid_metric = valid_metric
                    _ck = time.time()
                    self.save_checkpoint(self.params.best_checkpoint_path)
                    ckpt_time += time.time() - _ck

            # --- per-phase wall-clock accounting (diagnostic) ------------------------
            # Full epoch wall time vs the sum of measured phases. `other` = wall minus
            # train/data/valid/ckpt: it captures dataloader worker (re)spawn (persistent_
            # workers is off -> workers rebuilt every epoch), sampler.set_epoch, DDP
            # straggler sync at the epoch boundary, the scheduler step, and logging.
            # A large `data` means I/O-bound batches; a large `other` means the cost is
            # between epochs (worker spawn / cold file cache), not in the GPU step.
            if self.params.log_to_screen and self.params.world_rank == 0:
                epoch_wall = time.time() - epoch_wall_start
                other = epoch_wall - tr_time - data_time - valid_time - ckpt_time
                print(f"PHASE_TIMING,epoch={epoch + 1},wall={epoch_wall:.2f},"
                      f"train={tr_time:.2f},data={data_time:.2f},valid={valid_time:.2f},"
                      f"ckpt={ckpt_time:.2f},other={other:.2f}")
        
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print(f"!!! Training finished !!!")
            print(f"Epochs: {epoch + 1}")
            print(f"Final epoch's loss: {train_logs['loss_field']}")
            print(f"Final epoch's learning rate: {current_lr}")


#######################################################


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    ### !! IMPORTANT !! ###
    parser.add_argument('--config_filepath', type=str, default="./config/params_default.yaml") #This should be changed per-run if modifying many params! If only modifying a few, passing in args on the command line should suffice

    args = set_user_params(parser)
   
    params = YParams(args.config_filepath)
    params.override_from_cli(args)

    # Get SLURM info for DDP and set params
    # params["world_size"] = int(os.environ.get("WORLD_SIZE")) #Not currently used
    params["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl")
    params["world_rank"] = dist.get_rank() 

    # Seed every rank so runs are reproducible (needed for the throughput-sweep
    # loss-overlap sanity check). DDP broadcasts rank-0 weights at construction,
    # so model init is consistent across ranks regardless.
    set_random_seed(params.seed)

    if params.log_to_screen and params.world_rank == 0:
        print("------ PARAMETER VALUES ------")
        for key, val in params.items():
            print(f"{key}: {val}")
        print("------------------------------")

    trainer = Trainer(params)
    trainer.train()

    dist.destroy_process_group()