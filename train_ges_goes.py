import os
import sys

# Define local Conda environment's ptxas path
# Honor an externally-staged copy (e.g. node-local /tmp) if the launcher set one:
# exec'ing ptxas directly off Lustre from many ranks at once raises ETXTBSY.
conda_ptxas_path = os.environ.get(
    "TRITON_PTXAS_PATH",
    "/scratch3/BMC/wrfruc/aschein/miniconda/envs/ADAF_environment/bin/ptxas",
)

# Explicitly assign both Triton lookup variables so it works on any GPU generation
os.environ["TRITON_PTXAS_PATH"] = conda_ptxas_path
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = conda_ptxas_path

# Optional: Ensure the folder itself is visible in the system PATH for sub-shells
os.environ["PATH"] = f"/scratch3/BMC/wrfruc/aschein/miniconda/envs/ADAF_environment/bin:{os.environ.get('PATH', '')}"

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

from utils.dataloader_multifiles_ges_goes import get_data_loader
# from utils.logging_utils import log_to_file
from utils.YParams import YParams
from utils.misc_functions import set_user_params

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

    
    def prepare_batch(self, data):
        """Move a batch to device; assemble derived tensors on GPU when enabled.

        With params.gpu_assemble the dataloader returns *raw* components and the
        heavy arithmetic (field_obs_tar / residual / concatenate) runs here on the
        GPU instead of in the CPU workers -- lighter workers, fewer needed, less
        core contention on the training rank. Bit-faithful to the CPU path.
        Returns: inp, inp_ges, target_field, target_obs, target_field_obs,
             field_mask, obs_tar_mask, heldout_mask (all on device).
        """
        nb = self.params.non_blocking
        to_dev = lambda t: t.to(self.device, dtype=torch.float, non_blocking=nb)
        to_bool = lambda t: t.to(self.device, dtype=torch.bool, non_blocking=nb)
        use_sat = len(self.params.inp_sat_vars) != 0  # False for no-GOES runs: the dataloader's inp_sat is only a placeholder

        if getattr(self.params, "gpu_assemble", False):
            (inp_pred, inp_obs, inp_sat, topo, field_tar, obs_tar, field_mask, obs_tar_mask, heldout_mask, _, _) = data

            inp_pred = to_dev(inp_pred)
            inp_obs = to_dev(inp_obs)
            inp_sat = to_dev(inp_sat) if use_sat else None
            topo = to_dev(topo)
            field_tar = to_dev(field_tar)
            obs_tar = to_dev(obs_tar)
            obs_tar_mask = to_bool(obs_tar_mask)
            field_mask = to_bool(field_mask)
            heldout_mask = to_bool(heldout_mask).unsqueeze(1)

            # field_obs_tar: target field with obs substituted at observed locations
            field_obs_tar = field_tar.clone()
            field_obs_tar[obs_tar_mask] = 0
            field_obs_tar += obs_tar

            if self.params.learn_residual:
                field_tar = field_tar - inp_pred
                obs_tar = obs_tar - inp_pred
                field_obs_tar = field_obs_tar - inp_pred

            inp = torch.cat((inp_pred, inp_obs, *([inp_sat] if use_sat else []), topo), dim=1)  # (B,C,H,W): channel dim

            if self.channels_last:
                inp = inp.contiguous(memory_format=torch.channels_last)
            return (inp, inp_pred, field_tar, obs_tar, field_obs_tar, field_mask, obs_tar_mask, heldout_mask)

        # Same assembly when gpu_assemble is disabled.
        else:
            (inp_pred, inp_obs, inp_sat, topo, field_tar, obs_tar, field_mask, obs_tar_mask, heldout_mask, _, _) = data
            inp_obs = to_dev(inp_obs)
            inp_sat = to_dev(inp_sat) if use_sat else None
            topo = to_dev(topo)
            inp_pred = to_dev(inp_pred)
            field_tar = to_dev(field_tar)
            obs_tar = to_dev(obs_tar)
            obs_tar_mask = to_bool(obs_tar_mask)
            field_mask = to_bool(field_mask)
            heldout_mask = to_bool(heldout_mask).unsqueeze(1)
            field_obs_tar = field_tar.clone()
            field_obs_tar[obs_tar_mask] = 0
            field_obs_tar += obs_tar
            if self.params.learn_residual:
                field_tar = field_tar - inp_pred
                obs_tar = obs_tar - inp_pred
                field_obs_tar = field_obs_tar - inp_pred
            inp = torch.cat((inp_pred, inp_obs, *([inp_sat] if use_sat else []), topo), dim=1)
            if self.channels_last:
                inp = inp.contiguous(memory_format=torch.channels_last)
            return (inp, inp_pred, field_tar, obs_tar, field_obs_tar, field_mask, obs_tar_mask, heldout_mask)
    
    
    def __init__(self, params):
        self.params = params
        # self.set_device() #Should this be here when we set the device just below?

        # Set up local node
        torch.cuda.set_device(self.params.local_rank)
        self.device = torch.device("cuda", self.params.local_rank)
        print(f"world_rank: {self.params.world_rank} | local_rank: {self.params.local_rank} | device: {self.device} | num_data_workers={self.params.num_data_workers}")
        
        # Optimizations (experimental)
        if getattr(self.params, "tf32", False):
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        _amp_dtype = str(getattr(self.params, "amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if _amp_dtype in ("bf16", "bfloat16") else torch.float16
        self.channels_last = getattr(self.params, "channels_last", False)
        
        
        # Load model
        if params.arch == "lowres":
            from models.encdec_lowres import LowResEncDec as model
        else:
            from models.encdec import EncDec as model 
        self.model = model(self.params).to(self.device)
        
        # Experimental
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)


        # Load training and validation data
        print(f"[world_rank: {self.params.world_rank}] Begin data loading \n") #may need to be changed to rank 0 only
        (self.train_data_loader, self.train_dataset, self.train_sampler) = get_data_loader(self.params,
                                                                                           self.params.train_data_path,
                                                                                           dist.is_initialized(),
                                                                                           train=True)
        
        (self.valid_data_loader, self.valid_dataset) = get_data_loader(self.params,
                                           self.params.valid_data_path,
                                           dist.is_initialized(),
                                           train=False)
        print(f"[world_rank: {self.params.world_rank}] Data loaded \n") #may need to be changed to rank 0 only or removed

        # Set up optimizer
        weight_decay = float(getattr(self.params, "weight_decay", 0.0))
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.ndim <= 1 or "relative_position_bias_table" in name else decay).append(parameter)
        param_groups = [{"params": decay, "weight_decay": weight_decay},
                        {"params": no_decay, "weight_decay": 0.0}]
        if self.params.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=float(self.params.lr))
        elif self.params.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(param_groups, lr=float(self.params.lr))
        else:
            raise Exception("Only Adam and AdamW optimizers implemented")
        
        if self.params.enable_amp:
            self.gscaler = amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        if getattr(self.params, "compile_model", False): #may have problems with cuda.h on PATH 
            import torch._inductor.config as _ind
            _ind.cpp.simdlen = 0
            self.model = torch.compile(self.model)

        # Set up distributed training
        if dist.is_initialized():
            # gradient_as_bucket_view: grads alias the reduce buckets (saves a copy + memory; numerically identical).
            # static_graph: opt-in -- lets DDP assume a fixed graph each step for better all-reduce/backward overlap.
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.params.local_rank],
                                                 output_device=[self.params.local_rank],
                                                 find_unused_parameters=self.params.ddp_find_unused_parameters,
                                                 gradient_as_bucket_view=True,
                                                 # broadcast_buffers default True re-broadcasts buffers from rank 0 every forward.
                                                 # These are deterministic here, so rebroadcast is avoidable overhead.
                                                 broadcast_buffers=getattr(self.params, "ddp_broadcast_buffers", True),
                                                 static_graph=getattr(self.params, "ddp_static_graph", False))

        # Post-Local-SGD controls (disabled by default unless explicitly set).
        self._ddp_module = self.model if dist.is_initialized() else None
        self._localsgd_h = max(0, int(getattr(self.params, "localsgd_h", 0) or 0))
        self._localsgd_warmup = max(0, int(getattr(self.params, "localsgd_warmup", 0) or 0))
        self._grad_clip = float(getattr(self.params, "grad_clip", 0) or 0)
        self._warmup_iters = int(getattr(self.params, "warmup_iters", 0) or 0)
        self._heldout_weight = float(getattr(self.params, "heldout_loss_weight", 0) or 0)
        if self._heldout_weight > 0 and not self.params.hold_out_obs:
            raise ValueError("heldout_loss_weight > 0 requires hold_out_obs")
        if getattr(self.params, "heldout_metric_source", "all") != "all" or getattr(self.params, "heldout_loss_source", "all") != "all":
            raise ValueError("Station data has no source labels; held-out metric and loss source must be 'all'")
        weights = getattr(self.params, "loss_channel_weights", [1.0] * len(self.params.target_vars))
        if len(weights) != len(self.params.target_vars):
            raise ValueError("loss_channel_weights must match target_vars")
        channel_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.loss_weights = (channel_weights / channel_weights.mean()).view(1, -1, 1, 1)

        # True wall timing requires CUDA sync around epoch boundaries; keep this toggle explicit.
        self.sync_epoch_timing = bool(getattr(self.params, "sync_epoch_timing", True))
        
        #leaving out compile_loss for now (should always be false?)
        
        self.iters = 0
        self.startEpoch = 0
        self.best_valid_metric = float("inf")
        self.best_train_metric = float("inf")
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
            if self.params.resume_checkpoint_path is not None:
                if self.params.log_to_screen and self.params.world_rank==0:
                    print(f"Loading checkpoint from {self.params.resume_checkpoint_path}")
                self.restore_checkpoint(self.params.resume_checkpoint_path)
            else:
                if self.params.world_rank==0:
                    print(f"resuming is True but resume_checkpoint_path was not provided!")
        
        self.epoch = self.startEpoch

        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print(f"Number of trainable model parameters: {self.count_parameters()}")
            if self._localsgd_h > 0:
                print(f"Post-Local-SGD: averaging weights every {self._localsgd_h} steps after {self._localsgd_warmup} warmup steps")

    ##########
    
    def loss_function(self,
                      pre_field,
                      tar_field,
                      tar_obs,
                      tar_field_obs,
                      field_mask=None,
                      obs_tar_mask=None,
                      heldout_sel=None,
                      mask_out_of_range=True):
        """
        pre_field: model's output
        tar_field: label, after normalization
        """
        
        # Create masked versions for field loss
        if mask_out_of_range: # fill input with 0 where field_mask is False
            not_field_mask = ~field_mask
            pre_field_masked = pre_field.masked_fill(not_field_mask, 0)
            tar_field_masked = tar_field.masked_fill(not_field_mask, 0)
            tar_field_obs_masked = tar_field_obs.masked_fill(not_field_mask, 0)
        else:
            pre_field_masked = pre_field
            tar_field_masked = tar_field
            tar_field_obs_masked = tar_field_obs

        # type 1 loss -- compute the squared error once and derive both the scalar mean and the per-channel mean from it (was two separate full-res mse passes).
        se_field = (pre_field_masked - tar_field_masked) ** 2
        loss_field = (se_field * self.loss_weights).mean()
        loss_field_channel_wise = se_field.mean(dim=(0, 2, 3))

        # type 2 loss
        loss_field_obs = (((pre_field_masked - tar_field_obs_masked) ** 2) * self.loss_weights).mean()

        # type 3 loss - use fresh masks for obs loss
        not_obs_tar_mask = ~obs_tar_mask  # fill input with 0 where obs_tar_mask is False.
        pre_field_obs_masked = pre_field.masked_fill(not_obs_tar_mask, 0)
        tar_obs_masked = tar_obs.masked_fill(not_obs_tar_mask, 0)
        se_obs = (pre_field_obs_masked - tar_obs_masked) ** 2
        loss_obs = se_obs.mean()
        loss_obs_channel_wise = se_obs.mean(dim=(0, 2, 3))

        if heldout_sel is not None:
            selection = heldout_sel.float()
            loss_heldout = (((pre_field - tar_field_obs) ** 2) * self.loss_weights * selection).sum() / selection.sum().clamp(min=1.0)
        else:
            loss_heldout = pre_field.new_zeros(())

        return {"loss_field": loss_field,
                "loss_field_channel_wise": loss_field_channel_wise,
                "loss_obs": loss_obs,
                "loss_obs_channel_wise": loss_obs_channel_wise,
                "loss_field_obs": loss_field_obs,
                "loss_heldout": loss_heldout}
    
    ##########

    def average_model_params(self):
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
            for p, val in zip(params, torch._utils._unflatten_dense_tensors(flat, [p.data for p in params])):
                p.data.copy_(val)

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
        if self.sync_epoch_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        epoch_wall_start = time.perf_counter()
        # Accumulate scalar losses on-GPU and sync once at epoch end. Per-step .item() forced a host<->device sync every step, stalling the pipeline.
        loss_field = torch.zeros((), device=self.device)
        loss_obs = torch.zeros((), device=self.device)
        loss_field_obs = torch.zeros((), device=self.device)
        loss_heldout = torch.zeros((), device=self.device)
        loss_field_channel_wise = torch.zeros(len(self.params.target_vars), device=self.device, dtype=float)
        loss_obs_channel_wise = torch.zeros(len(self.params.target_vars), device=self.device, dtype=float)
        
        self.model.train()
        data_start = time.time() #Start before the loop, otherwise it doesn't track properly
        for i, data in enumerate(self.train_data_loader):
            data_time += time.time() - data_start #Time taken for data loading is in the for loop statement!
            self.iters += 1
            steps_in_one_epoch += 1

            if 0 < self.iters <= self._warmup_iters:
                warm_lr = float(self.params.lr) * self.iters / self._warmup_iters
                for group in self.optimizer.param_groups:
                    group["lr"] = warm_lr

            tr_start = time.time()

            self.optimizer.zero_grad()
            
            # Post-Local-SGD: after warmup, skip DDP's gradient all-reduce (grads stay local); weights are averaged across ranks every H steps below.
            in_local_phase = (self._localsgd_h > 0 and self._ddp_module is not None and self.iters > self._localsgd_warmup)
            sync_ctx = (self._ddp_module.no_sync() if in_local_phase else contextlib.nullcontext())

            with sync_ctx, amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                (inp, inp_ges, target_field, target_obs, target_field_obs,
                 field_mask, obs_tar_mask, heldout_mask) = self.prepare_batch(data)

                gen = self.model(inp)

                loss = self.loss_function(pre_field=gen,
                                          tar_field=target_field,
                                          tar_obs=target_obs,
                                          tar_field_obs=target_field_obs,
                                          field_mask=field_mask,
                                          obs_tar_mask=obs_tar_mask,
                                          heldout_sel=(heldout_mask & obs_tar_mask) if self._heldout_weight > 0 else None)
                
                loss_field += loss["loss_field"].detach()
                loss_obs += loss["loss_obs"].detach()
                loss_field_obs += loss["loss_field_obs"].detach()
                loss_heldout += loss["loss_heldout"].detach()
                loss_field_channel_wise += loss["loss_field_channel_wise"].detach()
                loss_obs_channel_wise += loss["loss_obs_channel_wise"].detach()

                loss_key = {"obs": "loss_obs", "analysis": "loss_field",
                            "analysis_obs": "loss_field_obs"}.get(self.params.target)
                if loss_key is None:
                    raise ValueError(f"unknown target: {self.params.target}")
                backward_loss = loss[loss_key] + self._heldout_weight * loss["loss_heldout"]
                if self.params.enable_amp:
                    self.gscaler.scale(backward_loss).backward()
                    if self._grad_clip > 0:
                        self.gscaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                else:
                    backward_loss.backward()
                    if self._grad_clip > 0:
                        clip_grad_norm_(self.model.parameters(), self._grad_clip)
                    self.optimizer.step()

                # Post-Local-SGD reconciliation: average weights across ranks every H steps.
                if in_local_phase and (self.iters % self._localsgd_h == 0):
                    self.average_model_params()

                tr_time += time.time() - tr_start
            data_start = time.time()  # start timing the wait for the next batch

        if self.sync_epoch_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        epoch_wall_time = time.perf_counter() - epoch_wall_start

        # Single host<->device sync per epoch (was once per step via .item()).
        logs = {"loss_field": (loss_field / steps_in_one_epoch).item(),
                "loss_obs": (loss_obs / steps_in_one_epoch).item(),
                "loss_field_obs": (loss_field_obs / steps_in_one_epoch).item(),
                "loss_heldout": (loss_heldout / steps_in_one_epoch).item(),
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
                val = logs[key]
                if torch.is_tensor(val):
                    tval = val.detach()
                    if tval.device != self.device:
                        tval = tval.to(self.device)
                else:
                    tval = torch.tensor(val, device=self.device)
                dist.all_reduce(tval)
                logs[key] = float((tval / dist.get_world_size()).item())

        step_time = epoch_wall_time / steps_in_one_epoch
        logs["train_epoch_cpu_dispatch_time"] = tr_time
        logs["train_epoch_wall_time"] = epoch_wall_time
        
        return epoch_wall_time, data_time, step_time, logs
    
    ##########

    def validate_one_epoch(self):
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print("Validating...")
        self.model.eval()

        valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)
        valid_loss_field = valid_buff[0].view(-1)
        valid_loss_obs = valid_buff[1].view(-1)
        valid_loss_field_obs = valid_buff[2].view(-1)
        valid_samples = valid_buff[3].view(-1)
        heldout_error = torch.zeros(len(self.params.target_vars), device=self.device)
        heldout_count = torch.zeros_like(heldout_error)
        anl_error = torch.zeros_like(heldout_error)
        ges_error = torch.zeros_like(heldout_error)

        valid_start = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader):
                (inp, inp_ges, target_field, target_obs, target_field_obs,
                 field_mask, obs_tar_mask, heldout_mask) = self.prepare_batch(data)
                batch_size = inp.shape[0]

                gen = self.model(inp)

                loss = self.loss_function(pre_field=gen,
                                          tar_field=target_field,
                                          tar_obs=target_obs,
                                          tar_field_obs=target_field_obs,
                                          field_mask=field_mask,
                                          obs_tar_mask=obs_tar_mask)
                
                valid_samples += batch_size
                valid_loss_field += loss["loss_field"] * batch_size
                valid_loss_obs += loss["loss_obs"] * batch_size
                valid_loss_field_obs += loss["loss_field_obs"] * batch_size
                selected = (heldout_mask & obs_tar_mask).float()
                observed = target_field_obs.float()
                heldout_error += (((gen.float() - observed) ** 2) * selected).sum(dim=(0, 2, 3))
                heldout_count += selected.sum(dim=(0, 2, 3))
                anl_error += (((target_field.float() - observed) ** 2) * selected).sum(dim=(0, 2, 3))
                ges_error += ((observed ** 2) * selected).sum(dim=(0, 2, 3))
        
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            for values in (heldout_error, heldout_count, anl_error, ges_error):
                dist.all_reduce(values)

        # Divide loss sums by the number of validation samples.
        valid_buff[0:3] = valid_buff[0:3] / valid_buff[3]
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        station_count = float(heldout_count.sum().item())
        denominator = heldout_count.sum().clamp(min=1)
        
        logs = {"valid_loss_field": valid_buff_cpu[0],
                "valid_loss_obs": valid_buff_cpu[1],
                "valid_loss_field_obs": valid_buff_cpu[2],
            "valid_heldout_n": station_count,
            "valid_heldout_mse": float((heldout_error.sum() / denominator).item()) if station_count else float("inf"),
            "valid_heldout_mse_anl": float((anl_error.sum() / denominator).item()) if station_count else float("inf"),
            "valid_heldout_mse_ges": float((ges_error.sum() / denominator).item()) if station_count else float("inf")}
        for index, variable in enumerate(self.params.target_vars):
            logs[f"valid_heldout_mse_{variable}"] = float((heldout_error[index] / heldout_count[index].clamp(min=1)).item())
        
        valid_time = time.time() - valid_start

        return valid_time, logs

    ##########

    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model

        print(f"Saving model to {checkpoint_path}")
        temporary_path = checkpoint_path + ".tmp"
        torch.save({"iters": self.iters,
                "epoch": self.epoch,
                "best_valid_metric": self.best_valid_metric,
                "best_train_metric": self.best_train_metric,
                "model_state": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None},
               temporary_path)
        os.replace(temporary_path, checkpoint_path)
        
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
        self.best_valid_metric = checkpoint.get("best_valid_metric", float("inf"))
        self.best_train_metric = checkpoint.get("best_train_metric", float("inf"))
        self.resumeEpoch = 0 
        if self.params.resuming: # restore checkpoint is used for finetuning as well as resuming.
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
                # keep patience/factor from the current config, not the checkpoint
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.patience = self.params.scheduler_patience
                    self.scheduler.factor = self.params.lr_reduce_factor
            elif self.scheduler is not None and self.params.world_rank == 0:
                print("Checkpoint has no scheduler state; scheduler starts fresh")
            resume_lr = getattr(self.params, "resume_lr", None)
            if resume_lr is not None:
                for g in self.optimizer.param_groups:
                    g["lr"] = float(resume_lr)
            if self.params.world_rank == 0:
                source = "resume_lr" if resume_lr is not None else "checkpoint"
                print(f"Resuming with lr={self.optimizer.param_groups[0]['lr']} (from {source})")

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
            self.train_sampler.set_epoch(epoch)
            if hasattr(self.valid_data_loader.sampler, "set_epoch"):
                self.valid_data_loader.sampler.set_epoch(epoch)

            # if self.train_sampler is not None and hasattr(self.train_sampler, "build_indices"):
            #     epoch_indices = self.train_sampler.build_indices()
            #     print(f"\n [world_rank {self.params.world_rank}] epoch {epoch + 1} train_sampler (len = {len(epoch_indices)}) indices: {epoch_indices} \n")
            
            # Train one epoch
            tr_time, data_time, step_time, train_logs = self.train_one_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            if self.params.log_to_screen and self.params.world_rank==0: #only print once
                print(f"Epoch: {epoch + 1}")
                print(f"Training steps this epoch={train_logs['steps']}")
                print(f"Training epoch time={tr_time: .2f} seconds")
                print(f"Training data load time={data_time: .2f} seconds")
                print(f"Training per-step time={step_time: .2f} seconds")
                print(f"Training CPU dispatch time={train_logs['train_epoch_cpu_dispatch_time']: .2f} seconds")
                print(f"Training loss: {train_logs['loss_field']}")
                print(f"Training held-out loss: {train_logs['loss_heldout']}")
                print(f"Learning rate: {current_lr}")

            # validate one epoch
            valid_logs = None
            if epoch % self.params.valid_frequency == 0:
                valid_time, valid_logs = self.validate_one_epoch()
                
                if self.params.log_to_screen and self.params.world_rank==0: #only print once
                    print(f"Valid time={valid_time: .2f} seconds")
                    print(f"Valid loss={valid_logs['valid_loss_field']}")
                    print("Held-out MSE:", valid_logs["valid_heldout_mse"],
                          "analysis:", valid_logs["valid_heldout_mse_anl"],
                          "GES:", valid_logs["valid_heldout_mse_ges"])

            # With no held-out stations (hold_out_obs False) fall back to the training objective on the validation set, so the LR scheduler and best checkpoint still follow validation
            valid_metric = None
            if valid_logs:
                valid_metric = (valid_logs["valid_heldout_mse"] if valid_logs["valid_heldout_n"] > 0
                                else float(valid_logs["valid_loss_field_obs"]))
            if self.params.scheduler == "ReduceLROnPlateau" and valid_metric is not None:
                self.scheduler.step(valid_metric)

            if self.params.world_rank == 0 and self.params.save_checkpoint:
                save_best = False
                if valid_metric is not None and valid_metric <= self.best_valid_metric:
                    print(f"Validation metric improved from {self.best_valid_metric} to {valid_metric}")
                    self.best_valid_metric = valid_metric
                    save_best = True
                train_metric = train_logs["loss_field"]
                if train_metric <= self.best_train_metric:
                    print(f"Training loss improved from {self.best_train_metric} to {train_metric}")
                    self.best_train_metric = train_metric
                    save_best = True
                if save_best:
                    self.save_checkpoint(self.params.best_checkpoint_path)

            if (self.params.world_rank == 0 and epoch % self.params.save_model_freq == 0 and self.params.save_checkpoint):
                self.save_checkpoint(self.params.checkpoint_path)
        
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print(f"!!! Training finished !!!")
            print(f"Epochs: {epoch + 1}")
            print(f"Final epoch's loss: {train_logs['loss_field']}")
            print(f"Final epoch's learning rate: {current_lr}")


#######################################################


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    ### !! IMPORTANT !! ###
    parser.add_argument('--config_filepath', type=str, default="./config/params_lowres_ges_goes.yaml") #This should be changed per-run if modifying many params! If only modifying a few, passing in args on the command line should suffice

    args = set_user_params(parser)
   
    params = YParams(args.config_filepath)
    params.override_from_cli(args)

    # Get SLURM info for DDP and set params
    params["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl")
    params["world_rank"] = dist.get_rank() 

    set_random_seed(params.seed)
    
    if params.log_to_screen and params.world_rank == 0:
        print("------ PARAMETER VALUES ------")
        for key, val in params.items():
            print(f"{key}: {val}")
        print("------------------------------")

    trainer = Trainer(params)
    trainer.train()

    dist.destroy_process_group()