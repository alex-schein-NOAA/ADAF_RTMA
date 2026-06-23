import os
import time
# import wandb
import random
import datetime
import argparse
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from utils.dataloader_multifiles import get_data_loader
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

    def __init__(self, params):
        self.params = params
        # self.set_device() #Should this be here when we set the device just below?

        # Set up local node
        torch.cuda.set_device(self.params.local_rank)
        self.device = torch.device("cuda", self.params.local_rank)
        print(f"world_rank: {self.params.world_rank} | local_rank: {self.params.local_rank} | device: {self.device} | num_data_workers={self.params.num_data_workers}")
        
        # Load model
        from models.encdec import EncDec as model #EncDec_two_encoder in the original script doesn't exist...
        self.model = model(self.params).to(self.device)

        # Load training and validation data
        print(f"[world_rank: {self.params.world_rank}] Begin data loading \n") #may need to be changed to rank 0 only
        (self.train_data_loader, self.train_dataset, self.train_sampler) = get_data_loader(self.params,
                                                                                           self.params.train_data_path,
                                                                                           dist.is_initialized(),
                                                                                           train=True)
        
        (self.valid_data_loader, self.valid_dataset, self.valid_sampler) = get_data_loader(self.params,
                                                                                           self.params.valid_data_path,
                                                                                           dist.is_initialized(),
                                                                                           train=True)
        print(f"[world_rank: {self.params.world_rank}] Data loaded \n") #may need to be changed to rank 0 only or removed

        # Set up optimizer
        if self.params.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        elif self.params.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr)
        else:
            raise Exception("Only Adam and AdamW optimizers implemented")
        
        if self.params.enable_amp:
            self.gscaler = amp.GradScaler()

        # Set up distributed training
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.params.local_rank],
                                                 output_device=[self.params.local_rank],
                                                 find_unused_parameters=True)
        self.iters = 0
        self.startEpoch = 0
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
        
        # Create masked versions for field loss
        # (2026-06-05) note these are still attached, i.e. differentiable for gradient flow
        pre_field_masked = pre_field.clone()
        tar_field_masked = tar_field.clone()
        tar_field_obs_masked = tar_field_obs.clone()

        if mask_out_of_range: # fill input with 0 where field_mask is False
            pre_field_masked = torch.masked_fill(input=pre_field_masked, mask=~field_mask, value=0) 
            tar_field_masked = torch.masked_fill(input=tar_field_masked, mask=~field_mask, value=0) 
            tar_field_obs_masked = torch.masked_fill(input=tar_field_obs_masked, mask=~field_mask, value=0)  
        
        # type 1 loss
        loss_field = F.mse_loss(pre_field_masked, tar_field_masked)
        loss_field_channel_wise = F.mse_loss(pre_field_masked, tar_field_masked, reduction="none")
        loss_field_channel_wise = torch.mean(loss_field_channel_wise, dim=(0, 2, 3))

        # type 2 loss
        loss_field_obs = F.mse_loss(pre_field_masked, tar_field_obs_masked)

        # type 3 loss - use fresh masks for obs loss
        pre_field_obs_masked = torch.masked_fill(input=pre_field.clone(), mask=~obs_tar_mask, value=0)  # fill input with 0 where obs_tar_mask is False.
        tar_obs_masked = torch.masked_fill(input=tar_obs.clone(), mask=~obs_tar_mask, value=0)
        loss_obs = F.mse_loss(pre_field_obs_masked, tar_obs_masked)
        loss_obs_channel_wise = F.mse_loss(pre_field_obs_masked, tar_obs_masked, reduction="none")
        loss_obs_channel_wise = torch.mean(loss_obs_channel_wise, dim=(0, 2, 3))

        return {"loss_field": loss_field,
                "loss_field_channel_wise": loss_field_channel_wise,
                "loss_obs": loss_obs,
                "loss_obs_channel_wise": loss_obs_channel_wise,
                "loss_field_obs": loss_field_obs}
    
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
        loss_field = 0.0
        loss_obs = 0.0
        loss_field_obs = 0.0
        loss_field_channel_wise = torch.zeros(len(self.params.target_vars), device=self.device, dtype=float)
        loss_obs_channel_wise = torch.zeros(len(self.params.target_vars), device=self.device, dtype=float)
        
        self.model.train()
        for i, data in enumerate(self.train_data_loader):
            self.iters += 1
            steps_in_one_epoch += 1
            data_start = time.time()

            # No EncDec_two_encoder switch here (DNE anyway)
            (inp,
             target_field,
             target_obs,
             target_field_obs,
             inp_hrrr,
             _,
             _,
             field_mask,
             obs_tar_mask) = data

            data_time += time.time() - data_start
            tr_start = time.time()

            self.optimizer.zero_grad()
            with amp.autocast(device_type=self.device.type):
                inp = inp.to(self.device, dtype=torch.float)
                inp_hrrr = inp_hrrr.to(self.device, dtype=torch.float)
                target_field = target_field.to(self.device, dtype=torch.float)
                target_obs = target_obs.to(self.device, dtype=torch.float)
                target_field_obs = target_field_obs.to(self.device, dtype=torch.float)
                field_mask = torch.as_tensor(field_mask, dtype=torch.bool, device=self.device)
                obs_tar_mask = torch.as_tensor(obs_tar_mask, dtype=torch.bool, device=self.device)

                # No EncDec_two_encoder code here either
                gen = self.model(inp)
                gen.to(self.device, dtype=torch.float)

                loss = self.loss_function(pre_field=gen,
                                          tar_field=target_field,
                                          tar_obs=target_obs,
                                          tar_field_obs=target_field_obs,
                                          field_mask=field_mask,
                                          obs_tar_mask=obs_tar_mask)
                
                loss_field += loss["loss_field"].detach().item()
                loss_obs += loss["loss_obs"].detach().item()
                loss_field_obs += loss["loss_field_obs"].detach().item()
                loss_field_channel_wise += loss["loss_field_channel_wise"].detach()
                loss_obs_channel_wise += loss["loss_obs_channel_wise"].detach()

                if self.params.target == "obs": # target = sparse observations only
                    if self.params.enable_amp:
                        self.gscaler.scale(loss["loss_obs"]).backward()
                        self.gscaler.step(self.optimizer)
                    else:
                        loss["loss_obs"].backward()
                        self.optimizer.step()
                if self.params.target == "analysis": # target = gridded fields only, no obs
                    if self.params.enable_amp:
                        self.gscaler.scale(loss["loss_field"]).backward()
                        self.gscaler.step(self.optimizer)
                    else:
                        loss["loss_field"].backward()
                        self.optimizer.step()
                if self.params.target == "analysis_obs": # target: gridded fields + sparse observations
                    if self.params.enable_amp:
                        self.gscaler.scale(loss["loss_field_obs"]).backward()
                        self.gscaler.step(self.optimizer)
                    else:
                        loss["loss_field_obs"].backward()
                        self.optimizer.step()

                if self.params.enable_amp:
                    self.gscaler.update()

                tr_time += time.time() - tr_start

        logs = {"loss_field": loss_field / steps_in_one_epoch,
                "loss_obs": loss_obs / steps_in_one_epoch,
                "loss_field_obs": loss_field_obs / steps_in_one_epoch}
        
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
        if self.params.log_to_screen and self.params.world_rank==0: #only print once
            print("Validating...")
        self.model.eval()

        valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)
        valid_loss_field = valid_buff[0].view(-1)
        valid_loss_obs = valid_buff[1].view(-1)
        valid_loss_field_obs = valid_buff[2].view(-1)
        valid_steps = valid_buff[3].view(-1)

        valid_start = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader):
                # No plotting code here
                # No EncDec_two_encoder code here
                (inp,
                 target_field,
                 target_obs,
                 target_field_obs,
                 inp_hrrr,
                 _,
                 _,
                 field_mask,
                 obs_tar_mask) = data
                
                inp = inp.to(self.device, dtype=torch.float)
                inp_hrrr = inp_hrrr.to(self.device, dtype=torch.float)
                target_field = target_field.to(self.device, dtype=torch.float)
                target_obs = target_obs.to(self.device, dtype=torch.float)
                target_field_obs = target_field_obs.to(self.device, dtype=torch.float)
                field_mask = field_mask.to(self.device, dtype=torch.bool)
                obs_tar_mask = obs_tar_mask.to(self.device, dtype=torch.bool)

                # No EncDec_two_encoder code here either
                gen = self.model(inp)
                gen.to(self.device, dtype=torch.float)

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
        
        if dist.is_initialized():
            dist.all_reduce(valid_buff)

        # divide by number of steps
        valid_buff[0:3] = valid_buff[0:3] / valid_buff[3]
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        
        logs = {"valid_loss_field": valid_buff_cpu[0],
                "valid_loss_obs": valid_buff_cpu[1],
                "valid_loss_field_obs": valid_buff_cpu[2]}
        
        valid_time = time.time() - valid_start

        return valid_time, logs

    ##########

    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model

        print(f"Saving model to {checkpoint_path}")
        torch.save({"iters": self.iters,
                    "epoch": self.epoch,
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

        best_train_loss = 1.0e6

        for epoch in range(self.startEpoch, self.params.max_epochs):
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

            # validate one epoch
            if (epoch != 0) and (epoch % self.params.valid_frequency == 0):
                valid_time, valid_logs = self.validate_one_epoch()
                
                if self.params.log_to_screen and self.params.world_rank==0: #only print once
                    print(f"Valid time={valid_time: .2f} seconds")
                    print(f"Valid loss={valid_logs['valid_loss_field']}")

            # LR scheduler
            # (2026-06-05) Does having this operate only on validated epochs cause issues? 
                # If only every 5th epoch is validated and patience = 20, does that mean 100 epochs to reduce LR when it should be 20? Test this later
            # (2026-06-11) Changing this to operate every epoch, not just per validation epoch
            if self.params.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(train_logs["loss_field"]) #valid_logs["valid_loss_field"])

            # Save model checkpoint
            if (self.params.world_rank == 0 and epoch % self.params.save_model_freq == 0 and self.params.save_checkpoint):
                self.save_checkpoint(self.params.checkpoint_path)

            # If model is the best yet (regardless of save_model_freq), save to the best checkpoint path
            # !! This will wipe out the previous best model !! Needs modification for that case
            if (self.params.world_rank == 0 and self.params.save_checkpoint):
                if train_logs["loss_field"] <= best_train_loss:
                    print(f"Loss improved from {best_train_loss} to {train_logs["loss_field"]}")
                    best_train_loss = train_logs["loss_field"]
                    self.save_checkpoint(self.params.best_checkpoint_path)
        
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

    if params.log_to_screen and params.world_rank == 0:
        print("------ PARAMETER VALUES ------")
        for key, val in params.items():
            print(f"{key}: {val}")
        print("------------------------------")

    trainer = Trainer(params)
    trainer.train()

    dist.destroy_process_group()