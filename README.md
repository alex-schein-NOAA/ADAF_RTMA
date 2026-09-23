## ADAF-RTMA
This repository contains the code needed to train and run the ADAF-RTMA model, an updated version of the ADAF model (https://github.com/microsoft/ADAF). This version of the model runs on the full RTMA domain at 2.5 km resolution, and has a modified model architecture to enable convolutional layers before and after the attention mechanism, allowing station observations to spread their influence over a wider domain.

### Disclaimers
The code here is intended to run on the Ursa supercomputer, and all filepaths in the code and this readme refer to that system.
Due to Github file size limits, the static terrain file must be copied directly from Ursa for dataset creation:

    cp /scratch3/BMC/wrfruc/aschein/ADAF_RTMA/data_preparation_ges/RTMA_TOPO_2p5km.nc [your /data_preparation_ges/ path]
    
And if you want to perform inference on an already-trained model,

    cp /scratch3/BMC/wrfruc/aschein/ADAF_RTMA/training_runs/[desired training run]

As of 2026/09/23, there should be an active data repository on `/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data_ges_goes`, so don't create a new dataset!

### Setup workflow

 1. Create a new python environment with the provided `ADAF_environment.yml` file. The training shell scripts will use the resulting filepath, so make note of the install location. **EXTREMELY IMPORTANT:** If you're making the environment on an HPC login node without a GPU, the install will not see cuda and will default to CPU-only install. To fix this, you MUST first get the CUDA version on a GPU compute node with "nvidia-smi" (should be 13.1 or similar) and then on the login node, run "export CONDA_OVERRIDE_CUDA=13.1" (or your CUDA version). Again, do this BEFORE any conda installation!!
    * Place the `zz_cuda_headers.sh` file into the resulting `../ADAF_environment/etc/conda/activate.d/` 
 2. Filepaths will need some changing:
     * `config/params_lowres_ges_goes.yaml`
         * Change `checkpoint_path` and `best_checkpoint_path` to refer to your directory. The data paths can be left as-is.
      * `train_ges_goes.py` 
          * Change `conda_ptxas_path` and the `os.environ["PATH"]` lines to refer to your environment
      * `train_launcher_sbatch_ges_goes.sh` and `train_resume_launcher_ges_goes_sbatch.sh`
          * Change `source` to your environment's source. 
          * Change `CHECKPOINT_DIR` to your directory. 
          * Change `ENV_BIN` to your environment's bin.
          * Change the `torch.distributed.run` and `train_ges_goes.py` calls to your directory; in `train_resume_launcher_ges_goes.sh`, also change the numbers to match the model you want to resume training.
      * `data_preparation_ges`
          * Modify filepaths as needed - though end users shouldn't have to mess with creating a new dataset

### Model training (and resuming training)
Once the filepath changes above are made, you should only need to run  `sbatch train_launcher_sbatch_ges_goes.sh`  to get a training job going. The header of this file contains very important parameters for job length, project account, and more, so make sure to modify it!

To resume training on a finished job, open `train_resume_launcher_ges_goes_sbatch.sh` and modify the `MODEL_NUMBER_TO_LOAD` and `RESUME_NUMBER_TO_LOAD` arguments (the latter if you're resuming a job that was itself a resumed job). Make sure you're using the correct `PREVIOUS_CHECKPOINT_DIR` line as well.

### Model inference
Once you have a trained model, you can perform inference on a single hour with `apply_lowres_to_ges_goes.ipynb`; change the `model_number` and `resume_number` arguments as needed. Run this on a GPU node; the inference takes ~4 seconds on a GPU but ~90 seconds on CPU!

-----------------------------------------------------
#### Known issues
(2026/09/23) The filenaming scheme is currently a mess and will be cleaned up in the future. 