import argparse
from collections.abc import Mapping

#########################

def set_user_params(parser):
    """
    Function to let the user override the parameters in an ADAF config file. 
    Designed to be called before the YParams object is instantiated in the main code; any arguments inputted here will override those config file parameters
    
    Input: instantiated argparse.ArgumentParser() object
    Output: args to be used for YParams call
    """
    # TRAINING PARAMETERS
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_data_workers', type=int, default=None)
    parser.add_argument('--save_checkpoint', type=str, default=None)
    parser.add_argument('--save_model_freq', type=int, default=None)
    parser.add_argument('--valid_frequency', type=int, default=None)
    parser.add_argument('--optimizer_type', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--scheduler_patience', type=int, default=None)
    parser.add_argument('--lr_reduce_factor', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--world_rank', type=int, default=None)

    # DATA PATHS AND SPECIFICATIONS
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--valid_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--experiment_dir', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--best_checkpoint_path', type=str, default=None)
    parser.add_argument('--inp_hrrr_vars', type=str, nargs='+', default=None)
    parser.add_argument('--inp_obs_vars', type=str, nargs='+', default=None)
    parser.add_argument('--field_tar_vars', type=str, nargs='+', default=None)
    parser.add_argument('--target_vars', type=str, nargs='+', default=None)
    parser.add_argument('--obs_time_window', type=int, default=None)

    # MODEL ARCHITECTURE
    parser.add_argument('--upscale', type=int, default=None)
    parser.add_argument('--in_chans', type=int, default=None)
    parser.add_argument('--out_chans', type=int, default=None)
    parser.add_argument('--img_size_x', type=int, default=None)
    parser.add_argument('--img_size_y', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None)
    parser.add_argument('--num_feat', type=int, default=None)
    parser.add_argument('--drop_rate', type=float, default=None)
    parser.add_argument('--drop_path_rate', type=float, default=None)
    parser.add_argument('--attn_drop_rate', type=float, default=None)
    parser.add_argument('--ape', type=str, default=None)
    parser.add_argument('--patch_norm', type=str, default=None)
    parser.add_argument('--use_checkpoint', type=str, default=None)
    parser.add_argument('--resi_connection', type=str, default=None)
    parser.add_argument('--qkv_bias', type=str, default=None)
    parser.add_argument('--qk_scale', type=float, default=None)
    parser.add_argument('--img_range', type=float, default=None)
    parser.add_argument('--depths', type=int, nargs='+', default=None)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--num_heads', type=int, nargs='+', default=None)
    parser.add_argument('--mlp_ratio', type=int, default=None)
    parser.add_argument('--upsampler', type=str, default=None)

    # TRAINING SPECIFICS
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--hold_out_obs', type=str, default=None)
    parser.add_argument('--hold_out_obs_ratio', type=float, default=None)
    parser.add_argument('--learn_residual', type=str, default=None)
    parser.add_argument('--obs_mask_seed', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resuming', type=str, default=None)
    parser.add_argument('--enable_amp', type=str, default=None)
    parser.add_argument('--log_to_screen', type=str, default=None)

    args = parser.parse_args()

    return args

####

def to_builtin(value):
    """Convert ruamel container/scalar types into plain Python builtins."""
    if isinstance(value, Mapping):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_builtin(v) for v in value)

    # Preserve exact builtin scalar types
    if type(value) in (bool, int, float, str) or value is None:
        return value

    # Coerce scalar subclasses to builtins (can cause issues with saved checkpoints if not sanitized)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)

    return value