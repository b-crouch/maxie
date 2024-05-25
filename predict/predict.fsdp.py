#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- OLCF specific imports
from maxie.plugins.olcf import init_dist_env_on_summit

# -- Basic imports
import os
import yaml
import tqdm
import signal
import argparse
import logging
import traceback

from functools  import partial
from contextlib import nullcontext
from datetime   import timedelta

# -- maxie specific imports
from maxie.datasets.ipc_segmented_dataset_dist import IPCDistributedSegmentedDatasetConfig, IPCDistributedSegmentedDataset, IPCDatasetConfig, IPCDataset
from maxie.modeling.adapted_mae import AdaptedViTMAEForPreTrainingConfig, AdaptedViTMAEForPreTraining
from maxie.utils.logger         import init_logger
from maxie.utils.seed           import set_seed
from maxie.utils.misc           import is_action_due
from maxie.lr_scheduler         import CosineLRScheduler
from maxie.perf                 import Timer
from maxie.tensor_transforms    import Pad, DownscaleLocalMean, RandomPatch, RandomRotate, RandomShift, Patchify, Norm
from maxie.utils_fsdp           import (
    MemoryMaximizer,
    verify_bfloat_support,
    TrainingStateDictConfig,
    FullStateDictCheckpointConfig,
    FullStateDictCheckpoint,
    broadcast_dict,
)

# -- Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim

# -- Fully Sharded Data Parallel (FSDP)
# --- Main
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)

# --- Policy wrapper
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
)
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAELayer,
    ## ViTMAEAttention,
    ## ViTMAESelfAttention,
)
from packaging import version

# --- Scaler for float16
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# --- Activation checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

# --- Distributed library
import torch.distributed as dist

# -- Debug
torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using bfloat16

# -- Reporting specific imports
import colorama
colorama.init(autoreset=True)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  COMMAND LINE INTERFACE
# ----------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Load training configuration from a YAML file to a dictionary.")
parser.add_argument("yaml_file", help="Path to the YAML file")
args = parser.parse_args()

# ----------------------------------------------------------------------- #
#  CONFIGURATION
# ----------------------------------------------------------------------- #
# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# -- Checkpoint
chkpt_config        = config.get("checkpoint")
dir_root_chkpt      = chkpt_config.get("directory")
fl_chkpt_prefix     = chkpt_config.get("filename_prefix")
dir_chkpt_prefix    = chkpt_config.get("dir_chkpt_prefix")
path_chkpt_prev     = chkpt_config.get("path_chkpt_prev")
chkpt_saving_period = chkpt_config.get("chkpt_saving_period")

# -- Dataset
dataset_config       = config.get("dataset")
path_predict_json      = dataset_config.get("path_predict")
batch_size           = dataset_config.get("batch_size")
num_workers          = dataset_config.get("num_workers")
seg_size             = dataset_config.get("seg_size")
server_address       = dataset_config.get("server_address")
transforms_config    = dataset_config.get("transforms")
num_patch            = transforms_config.get("num_patch")
size_patch           = transforms_config.get("size_patch")
frac_shift_max       = transforms_config.get("frac_shift_max")
angle_max            = transforms_config.get("angle_max")
var_size_patch       = transforms_config.get("var_size_patch")
downscale_factors    = transforms_config.get("downscale_factors")
H_pad                = transforms_config.get("H_pad")
W_pad                = transforms_config.get("W_pad")
patch_size           = transforms_config.get("patch_size")
stride               = transforms_config.get("stride")
detector_norm_params = transforms_config.get("norm")

# -- Model
model_params = config.get("model")
model_name   = model_params.get("name")

# -- Distributed envs
dist_config            = config.get("dist")
dist_backend           = dist_config.get("backend")
uses_unique_world_seed = dist_config.get("uses_unique_world_seed")
dist_dtype             = dist_config.get("dtype")

# -- Logging
logging_config = config.get("logging")
drc_log       = logging_config.get("directory")
fl_log_prefix = logging_config.get("filename_prefix")

misc_config = config.get("misc")
num_gpus             = misc_config.get("num_gpus")
compiles_model       = misc_config.get("compiles_model")
data_dump_on         = misc_config.get("data_dump_on", False)

# -- Saving
save_filepath = config.get("predictions").get("save_filepath")
save_tensor = config.get("predictions").get("save_tensor")
save_img = config.get("predictions").get("save_img")

# ----------------------------------------------------------------------- #
#  MISC FEATURES
# ----------------------------------------------------------------------- #
def signal_handler(signal, frame):
    # Emit Ctrl+C like signal which is then caught by our try/except block
    raise KeyboardInterrupt

# Register the signal handler
signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ----------------------------------------------------------------------- #
#  DIST SETUP
# ----------------------------------------------------------------------- #
# -- DIST init
# --- OLCF specific env
# torchrun doesn't work well on OLCF.  Refer to https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html#torchrun
# Thanks to the suggestion by @frobnitzem
torchrun_exists = int(os.environ.get("RANK", -1)) != -1
if not torchrun_exists: init_dist_env_on_summit()

# --- Initialize distributed environment
uses_dist = int(os.environ.get("RANK", -1)) != -1
if uses_dist:
    dist_rank       = int(os.environ["RANK"      ])
    dist_local_rank = int(os.environ["LOCAL_RANK"])
    dist_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = dist_backend,
                            rank        = dist_rank,
                            world_size  = dist_world_size,
                            timeout     = timedelta(seconds=1800),
                            init_method = "env://",)
    print(f"RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")
else:
    dist_rank       = 0
    dist_local_rank = 0
    dist_world_size = 1
    print(f"NO FSDP is used.  RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")

# --- Set up GPU device
gpu_idx = dist_local_rank % torch.cuda.device_count()    # dist_local_rank is node-centric, whereas torch.cuda.device_count() is resource-centeric (on LSF)
device = f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = dist_rank if uses_unique_world_seed else 0

# --- Set up performance utility
memmax = MemoryMaximizer() if dist_local_rank == 0 else None


# ----------------------------------------------------------------------- #
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP policy
# --- Mixed precision
mixed_precision = None
if verify_bfloat_support:
    dist_dtype = 'bfloat16'
    mixed_precision = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.bfloat16,
        buffer_dtype = torch.bfloat16,
    )
else:
    dist_dtype = 'float16'
    mixed_precision = MixedPrecision(
        param_dtype  = torch.float16,
        reduce_dtype = torch.float16,
        buffer_dtype = torch.float16,
    )

# --- Sharding strategy
sharding_strategy = ShardingStrategy.FULL_SHARD

# --- Wrapping strategy
# ---- Use built-in transformer wrap policy
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        ViTMAELayer,
        ## ViTMAEAttention,
        ## ViTMAESelfAttention,
    },
)

# --- Activation checkpointing
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu  = False,
    checkpoint_impl = CheckpointImpl.NO_REENTRANT,
)

# --- Backward prefetch policy
backward_prefetch = BackwardPrefetch.BACKWARD_PRE


# ----------------------------------------------------------------------- #
#  LOGGING
# ----------------------------------------------------------------------- #
timestamp = None
if dist_rank == 0:
    # Fetch the current timestamp...
    timestamp = init_logger(fl_prefix = fl_log_prefix, drc_log = drc_log, returns_timestamp = True)

    # Convert dictionary to yaml formatted string...
    config_yaml = yaml.dump(config)

    # Log the config...
    logger.info(config_yaml)
timestamp = broadcast_dict(dict(timestamp=timestamp), src = 0, device = device).get('timestamp')

# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring dataset...')
# -- Seeding
base_seed  = 0
world_seed = base_seed + seed_offset
set_seed(world_seed)

# -- Set up transformation
transforms = (
    Norm(detector_norm_params),
    Pad(H_pad, W_pad),
    ## DownscaleLocalMean(factors = downscale_factors),
    ## RandomPatch(num_patch = num_patch, H_patch = size_patch, W_patch = size_patch, var_H_patch = var_size_patch, var_W_patch = var_size_patch, returns_mask = False),
    ## RandomRotate(angle_max),
    ## RandomShift(frac_y_shift_max=frac_shift_max, frac_x_shift_max=frac_shift_max),
    Patchify(patch_size, stride),
)

# -- Initialize dataset
ipc_dataset_config = IPCDistributedSegmentedDatasetConfig(
    path_json             = path_predict_json,
    seg_size              = seg_size,
    world_size            = dist_world_size,
    transforms            = transforms,
    is_perf               = True,
    server_address        = tuple(server_address),
    loads_segment_in_init = False,
)
dataset = IPCDistributedSegmentedDataset(ipc_dataset_config)

# -- Custom collate to merge patch and batch dimension using concatenation
## custom_collate = lambda batch: torch.cat(batch, dim=0)  # batch of [N, C, H, W] -> [B * N, C, H, W]
def custom_collate(batch):
    batch_filtered = [x for x in batch if x is not None]
    return torch.cat(batch_filtered, dim=0) if len(batch_filtered) else None

# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring model...')
# -- Config the model
model_config = AdaptedViTMAEForPreTrainingConfig(model_name = model_name)
model = AdaptedViTMAEForPreTraining(model_config)

# !! Make all params trainable, a workaround for pytorch 2.0.1
torch_version = torch.__version__
torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
if version.parse(torch_version) <= version.parse("2.0.1"):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

if dist_rank == 0:
    print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# -- Mixed precision
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=mixed_precision_dtype)

# -- Compile the model
if compiles_model:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# -- CHECKPOINT (FULL STATE DICT)
print(f'[RANK {dist_rank}] Confguring model checkpoint...')
chkpt_config = FullStateDictCheckpointConfig(
    model           = model,
    optimizer       = None,
    lr_scheduler    = None,
    training_state  = None,
    rank            = dist_rank,
    device          = device,
    path_checkpoint = path_chkpt_prev,
)
checkpointer = FullStateDictCheckpoint(config = chkpt_config)
from_resume = path_chkpt_prev is not None
if from_resume:
    if isinstance(checkpointer, FullStateDictCheckpoint):
        # Model is loaded
        checkpointer.pre_fsdp_load()

# -- Wrapping the model in FSDP...
if uses_dist:
    # Convert BatchNorm to SyncBatchNorm...
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using FSDP...
    model = FSDP(
        model,
        auto_wrap_policy  = auto_wrap_policy,
        mixed_precision   = mixed_precision,
        backward_prefetch = backward_prefetch,
        forward_prefetch  = True,
        sharding_strategy = sharding_strategy,
        limit_all_gathers = True,
        use_orig_params   = True,
        device_id         = device,
    )

    sharded_param_count = sum(p.numel() for p in model.module.parameters())
    print(f"RANK {dist_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()

dataset.reset()
dataset.set_start_idx(0)
dataloader = torch.utils.data.DataLoader(dataset, collate_fn=custom_collate)
model.eval()

for i, tensor in enumerate(dataloader):
    if tensor is not None:
        input = tensor.to(device, non_blocking = True)
        output = model(input)

        # Denormalize input and output images
        exp, run, event, detector_name = dataset.get_info(i)
        norm = transforms[0]

        raw_image = torch.einsum('nchw->nhwc', tensor).cpu()
        raw_image = norm.invert(raw_image, detector_name)
        
        generated_image = model.unpatchify(output.logits)
        generated_image = torch.einsum('nchw->nhwc', generated_image).detach().cpu()
        generated_image = norm.invert(generated_image, detector_name)

        mask = output.mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2)
        mask = model.unpatchify(mask)
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        if save_tensor:
            path_raw_save = os.path.join(save_filepath, f"{exp}_r{run}_e{event}_raw_image.pt")
            torch.save(raw_image, path_raw_save)
            path_generated_save = os.path.join(save_filepath, f"{exp}_r{run}_e{event}_gen_image.pt") 
            torch.save(generated_image, path_generated_save)
            path_mask_save = os.path.join(save_filepath, f"{exp}_r{run}_e{event}_mask.pt") 
            torch.save(mask, path_mask_save)

        # detector = dataset.get_detector(i)
        # # Assume we only pass in a single event for image generation, so B = 1
        # N, C, H, W = image_tensor.shape
        # image_tensor = image_tensor.view(1, N, C, H, W)
        # # Apply inverse transforms to recover image
        # for trans in transforms[::-1]:
        #      image_tensor = trans.invert(image_tensor, detector_name=detector)
        