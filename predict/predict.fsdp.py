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
torch.autograd.set_detect_anomaly(True)    # [WARNING] Making it True may throw errors when using bfloat16

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

print("Hello world!")