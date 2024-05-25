#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn            as nn
import torch.nn.functional as F

from transformers import ViTMAEForPreTraining

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Optional

# ----------------------------------------------------------------------- #
#  Helper
# ----------------------------------------------------------------------- #
def update_num_channels(model, new_channels=1):
    for child_name, child in model.named_children():
        if hasattr(child, 'num_channels'):
            print(f"Updating {child_name} num_channels from {child.num_channels} to {new_channels}")
            child.num_channels = new_channels

        # Recursively update submodules
        update_num_channels(child, new_channels)


# ----------------------------------------------------------------------- #
#  Model
# ----------------------------------------------------------------------- #
@dataclass
class AdaptedViTMAEForPreTrainingConfig:
    model_name: str = "facebook/vit-mae-base"

class AdaptedViTMAEForPreTraining(nn.Module):
    IMG_SIZE           = 224
    NUM_RGB_CHANNEL    = 3
    DECODE_IN_FEATURES = 512

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model  = AdaptedViTMAEForPreTraining.adapt_pretrained_model(self.config.model_name)

        vit_mae_model_config_dict = {
            "facebook/vit-mae-base"  : { "emb_size" : 768,  "win_size" : 16 },
            "facebook/vit-mae-large" : { "emb_size" : 1024, "win_size" : 16 },
            "facebook/vit-mae-huge"  : { "emb_size" : 1280, "win_size" : 14 },
        }
        self.patch_size = vit_mae_model_config_dict[self.config.model_name]["win_size"]

    @staticmethod
    def adapt_pretrained_model(model_name):
        # -- Which pretrained model is in use
        vit_mae_model_config_dict = {
            "facebook/vit-mae-base"  : { "emb_size" : 768,  "win_size" : 16 },
            "facebook/vit-mae-large" : { "emb_size" : 1024, "win_size" : 16 },
            "facebook/vit-mae-huge"  : { "emb_size" : 1280, "win_size" : 14 },
        }
        vit_mae_model_config = vit_mae_model_config_dict[model_name]
        emb_size = vit_mae_model_config['emb_size']
        win_size = vit_mae_model_config['win_size']
        grid_size = int(AdaptedViTMAEForPreTraining.IMG_SIZE / win_size)

        # -- Initialize the pretrained model
        model = ViTMAEForPreTraining.from_pretrained(model_name)

        # -- Adapt
        # --- Update channel number
        update_num_channels(model)
        model.config.num_channels = 1

        # --- Adapt to one channel input
        avg_weight_patch_embd = model.vit.embeddings.patch_embeddings.projection.weight.data.mean(dim = 1, keepdim = True)
        model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels  = 1,
            out_channels = emb_size,
            kernel_size  =(win_size, win_size),
            stride       =(win_size, win_size),
        )
        model.vit.embeddings.patch_embeddings.projection.weight.data = avg_weight_patch_embd

        # --- Adapt to correct output
        avg_weight_decoder_pred = model.decoder.decoder_pred.weight.data.view(
            AdaptedViTMAEForPreTraining.NUM_RGB_CHANNEL, win_size, win_size, -1
        ).mean(dim = 0).view(win_size * win_size, -1)
        model.decoder.decoder_pred = nn.Linear(
            in_features  = AdaptedViTMAEForPreTraining.DECODE_IN_FEATURES,
            out_features = win_size*win_size,
            bias         = True)
        model.decoder.decoder_pred.weight.data = avg_weight_decoder_pred
        return model

    def forward(self, x):
        return self.model(x)
    
    def unpatchify(self, patchified_pixel_values):
        # Adapted from https://github.com/huggingface/transformers/blob/bdb9106f247fca48a71eb384be25dbbd29b065a8/src/transformers/models/vit_mae/modeling_vit_mae.py#L1023
        num_channels = 1
        patch_size = self.win_size
        original_image_size = (AdaptedViTMAEForPreTraining.IMG_SIZE, AdaptedViTMAEForPreTraining.IMG_SIZE)
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )
        maxie_patch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            maxie_patch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            maxie_patch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values
