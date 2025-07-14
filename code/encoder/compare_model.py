import os
import argparse
import numpy as np
import json
import torch
from torch import nn
import math
import torch.nn.functional as nnf
import copy


def interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    # Interpolate the position embeddings without changing the first row (class token)
    patch_pos_embed = nnf.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)


checkpoint_path1 = '/mnt/madehua/model/checkpoints/food_vitl14/model_0006365.rank_0.pth'
checkpoint_path2 = '/mnt/madehua/model/checkpoints/food_vitl14/model_0063659.rank_5.pth'
checkpoint_path3 = '/mnt/madehua/model/checkpoints/food_vitl14/model_0006365.rank_2.pth'
checkpoint_path4 = '/mnt/madehua/model/checkpoints/food_vitl14/model_0006365.rank_3.pth'
checkpoint_path5 = '/mnt/madehua/model/checkpoints/food_vitl14/model_0006365.rank_4.pth'
checkpoint_path6 = '/mnt/madehua/model/checkpoints/food_vitl14/model_0006365.rank_5.pth'

# ... existing imports and code ...

# Load the original model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
input_tensor = model.pos_embed
tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
pos_embed = nn.Parameter(torch.zeros(1, 257))
pos_embed.data = tensor_corr_shape
model.pos_embed = pos_embed
# Save the original model's parameters
original_state_dict = copy.deepcopy(model.state_dict())

def load_and_compare_weights(model, checkpoint_paths):

    for i, checkpoint_path in enumerate(checkpoint_paths, start=1):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Adjust keys by removing 'student.backbone.' or 'teacher.backbone.' prefix
        new_state_dict = {}
        for k, v in checkpoint["model"].items():
            if k.startswith('student.backbone.'):
                new_key = k.replace('student.backbone.', '')
                new_state_dict[new_key] = v
            # elif k.startswith('teacher.backbone.'):
            #     new_key = k.replace('teacher.backbone.', '')
            else:
                new_key = k

        
        # Load the adjusted state dictionary
        model.load_state_dict(new_state_dict, strict=True)
        
        idx = 0
        # Compare weights
        for key in original_state_dict:
            if key in new_state_dict:
                if not torch.equal(original_state_dict[key], new_state_dict[key]):
                    if idx == 0:
                        print(new_state_dict[key])
                        print(f"Difference found in {key} for checkpoint {i}")
                        idx =1
                else:
                    if idx == 0:
                        print(original_state_dict[key])
                        idx =1
                        print(f"No difference in {key} for checkpoint {i}")
            else:
                print(f"{key} not found in checkpoint {i}")

# List of checkpoint paths
checkpoint_paths = [
    checkpoint_path1,
    checkpoint_path2,
    checkpoint_path3,
    checkpoint_path4,
    checkpoint_path5,
    checkpoint_path6
]

# Call the function to load and compare weights
load_and_compare_weights(model, checkpoint_paths)

