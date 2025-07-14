
import os
import argparse
import numpy as np
import json
import torch
from torch import nn
import math
import torch.nn.functional as nnf
import copy
from code.utils import interpolate_pos_encoding



torch.hub.set_dir("/map-vepfs/huggingface")

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
print(1)

import torch
import copy

def process_checkpoints(checkpoint_path, student_checkpoint_path, teacher_checkpoint_path):
    input_tensor = model.pos_embed
    tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed
    original_state_dict = copy.deepcopy(model.state_dict())
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Initialize dictionaries for student and teacher models
    new_state_teacher_dict = {}
    new_state_student_dict = {}

    # Separate the state dicts for student and teacher
    for k, v in checkpoint["model"].items():
        if k.startswith('student.backbone.'):
            new_key = k.replace('student.backbone.', '')
            new_state_student_dict[new_key] = v
        elif k.startswith('teacher.backbone.'):
            new_key = k.replace('teacher.backbone.', '')
            new_state_teacher_dict[new_key] = v

    # Load and save student model state dict
    model.load_state_dict(new_state_student_dict, strict=True)
    student_model_state_dict = copy.deepcopy(model.state_dict())
    torch.save(model.state_dict(), student_checkpoint_path)

    # Load and save teacher model state dict
    model.load_state_dict(new_state_teacher_dict, strict=True)
    teacher_model_state_dict = copy.deepcopy(model.state_dict())
    torch.save(model.state_dict(), teacher_checkpoint_path)

    # Compare original and student model state dicts
    for key in original_state_dict:
        if key in student_model_state_dict:
            if not torch.equal(original_state_dict[key], student_model_state_dict[key]):
                print(f"Difference found in {key}")
            else:
                print(f"No difference in {key}")
        else:
            print(f"{key} not found in student model")

    # Compare student and teacher model state dicts
    for key in student_model_state_dict:
        if key in teacher_model_state_dict:
            if not torch.equal(student_model_state_dict[key], teacher_model_state_dict[key]):
                print(f"Difference found in {key}")
            else:
                print(f"No difference in {key}")
        else:
            print(f"{key} not found in teacher model")

def process_checkpoints_cls(checkpoint_path, targe_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_state_dict = {k.replace('base_model.', ''): v for k, v in checkpoint.items() if k not in ['classifier.weight', 'classifier.bias']}

    # Load the adjusted state dictionary
    model.load_state_dict(new_state_dict, strict=True)
    input_tensor = model.pos_embed
    tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed
    torch.save(model.state_dict(), targe_path)

# Example usage
checkpoint_path = '/map-vepfs/dehua/code/dinov2/model_0160709.rank_0.pth'
student_checkpoint_path = '/map-vepfs/dehua/model/dinov2/model_0160709_student_model.pth'
teacher_checkpoint_path = '/map-vepfs/dehua/model/dinov2/model_0160709_teacher_model.pth'

process_checkpoints(checkpoint_path, student_checkpoint_path, teacher_checkpoint_path)

# checkpoint_path = '/mnt/madehua/model/checkpoints/dinov2/food2k/model_epoch_13.pth'
# target_path = '/mnt/madehua/model/checkpoints/dinov2/food2k/model_epoch_13_cls.pth'
# process_checkpoints_cls(checkpoint_path, target_path)
