from __future__ import absolute_import, print_function

import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

""" Input transform implementation to tackle transformation robustness."""
np.random.seed(0)
torch.manual_seed(0)


def standard_transforms(tensor) -> torch.Tensor:
    jitter1 = 16
    jitter2 = 16
    # normalization = transforms.Normalize([0.485, 0.456, 0.406],
    #                                      [0.229, 0.224, 0.225])
    normalization = transforms.Normalize([0.5162, 0.4644, 0.3975],
                                         [0.2724, 0.2640, 0.2574])
    tensor = normalization(tensor)
    tensor = F.pad(tensor, padding=[32], fill=0, padding_mode="symmetric")
    # Affine using a random choice of number in range
    dx = np.random.choice(jitter1)
    dy = np.random.choice(jitter1)
    tensor = F.affine(tensor, angle=0, translate=[dx, dy], scale=1, shear=[0])
    # Apply scaling to image tensor, through a random choice of scales in
    # range 0.8 to 1.2 (slightly more flexible range than that of Lucid.)
    scales = np.arange(0.7, 1.3, 0.1)
    scale_selected = np.random.choice(scales)
    # Will have to work on the scaling a bit since it is a little bizarre now
    # Lucid upsamples scaled images, but it uses pad instead of affine and
    # the upsampling operation seems a bit... redundant?
    tensor = F.affine(tensor, angle=0, translate=[0, 0],
                      scale=scale_selected, shear=[0])
    # Rotate the image with weighted degrees selection.
    # Reminder!!! Angles need to be int, so get the item from numpy.
    angles = np.arange(0, 90, 5)
    probability_array = np.linspace(1, 0, 18)
    probability_array[0] += 10  # modify 0 angle to have higher probability
    probabilities = probability_array / sum(probability_array)
    selected_angle = np.random.choice(angles, p=probabilities)
    tensor = F.rotate(tensor, angle=selected_angle.item())
    # Affine using a random choice of number in range
    # dx = np.random.choice(jitter2)
    # dy = np.random.choice(jitter2)
    # tensor = F.affine(tensor, angle=0, translate=[dx, dy], scale=1, shear=[0])
    # Apply blurring on each image. Seems to discourage noise further,
    # but the result is too subtle, or unwanted.
    # tensor = F.gaussian_blur(tensor, kernel_size=[5, 5], sigma=[1.0])
    return tensor
