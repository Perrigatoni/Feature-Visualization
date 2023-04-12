from __future__ import absolute_import, print_function

import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


np.random.seed(0)
torch.manual_seed(0)


def standard_transforms(tensor) -> torch.Tensor:
    """ Input transform implementation to tackle transformation robustness."""
    jitter1 = 4
    # jitter2 = 8
    # normalization = transforms.Normalize([0.485, 0.456, 0.406],
    #                                      [0.229, 0.224, 0.225])
    normalization = transforms.Normalize([0.5162, 0.4644, 0.3975],
                                         [0.2724, 0.2640, 0.2574])
    tensor = normalization(tensor)
    tensor = F.pad(tensor, padding=[8], fill=0, padding_mode="constant")
    # Affine using a random choice of number in range
    dx = np.random.choice(jitter1)
    dy = np.random.choice(jitter1)
    scales = np.arange(0.9, 1.1, 0.01)
    scale_selected = np.random.choice(scales)
    # Reminder!!! Angles need to be int, so get the item from numpy.
    angles = np.arange(0, 90, 5)
    probability_array = np.linspace(1, 0, 18)
    probability_array[0] += 10  # modify 0 angle to have higher probability
    probabilities = probability_array / sum(probability_array)
    selected_angle = np.random.choice(angles, p=probabilities)
    tensor = F.affine(tensor,
                      angle=selected_angle.item(),
                      translate=[dx, dy],
                      scale=scale_selected,
                      shear=[0])
    # Apply blurring on each image. Seems to discourage noise further,
    # but the result is too subtle, or unwanted.
    # tensor = F.gaussian_blur(tensor, kernel_size=[5, 5], sigma=[1.0])
    return tensor
