"""
Based on Binary Studies' article:
https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html

Method to extract mean and std color values for normalization,
works with any dataset, needs dataloader class from
torch.utils.data package.
"""

import os

from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main():
    data_dir = r"C:\Users\Noel\Documents\THESIS STUFF\Data\artbench-10-imagefolder-split"

    batch_size = 256
    input_size = 224
    data_transforms = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5162,
                                                           0.4644,
                                                           0.3975],
                                                          [0.2724,
                                                           0.2640,
                                                           0.2574])]),
        'test': transforms.Compose([transforms.Resize(input_size),
                                    transforms.CenterCrop(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5162,
                                                          0.4644,
                                                          0.3975],
                                                         [0.2724,
                                                          0.2640,
                                                          0.2574])]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                      data_transforms[x]) for x in ['train', 'test']}

    loader = DataLoader(image_datasets['train'],
                        batch_size=batch_size,
                        num_workers=2)

    def batch_mean_and_sd(loader):

        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for images, _ in tqdm(loader):
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                      dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (
                        cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (
                                cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
        return mean, std

    mean, std = batch_mean_and_sd(loader)
    print("mean and std: \n", mean, std)


if __name__ == '__main__':
    main()
