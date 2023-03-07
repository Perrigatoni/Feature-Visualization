
import os
import numpy as np
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imgaug
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

data_dir = r"C:\Users\Noel\Documents\THESIS\Data\artbench-10-imagefolder-split"
batch_size = 384

input_size=224
def main():
    transformations = T.Compose([
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.RandomAffine(degrees=0,
                                           translate=(0.05, 0.05),
                                           scale=(0.8, 1.2),
                                           interpolation=InterpolationMode.BILINEAR),
                            T.RandomResizedCrop(input_size, scale=(0.5, 1)),
                            T.ToTensor(),
                            # T.Normalize([0.5162, 0.4644, 0.3975],
                            #             [0.2724, 0.2640, 0.2574]),
                            T.RandomErasing(),
                                          ])
    # transformations = A.Compose([A.SmallestMaxSize(max_size=160),
    #                              A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    #                              A.RandomCrop(height=128, width=128),
    #                              A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    #                              A.RandomBrightnessContrast(p=0.5),
    #                              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #                              ToTensorV2(),
    #                             ])

    # class ImageFolderAlbumentations(ImageFolder):
    #     """Implements new __getitem__ for use with
    #         albumentations augmentations.
    #         Albumentations library requires the image
    #         be returned from a dictionary entry named
    #         'image' in order to apply transforms.
    #         The image must also be converted from PIL
    #         to ndarray and then to tensor through 
    #         ToTensor operation.
            
    #         Returns:
    #             Tuple[image: ndarray,
    #                   target: int
    #                   ]
    #         """
    #     def __getitem__(self, index: int):
    #         path, target = self.samples[index]
    #         img = np.asarray(self.loader(path))
    #         if self.transform is not None:
    #             image = self.transform(image=img)["image"]
    #         else:
    #             raise NotImplementedError

    #         return image, target

    dataset = ImageFolder(os.path.join(data_dir, "train"),
                                        transformations)

    inputs, _ = dataset.__getitem__(163)
    display_out(inputs.unsqueeze(0))
    




# Convert Tensor to numpy array.
def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:  # add more
    # things like these, they make it easier to debug
    img_array = tensor.cpu().detach().numpy()
    img_array = np.transpose(img_array, [0, 2, 3, 1])
    return img_array

# Viewing function
def display_out(tensor: torch.Tensor):
    image = tensor_to_array(tensor)
    # Change datatype to PIL supported.
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    return Image.fromarray(image).show()


if __name__ == '__main__':
    main()
