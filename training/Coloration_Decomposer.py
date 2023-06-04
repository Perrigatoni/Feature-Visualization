"""
    Method for color decorellation based on ProGamerGov's original
    implementation found here:
    https://github.com/ProGamerGov/dream-creator/blob/master/data_tools/calc_cm.py

    - This method calculates the color correlation svd matrix needed to perform
    color decorellation in the image_classes.py, so as to achieve results based
    in a decorellated space both spatially but also chromatically.

    Does not exactly implement Cholesky Decomposition as proposed by the Lucid
    authors but instead achieves similar results.

    For more information, check this blogpost:
    https://stackoverflow.com/questions/64015444/how-to-calculate-the-3x3-covariance-matrix-for-rgb-values-across-an-image-datase
    ,and the resulting matrix on Lucid's code:
    https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/color.py#L24

"""

import os

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from torchvision import datasets
from torch.utils.data import DataLoader


data_dir = r"C:\Users\Noel\Documents\THESIS\Data\artbench-10-imagefolder-split"
batch_size = 384


# Method that handles the rgb covariance calculation.
# Returns a 3x3 torch.Tensor representing the covariance
# matrix for each image.
def rgb_cov(im) -> torch.Tensor:
    """
    Assuming im is a torch.Tensor of shape (H,W,3)
    """
    im_re = im.reshape(-1, 3)
    # mean_extr = im_re.mean(0, keepdim=True)
    # im_re = im_re - mean_extr
    im_re -= im_re.mean(0, keepdim=True)
    # return 1/(im_re.shape[0]-1) * im_re.T.__matmul__(im_re)
    return 1/(im_re.shape[0]-1) * torch.matmul(im_re.T, im_re)


def main():
    # Bring standard transorms for the dataset. We only
    # require a few transforms to standardize the data,
    # not including training transforms or augmentation.
    transformations = transforms.Compose([transforms.Resize(256),  # Artbench alr. 256.
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    # Next we require a Dataloader implementation to load
    # the data in batched form.
    dataset = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                   transformations)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,  # Shuffling makes no difference here.
                            num_workers=4)

    print('Computation of covariance is heavily dependent on\n' +
        'the dataset used. This could take a while.')

    covariance_matrix = 0
    for images, _ in tqdm(dataloader):
        for index in range(images.size(0)):
            # Permute each image of the batch to spec, passing it
            # to the rgb_cov function.
            covariance_matrix += rgb_cov(images[index].permute(1, 2, 0))

    covariance_matrix = covariance_matrix / len(dataset)
    # Check online the theory about
    # covariance calculation and svd.
    U, S, V = torch.linalg.svd(covariance_matrix)  # V is not used
    # Define a small value to avoid calculations with random zeros.
    eps = 1e-10
    # S is a vector, has to become a matrix with the values places
    # diagonally.
    svd_sqrt = torch.matmul(U, torch.diag(torch.sqrt(S + eps)))

    print('The Color Correlation matrix is:\n')
    print(svd_sqrt)


if __name__ == '__main__':
    main()
