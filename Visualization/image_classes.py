from __future__ import absolute_import, print_function

import torch
import numpy as np
import torch.fft

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RGB_decorrelation():
    def __init__(self):
        self.color_correlation_svd_sqrt = torch.tensor([[-0.2051,  0.0626,  0.0137],
                                                        [-0.2001, -0.0070, -0.0286],
                                                        [-0.1776, -0.0644,  0.0164]])
        # IMAGENET VALUES
        # self.color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02],
        #                                                [0.27, 0.00, -0.05],
        #                                                [0.27, -0.09, 0.03]])
        self.max_norm_svd_sqrt = torch.max(torch.linalg.norm(self.color_correlation_svd_sqrt, axis=0))
        self.color_correlation_normalized = (self.color_correlation_svd_sqrt / self.max_norm_svd_sqrt).to(device)

    def linearly_decorrelated_color_space(self, tensor_img):
        # permute tensor to appropriate shape for matrix multiplication
        tensor_permuted = tensor_img.permute(0, 2, 3, 1)
        #print(f"Permuted tensor image is of shape: {tensor_permuted.shape}")
        # Multiply with the correlation matrix (product of Cholesky decomp.)
        tensor_permuted = torch.matmul(tensor_permuted, self.color_correlation_normalized.T)
        #print(tensor_permuted.shape)
        # Permute image tensor back to more common/usable shape
        tensor_img = tensor_permuted.permute(0, 3, 1, 2)
        return tensor_img


class Pixel_Image(RGB_decorrelation):
    def __init__(self, shape, sd=None) -> None:
        super().__init__()
        self.shape = shape
        self.sd = sd or 0.01
        self.parameter = (torch.randn(*self.shape) * self.sd).to(device).requires_grad_(True)

    def __call__(self) -> torch.Tensor:
        output = self.linearly_decorrelated_color_space(self.parameter)
        return torch.sigmoid(output)  # get valid RGB with sigmoid


class FFT_Image(RGB_decorrelation):
    
    def __init__(self, shape, sd=None, decay_power=None) -> None:
        super().__init__()
        self.shape = shape
        self.batch, self.channels, self.h, self.w = shape
        self.freq2d = self.sample_frequencies()
        self.real_size = (self.batch, self.channels) + self.freq2d.shape + (2,)
        self.sd = sd or 0.01
        self.decay_power = decay_power or 1

        self.spectrum_random_noise = (torch.randn(*self.real_size) * self.sd).to(device).requires_grad_(True)
        # So this operation just replaces the values in the
        # array of sample frequencies with a predetermined value 
        # to replace those that are too low... e.g. 0.0 is replaced
        # with 0.0045.
        scale = 1.0 / np.maximum(self.freq2d, 1.0/max(self.w, self.h)) ** self.decay_power
        #scale = 1.0 / np.maximum(self.freq2d, 0.001)
        # scale = np.maximum(self.freq2d, 1.0/max(self.w, self.h))
        # Clone the tensor to suppress warning.
        # The three dots, called ellipsis, denotes "as many/as needed".
        # Labeling the other dimensions with None creates dummy ones.
        self.scale = scale.clone().float()[None, None, Ellipsis, None].to(device)

    def sample_frequencies(self):
        """Compute 2D spectrum frequencies."""
        height = self.h
        width = self.w
        vertical_freq = torch.fft.fftfreq(height)[:, None]
        if width % 2 == 1:
            horizontal_freq = torch.fft.fftfreq(width)[: width // 2 + 2]
        else:
            horizontal_freq = torch.fft.fftfreq(width)[: width // 2 + 1]
        # Calculate the fft frequencies
        # This will be a 2D array with "distances" from the origin (0,0).
        # We will need these to compute the necessary "punishing" scale,
        # in essence directly minimizing the coefficients related with
        # high frequencies, those away from the origin.
        frequencies = np.sqrt(horizontal_freq**2 + vertical_freq**2)
        return frequencies

    def __call__(self) -> torch.Tensor:
        """An image paramaterization using 2D Fourier coefficients.

            The coefficients further from the origin (frequency origin (0,0),
            referring to the usual u and v spectrum frequencies when
            computing DFT) are multiplied with the inverse of their frequency
            as a means to scale their energy. It seems to be a way of directly
            'impacting' high frequency noisy patterns in the image, promoting
            the values of low frequency coefficients."""
        # Convolution of Fourier coefficients with scale array.
        regularized_coefficient_array = self.scale * self.spectrum_random_noise
        # Requires torch version greater than 1.7.0
        if type(regularized_coefficient_array) is not torch.complex64:
            regularized_coefficient_array = torch.view_as_complex(regularized_coefficient_array)
        # Computes the inverse FFT with respect to Hermitian property
        # of the input tensor.
        # Hermitian property -- "The input has complex conjugate
        # with same value but different sign in order to cancel out,
        # returning a scalar value."
        tensor_image = torch.fft.irfftn(regularized_coefficient_array,
                                        s=(self.h, self.w),
                                        norm='ortho') # Need to define orthonormal basis!

        tensor_image = tensor_image[:self.batch, :self.channels, :self.h, :self.w]
        # Indeed this seems to saturate colors... don't know why, inadequate 
        # documentation on Lucid's implementation
        magic_const = 4.0
        tensor_image = tensor_image / magic_const
        tensor_image = self.linearly_decorrelated_color_space(tensor_image)
        return torch.sigmoid(tensor_image)


""" Does not work, obviously since you just convert
    to frequency and do no other operation before
    inverting the conversion.
"""


# class FFT_Image2():
#     def __init__(self, shape, sd=None, decay_power=None) -> None:
#         self.real_size = shape
#         self.sd = 0.01
#         self.real_image_tensor = (torch.randn(*self.real_size) * self.sd).to(device).requires_grad_(True)

#     def __call__(self) -> torch.Tensor:
#         fft_image_tensor = torch.fft.rfftn(self.real_image_tensor).to(device)
#         #fft_image_tensor = fft_image_tensor * 2
#         tensor_image = torch.fft.irfftn(fft_image_tensor).to(device)
#         # magic_const = 0.10
#         # tensor_image = tensor_image / magic_const
#         return torch.sigmoid(tensor_image)
