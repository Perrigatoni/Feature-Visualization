from __future__ import absolute_import, print_function
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models
from tqdm import tqdm

import image_classes
import transformations

from objective_classes_mk2 import *
# from scratch_tiny_resnet import ResNetX, ResNet10


# Convert Tensor to numpy array.
def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """Conversion of 4d tensor to numpy array equivalent"""
    img_array = tensor.cpu().detach().numpy()
    img_array = np.transpose(img_array, [0, 2, 3, 1])
    return img_array


# Viewing function
def display_out(tensor: torch.Tensor):
    """Primary display function"""
    image = tensor_to_array(tensor)
    # Change datatype to PIL supported.
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    return Image.fromarray(image).show()


# Saving function
def save_image(tensor: torch.Tensor,
               path: str,
               name: str) -> None:
    """Primary saving function"""
    name = name or "image_temp.jpg"
    image = tensor_to_array(tensor)
    # Change datatype to PIL supported.
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    check_path(path)
    # if os.path.exists(path) is False:
    #     os.mkdir(path)
    Image.fromarray(image).save(path + name)


# Path checking and creation
def check_path(path: str):
    """Checks if path exists.
        Will create the specified directories if
        they don't already exist."""
    if os.path.exists(path) is False:
        os.makedirs(path)


# This method converts modules into other specified types.
def module_convertor(model,
                     module_type_pre,
                     module_type_post):
    conversions_made = 0
    for name, module in model._modules.items():
        if len(list(model.children())) > 0:
            model._modules[name] = module_convertor(module,
                                                    module_type_pre,
                                                    module_type_post)

        if type(module) == module_type_pre:
            conversions_made += 1
            # module_pre = module
            module_post = module_type_post
            model._modules[name] = module_post
    # print(conversions_made)
    return model


def module_fill(model):
    """Returns a dictionary with entries of all conv
        layers present, as well as any linear ones
        (to accommodate for the fc layers.)"""
    module_dict = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                underscored_name = name.replace('.', ' ')
                module_dict[underscored_name] = mod
    return module_dict


def main():
    """
    Standalone render function able to reproduce results
    achieved with the Interaction handling,
    without the need for gradio library,
    allowing more flexibility to the end user.

    Adapt parameters to save images in custom directories,
    mix different objectives, view customized results,
    import own weights and visualize personalized images,
    create elaborate save-files for all layers/channels
    in given network.

    """
    local_machine_results = True
    verbose_logs = False
    # Hyper Parameters
    threshold = 1024
    parameterization = 'fft'
    # Initializing the shape.
    shape = [1, 3, 224, 224]
    multiple_objectives = False  # in case of Mixing objs
    operator = 'Positive'

    # Execute utilizing GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model with no weights.
    model = models.resnet18(weights=None)
    # model = ResNet10()
    # model = models.resnet18(weights=models.ResNet18_Weights)
    # Reshape last layer.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    if local_machine_results:
        model.load_state_dict(torch.load("C://Users//Noel//Documents//THESIS"
                                         "//Feature Visualization//Weights"
                                         "//resnet18_torchvision"
                                         "//test65_epoch598.pth"))
    else:
        model.load_state_dict(torch.load("/home/perryman1997/"
                                         "saved_model_parameters/test71"
                                         "/test71_epoch98.pth"))

    model.to(device).eval()

    # Conversion of ReLU activation function to LeakyReLU.
    module_convertor(model,
                     nn.ReLU,
                     nn.LeakyReLU(inplace=True))
    module_dict = module_fill(model)

    since = time.time()

    # for layer_name, layer in module_dict.items():
    layer = model.fc
    layer_name = "fc"
    # Remove loop if you are not interested in creating directories.
    for channel_n in range(0, layer.out_features):

        # Create image object (image to parameterize, starting from noise)
        if parameterization == "pixel":
            image_object = image_classes.Pixel_Image(shape=shape)
            parameter = [image_object.parameter]
        elif parameterization == "fft":
            image_object = image_classes.FFT_Image(shape=shape)
            parameter = [image_object.spectrum_random_noise]
        else:
            sys.exit("Unsupported initial image, please select \
                        parameterization options: 'pixel' or 'fft'!")

        # Define optimizer and pass the parameters to optimize
        optimizer = torch.optim.Adam(parameter, lr=0.05)

        objective = Channel_Obj(layer=layer, channel=channel_n)
        # secondary_obj = Channel_Obj(layer=module_dict[sec_layer_name],
        #                             channel=9)
        for step in tqdm(range(0, threshold), total=threshold):

            def closure() -> float:
                optimizer.zero_grad()
                # Forward pass
                model(transformations.standard_transforms(image_object()))
                if multiple_objectives:
                    loss = operation(operator,
                                        objective(),
                                        secondary_obj())
                    # print(loss)
                else:
                    loss = operation(operator,
                                        objective())
                    # print(loss)
                if verbose_logs and step == threshold - 1:
                    print(f"Loss at step {step}:{loss}")
                loss.backward()
                return loss.item()

            optimizer.step(closure)

        # Display final image after optimization
        # display_out(image_object())
        if local_machine_results:
            save_path = r"C:\Users\Noel\Documents\THESIS"\
                r"\Outputs_Feature_Visualization"\
                rf"\test65outputs\{layer_name.replace(' ', '_')}"
        else:
            save_path = r"/home/perryman1997"\
                rf"/outputs/test71/{layer_name.replace(' ', '_')}"

        check_path(save_path)

        # Save each image
        save_image(image_object(),
                    path=save_path,
                    name=f"/{str(channel_n)}_{operator}_epoch598.jpg")

    elapsed_time = time.time() - since
    if verbose_logs:
        print(f'Runtime: {elapsed_time}')


if __name__ == "__main__":
    main()
