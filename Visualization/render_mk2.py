from __future__ import absolute_import, print_function
import os

import numpy as np
import gradio as gr
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models
# from tqdm import tqdm

import image_classes
import transformations

from objective_classes_mk2 import *


class Render_Class():
    def __init__(self) -> None:
        self.flag = False

    def set_flag(self):
        # print('Been through here.')
        self.flag = True

    def render(self,
               type,
               operator,
               layer,
               channel,
               param,
               threshold,
               image_shape,
               layer2=None,
               channel2=None,
               progress=gr.Progress()
               ):
        """Customizable method creating visualizations based on
            specified objectives and parameters.

            Utilizes case matching for each type of objective in
            'objective_classes' and performs gradient ascend based
            on input parameters.

            Is usually called through 'Interaction_Handling.py',
            implementing gradio interface on localhost.

            Args:
                type: str,
                operator: str,
                layer: str,
                channel: int,
                param: str (e.g. 'fft'),
                image_shape: int (batch dimension),
                layer2: str,
                channel2: int

            Return:
                PIL.Image
        """
        self.flag = False
        # Hyper Parameters
        threshold = threshold or 256
        parameterization = param or 'fft'
        # Initializing the shape.
        if type.strip('<p>\n/') == 'Diversity':
            shape = [image_shape, 3, 128, 128]
        else:
            shape = [image_shape, 3, 224, 224]
        multiple_objectives = False

        # Execute utilizing GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model with no weights.
        model = models.resnet18(weights=None)
        # Reshape last layer.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10)
        model.load_state_dict(torch.load("C://Users//Noel//Documents//THESIS"
                                         "//Feature Visualization//Weights"
                                         "//resnet18_torchvision"
                                         "//test65_epoch598.pth"))
        model.to(device).eval()

        # Conversion of ReLU activation function to LeakyReLU.
        module_convertor(model, nn.ReLU, nn.LeakyReLU(inplace=True))
        module_dict = module_fill(model)

        # Create image object ( image to parameterize, starting from noise)
        if parameterization == "pixel":
            image_object = image_classes.Pixel_Image(shape=shape)
            parameter = [image_object.parameter]
        elif parameterization == "fft":
            image_object = image_classes.FFT_Image(shape=shape)
            parameter = [image_object.spectrum_random_noise]
        else:
            exit("Unsupported initial image, please select parameterization \
                options: 'pixel' or 'fft'!")
        
        # Define optimizer and pass the parameters to optimize
        optimizer = torch.optim.Adam(parameter, lr=0.05)

        match type.strip('<p>\n/'):
            case 'DeepDream':
                objective = DeepDream_Obj(module_dict[layer])
            case 'Channel':
                objective = Channel_Obj(module_dict[layer], channel)
            case 'Neuron':
                objective = Neuron_Obj(module_dict[layer], channel)
            case 'Interpolation':
                objective = Channel_Interpolate(module_dict[layer],
                                                channel,
                                                module_dict[layer2],
                                                channel2)
            case 'Joint':
                objective = Channel_Obj(module_dict[layer], channel)
                secondary_obj = Channel_Obj(module_dict[layer2], channel2)
                multiple_objectives = True
            case 'Diversity':
                objective = 1e-2 * Diversity_Obj(module_dict[layer], channel)
            case 'Channel Weight':
                objective = Channel_Weight(module_dict[layer],
                                        weight=torch.rand(module_dict[layer].out_channels,
                                        device=device))
            case 'Direction':
                objective = Direction(module_dict[layer],
                                    direction=torch.rand(module_dict[layer].out_channels,
                                    device=device))
            case 'WRT Classes':
                objective = WRT_Classes(module_dict[layer],
                                        channel,
                                        model)
            case _:
                exit('No valid objective was selected from objective list.')

        for _ in progress.tqdm(range(0, threshold), total=threshold):

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                # Forward pass
                model(transformations.standard_transforms(image_object()))

                if multiple_objectives:
                    loss = operation(operator, objective(), secondary_obj())
                    # print(loss)
                else:
                    loss = operation(operator, objective())
                    # print(loss)

                loss.backward()
                return loss

            optimizer.step(closure)
            if self.flag:
                return None  # display_out(image_object())

        # Display final image after optimization
        return display_out(image_object())


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
    return Image.fromarray(image)


# Saving function
def save_image(tensor: torch.Tensor, path: str, name: str) -> None:
    name = name or "image_temp.jpg"
    image = tensor_to_array(tensor)
    # Change datatype to PIL supported.
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    if os.path.exists(path) is False:
        os.mkdir(path)
    Image.fromarray(image).save(path + name)


# This method converts modules into other specified types.
def module_convertor(model, module_type_pre, module_type_post):
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
    module_dict = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                underscored_name = name.replace('.', ' ')
                module_dict[underscored_name] = mod
    return module_dict
