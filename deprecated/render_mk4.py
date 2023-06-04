from __future__ import absolute_import, print_function
import os

import numpy as np
import gradio as gr
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import list_models, get_model

from Visualization_with_Gradio import image_classes, transformations
from Visualization_with_Gradio.objective_classes import *

# Execute utilizing GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            model._modules[name] = module_convertor(
                module, module_type_pre, module_type_post
            )

        if type(module) == module_type_pre:
            conversions_made += 1
            # module_pre = module
            module_post = module_type_post
            model._modules[name] = module_post
    return model


def module_fill(model):
    """Use this method to create dictionary of all available
    CNN Conv2d and Linear layers. Commonly used to gain
    access to naming schemes of different layers in or out of
    Sequential containers. Would advise to always let this
    method fill the respective dictionary so as to reference
    model layers later on when visualizing features.

    Args:
        model: nn.Module (the model class to use)
    Returns:
        module_dict: dict (dictionary with named conv
                            and linear modules)
    """
    module_dict = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                if "downsample" in name:
                    pass
                else:
                    underscored_name = name.replace(".", " ")
                    module_dict[underscored_name] = mod
    return module_dict


def get_out_classes(dict1, dict2):
    """Compare two state dicts to find if parameters keys
    match between the two.

    Meant to be used when trying to load a new state
    dict to a torchvision model.
    Cannot ensure compatibility with scratch made
    models yet.

    Args:
        dict1, dict2: OrderedDict (The state dicts to compare)
    Returns:
        Number of Output classes. (Used to reshape the FC
                                   classifier in the end of
                                   a CNN.)
    """
    if dict1.keys() != dict2.keys():
        raise ValueError("Incompatible state dict and model!\
        Unable to load weights.")
    else:
        print("Compatible State Dicts! Checking Classifier shape...")
        classifier_t = list(dict1.values())[-1]
        print(classifier_t.shape)
        return classifier_t.shape[0]


class Render_Class:
    def __init__(self) -> None:
        self.abort_flag = False
        self.model = None
        self.change_act_func = False
        self.module_dict = {}

    def available_layers(self, model_name):
        """Set model and fill dict with all available conv and linear
        layers.
        """
        self.model_name = model_name
        if self.model_name in list_models():
            self.model = get_model(
                model_name.strip("<p>\n/").lower(), weights="DEFAULT"
            )
            self.module_dict = module_fill(self.model)
            return gr.Radio.update(
                choices=list(self.module_dict.keys()),
                value=list(self.module_dict.keys())[0],
            )

    def state_dict_upload(self, pth_file):
        if os.path.exists(pth_file.strip("\"\"")):
            self.file_path = pth_file.strip("\"\"")  # .name
            state_dict = torch.load(self.file_path,
                                    map_location=torch.device("cpu"))
            dataset_out_classes = get_out_classes(state_dict,
                                                  self.model.state_dict())
            classifier_name = list(self.module_dict)[-1]
            classifier = getattr(self.model, classifier_name)
            classifier.out_features = dataset_out_classes
            self.module_dict = module_fill(self.model)
            print(self.model)
            self.model.load_state_dict(torch.load(self.file_path))
        else:
            self.model = get_model(
                self.model_name.strip("<p>\n/").lower(), weights="DEFAULT"
            )
            self.module_dict = module_fill(self.model)

    def update_sliders(self, layer_name):
        """Update channel sliders' maximum cap."""
        if isinstance(self.module_dict[layer_name], nn.Linear):
            max_channel_num = self.module_dict[layer_name].out_features - 1
        else:
            max_channel_num = self.module_dict[layer_name].out_channels - 1
        return gr.update(maximum=max_channel_num)

    def abort_operation(self):
        """Aborts render operation if needed."""
        self.abort_flag = True

    def handle_act_func(self, act_func):
        """Handles the change of activation function to Leaky ReLU.
        Leaky ReLU behaves a bit better when trying to
        visualize features.
        """
        act_func = act_func.strip("<p>\n/")
        self.change_act_func = True if "Leaky" in act_func else False
        # print(self.change_act_func)

    def render(
        self,
        type,
        operator,
        layer,
        channel,
        param,
        threshold,
        image_shape,
        layer2=None,
        channel2=None,
        progress=gr.Progress(),
    ):
        """Customizable method creating visualizations based on
        specified objectives and parameters.

        Utilizes case matching for types of objective in
        'objective_classes' and performs gradient ascend based
        on input parameters.

        This function works with all torchvision models with
        pretrained weights and is designed to be called by a
        gradio interface, hence all arguments are strings and integers.

        See the standalone render script if you want to experiment
        with backend operations and precise mixing or directory
        creation.

        Args:
            type: str,
            operator: str,
            layer: str,
            channel: int,
            param: str (e.g. 'fft'),
            threshold: int (evaluation steps)
            image_shape: int (batch dimension),
            layer2: str,
            channel2: int

        Return:
            PIL.Image
        """
        self.abort_flag = False
        self.model.to(device).eval()
        # Hyper Parameters
        threshold = threshold or 256
        parameterization = param or "fft"
        # Initializing the shape.
        if type.strip("<p>\n/") == "Diversity":
            shape = [image_shape, 3, 128, 128]
        else:
            shape = [image_shape, 3, 224, 224]
        multiple_objectives = False

        # Conversion of ReLU activation function to LeakyReLU.
        if self.change_act_func:
            module_convertor(self.model, nn.ReLU, nn.LeakyReLU(inplace=True))
        # module_dict = module_fill(self.model)

        # Create image object ( image to parameterize, starting from noise)
        if parameterization == "pixel":
            image_object = image_classes.Pixel_Image(shape=shape)
            parameter = [image_object.parameter]
        elif parameterization == "fft":
            image_object = image_classes.FFT_Image(shape=shape)
            parameter = [image_object.spectrum_random_noise]
        else:
            exit(
                "Unsupported initial image, please select parameterization \
                options: 'pixel' or 'fft'!"
            )

        # Define optimizer and pass the parameters to optimize
        optimizer = torch.optim.Adam(parameter, lr=0.05)

        match type.strip("<p>\n/"):
            case "DeepDream":
                objective = DeepDream_Obj(self.module_dict[layer])
            case "Channel":
                objective = Channel_Obj(self.module_dict[layer], channel)
            case "Neuron":
                objective = Neuron_Obj(self.module_dict[layer], channel)
            case "Interpolation":
                objective = Channel_Interpolate(
                    self.module_dict[layer], channel, self.module_dict[layer2], channel2
                )
            case "Joint":
                objective = Channel_Obj(self.module_dict[layer], channel)
                secondary_obj = Channel_Obj(self.module_dict[layer2], channel2)
                multiple_objectives = True
            case "Diversity":
                objective = Diversity_Obj(self.module_dict[layer], channel)
            case _:
                exit("No valid objective was selected from objective list.")

        for _ in progress.tqdm(range(0, threshold), total=threshold):

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                # Forward pass
                self.model(transformations.standard_transforms(image_object()))
                # self.model(image_object())
                if multiple_objectives:
                    loss = operation(operator, objective(), secondary_obj())
                    # print(loss)
                else:
                    loss = operation(operator, objective())
                    # print(loss)

                loss.backward()
                return loss

            optimizer.step(closure)
            if self.abort_flag:
                return None  # display_out(image_object())
        # Display final image after optimization
        return display_out(image_object())
