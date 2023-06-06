from __future__ import division, print_function

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from scratch_tiny_resnet import ResNetX, ResNet10
from scratch_tiny_plain import Plain18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


""" MAKE THIS INTO A CLASS MAYBE? """


# I need it to have a conversion made attribute
# so the recurrence does not actually reset the counter.
# The fact that I use no main function means I cannot just
# drop a global variable and call it a day.
class Converter():
    def __init__(self):
        self.conversions_made = 0
        self.model = None

    # This method converts modules into other specified types.
    def converter(self, model, module_type_pre, module_type_post):
        for name, module in model._modules.items():
            if len(list(module.children())) > 0:
                model._modules[name] = self.converter(module,
                                                      module_type_pre,
                                                      module_type_post)
            if type(module) == module_type_pre:
                self.conversions_made += 1
                # module_pre = module
                module_post = module_type_post
                model._modules[name] = module_post
        self.model = model
        return model


def initialize_model(model_name,
                     num_classes,
                     feature_extract,
                     use_pretrained=True,
                     change_AF=False):
    # We need to initialize a few values first
    model = None
    input_size = 0

    if model_name == "resnet18":
        """ Use torchvision's resnet18,34,50,152"""
        if use_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        # Pass the actual model from torchvision models to this variable.
        print(f"Selected Weights are: {weights}")
        model = models.resnet18(weights=weights)
        # Check for Activation Function conversion
        if change_AF:
            convert_modules = Converter()
            convert_modules.converter(model,
                                      nn.ReLU,
                                      nn.LeakyReLU(inplace=True))
            print(f'Conversions made: {convert_modules.conversions_made}')

        set_parameter_requires_grad(model, feature_extract)
        # Get the number of input features in
        # the Fully Connected layer of our model.
        num_ftrs = model.fc.in_features
        # Define and Initialize a new FC layer with the necessary output size.
        model.fc = nn.Linear(num_ftrs, num_classes)
        # Image input size for the model.
        input_size = 224
    # Add more conditions in the future...
    elif model_name == "resnet34":
        """ Use torchvision's resnet18,34,50,152"""
        if use_pretrained:
            weights = torchvision.models.ResNet34_Weights.DEFAULT
        else:
            weights = None
        # Pass the actual model from torchvision models to this variable.
        print(f"Selected Weights are: {weights}")
        model = models.resnet34(weights=weights)
        # Check for Activation Function conversion
        if change_AF:
            convert_modules = Converter()
            convert_modules.converter(model,
                                      nn.ReLU,
                                      nn.LeakyReLU(inplace=True))
            print(f'Conversions made: {convert_modules.conversions_made}')

        set_parameter_requires_grad(model, feature_extract)
        # Get the number of input features in
        # the Fully Connected layer of our model.
        num_ftrs = model.fc.in_features
        # Define and Initialize a new FC layer with the necessary output size.
        model.fc = nn.Linear(num_ftrs, num_classes)
        # Image input size for the model.
        input_size = 224
    elif model_name == "resnet_scratch":
        """ Use scratch made resnets."""
        if use_pretrained:
            weights = None
            print("No available pretrained weights for finetuning with scratch made resnets!")
        else:
            weights = None
        # Pass the actual model from torchvision models to this variable.
        print(f"Selected Weights are: {weights}")
        model = ResNet10()
        # Check for Activation Function conversion
        if change_AF:
            convert_modules = Converter()
            convert_modules.converter(model,
                                      nn.ReLU,
                                      nn.LeakyReLU(inplace=True))
            print(f'Conversions made: {convert_modules.conversions_made}')
            # model_ft = convert_modules.model

        set_parameter_requires_grad(model, feature_extract)
        # Get the number of input features in
        # the Fully Connected layer of our model.
        num_ftrs = model.fc.in_features
        # Define and Initialize a new FC layer with the necessary output size.
        model.fc = nn.Linear(num_ftrs, num_classes)
        # Image input size for the model.
        input_size = 224
    elif model_name == "plain18_scratch":
        """ Use scratch made plain net equivalent."""
        if use_pretrained:
            weights = None
            print("No available pretrained weights for finetuning with scratch-made plain nets!")
        else:
            weights = None
        # Pass the actual model from torchvision models to this variable.
        print(f"Selected Weights are: {weights}")
        model = Plain18()
        # Check for Activation Function conversion
        if change_AF:
            convert_modules = Converter()
            convert_modules.converter(model,
                                      nn.ReLU,
                                      nn.LeakyReLU(inplace=True))
            print(f'Conversions made: {convert_modules.conversions_made}')
            # model_ft = convert_modules.model

        set_parameter_requires_grad(model, feature_extract)
        # Get the number of input features in
        # the Fully Connected layer of our model.
        num_ftrs = model.fc.in_features
        # Define and Initialize a new FC layer with the necessary output size.
        model.fc = nn.Linear(num_ftrs, num_classes)
        # Image input size for the model.
        input_size = 224
    # Add more conditions in the future...
    else:
        print('Invalid Model name, exiting...')
        exit()

    return model, input_size