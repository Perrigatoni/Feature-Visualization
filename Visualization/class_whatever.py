from __future__ import absolute_import, print_function
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image
from torchvision import models

import image_classes

from objective_classes_mk2 import *



def main():

    parameterization = 'fft'
    # Initializing the shape.
    shape = [1, 3, 224, 224]
    

    # Execute utilizing GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model with no weights.
    model = models.resnet18(weights=None)
    # Reshape last layer.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    model.load_state_dict(torch.load("C://Users//Noel//Documents//THESIS"\
                                     "//PYTHON-THINGIES//Saved Model Parameters"
                                     "//resnet18_torchvision//test40_epoch198.pth"))
    model.to(device).eval()

    # Conversion of ReLU activation function to LeakyReLU.
    module_convertor(model,
                     nn.ReLU,
                     nn.LeakyReLU(inplace=True))

    image = Image.open(r"C:\Users\Noel\Documents\THESIS\Outputs_Feature_Visualization\test40\fc\9_Positive.jpg")
    transforms = T.Compose([
                            T.ToTensor(),
                            T.Resize(224),
                            T.Normalize([0.5162, 0.4644, 0.3975],
                                        [0.2724, 0.2640, 0.2574]),
                           ])

    img = transforms(image).to(device)
    # Forward pass
    preds = model(img.unsqueeze(0))
    prediction = torch.nn.functional.softmax(preds, dim=1)
    print(prediction)



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
    module_dict = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                underscored_name = name.replace('.', ' ')
                module_dict[underscored_name] = mod
    return module_dict

if __name__ == "__main__":
    main()