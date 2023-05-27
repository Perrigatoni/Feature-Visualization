""" Resnet architecture based on the original published paper.
    
    This is a naive implementation on which we are to perform
    customizations based on our need for a relatively shallower
    residual network to train with smaller/easier datasets,
    like CelebA and Food101.
    """

import torch
import torch.nn as nn
# from torchvision.models import resnet18
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class block_tiny(nn.Module):
    """This is the basic building block utilized by all
        ResNet variants as a creation kit."""
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1) -> None:
        # The identity downsample is a conv layer, which we might not need
        # depending on whether we are changing the input size OR
        # if the number of channels is modified.
        super().__init__()
        self.expansion = 1  # num of channels post block is 1x the input channels

        # Define the first conv layer of the block, like a reduction, it is
        # a 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # Define the first batch normalization of the block
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Define the second conv layer of the block
        # (stride here is what we input to the block object)
        # --- Take also note that input and output channels
        # in this layer remain constant.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Define the second batch normalization of the block
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Identity mapping with this attribute. This will be a convolution
        # aiming to retain the same shape later on when doing layer
        # operations. I would describe it as size-matching.
        self.identity_downsample = identity_downsample

    """Definitition of the actual forward method followed in this
        basic block we just designed."""
    def forward(self, x):
        # retain the input as identity (ResNet spec)
        identity = x
        # Actual forward pass.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # Check for identity downsampling
        # If present, this will be a Sequential container
        # that handles identity re-mapping so it can match
        # in dimensions the output of the block.
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        # Continuing the forward pass.
        x += identity  # this it the residuality of ResNet.
        x = self.relu(x)
        return x


# ResNet class, inheriting from nn.Module itself.
class ResNet(nn.Module):
    """This class creates the actual ResNet object
        later to be used as the definition of the
        model.

        args:
            block(opt, nn.Module): the basic block creator class

            layers(list[int]): the amount of times we
                                will utilize the block
            image_channels(int): the input channels,
                                commonly 3.
            num_classes(int): the number of classes
                                in our application,
                                1000 on imagenet's task."""
    def __init__(self, block_tiny, layers, image_channels, num_classes) ->None:
        """Initialize a couple of attributes for our module.
            In ResNets, the first convolution, batch norm,
            relu and maxpool do remain the same for all variants.
            """
        super().__init__()
        self.layers = layers
        self.in_channels = 64  # initial channels after the first conv layer
        # This is the initial convolution, made to serve as a precursor
        # to any residual blocks that follow. Does NOT change, do not modify.
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # maxpool does not do anything for the channels.

        """ Here we define the 4 Residual Layers, each
            containing a variable number of blocks, as per spec."""
        # ResNet Layers (these are the definitiions of the actual residual layers used.)
        self.layer1 = self._make_layer(block_tiny, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block_tiny, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block_tiny, layers[2], out_channels=256, stride=2)
        if len(self.layers) == 4: self.layer4 = self._make_layer(block_tiny, layers[3], out_channels=512, stride=2)
        # The above structure is followed by an average pooling layer, an adaptive one at that
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if len(self.layers) == 4:
            self.fc0 = nn.Linear(512, 1024)
            self.fc1 = nn.Linear(1024, 512)
            self.fc = nn.Linear(512, num_classes) # ! CHANGE THAT IN CASE OF RESNETX TO MATCH LAYER 3 OUT
        else:
            self.fc0 = nn.Linear(256, 1024)
            self.fc1 = nn.Linear(1024, 256)
            self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # followed by the resnet layers... finally, what a complex structuring that was... </3
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if len(self.layers) == 4 : x = self.layer4(x)
        # finalize with the last two, common for all variants, layers
        x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)  # do not forget to reshape the tensor so it can pass throught the linear layer
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc(x)
        return x


    # block is the class we first defined, num_residual_blocks is the number of times
    # those blocks are going to be used.
    def _make_layer(self, block_tiny: nn.Module, num_residual_blocks: int, out_channels: int, stride: int) -> nn.Sequential:
        """ Creates blocks of layers (as objects) by calling the block class.
            Also handles downsampling in case of different identity size.
            I would describe this as a helper function, actually calling
            the block class and passing necessary parameters and attributes
            for the creation of the blocks.
            """
        identity_downsample = None
        layers = []  # empty list on which layers are to be appended next.

        # We need to know when are we going to do an identity downsample.
        # Identity downsample -> conv layer modifier of identity size.

        # There are 2 main cases when identity downsampling should be done.
        # Either the input_size changes (like when jumping from residual
        # blocks of one type to another, see input_size going from 28x28
        # to 14x14 after layer3_x on resnet18 (arxiv pub.)) i.e. the
        # stride isn't 1, OR, the input_channels are not out_channels.

        if stride != 1 or self.in_channels != out_channels * 1:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*1, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*1))
        # Append the newly created layer to layers list.
        # This changes the number of channels, and is the final layer of the 
        layers.append(block_tiny(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 1  # e.g. 64 * 1 = 64 in the first res layer of resnet18

        for _ in range(num_residual_blocks - 1):
            layers.append(block_tiny(self.in_channels, out_channels))
        # Return the list contents( i.e. the objects of class block)
        # in a Sequential container.
        return nn.Sequential(*layers)  # the * modifier unpacks the list
        # Return to the ResNet layers, and look at the argument parameters.


"""
We can now define the ResNet variants as follows.
"""

def ResNet34(img_channels=3, num_classes=1000):
    return ResNet(block_tiny, [3, 4, 6, 3], img_channels, num_classes)


def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(block_tiny, [2, 2, 2, 2], img_channels, num_classes)

def ResNetX(img_channels=3, num_classes=1000):
    return ResNet(block_tiny, [2, 2, 2], img_channels, num_classes)

def ResNet10(img_channels=3, num_classes=1000):
    return ResNet(block_tiny, [1, 1, 1, 1], img_channels, num_classes)


# def test():
#     net = ResNet18()
#     x = torch.randn(32, 3, 224, 224)
#     y = net(x).to(device)
#     print(y.shape)

# test()
# ================================================================
# model = ResNet10()
# module_dict = {}

# def module_fill(model):
#         for name, mod in model.named_modules():
#             if len(list(mod.children())) == 0:
#                 if isinstance(mod, nn.Conv2d):
#                     module_dict[name] = mod

# module_fill(model=model)
# for name in list(module_dict.keys()):
#     print(name)