from __future__ import absolute_import, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def operation(operator, loss1, loss2=torch.zeros(1,).to(device)):
    match operator:
        case '+':
            loss = loss1 + loss2
        case '-':
            loss = loss1 - loss2
        case 'Negative':
            loss =  - loss1
        case 'Positive':
            loss = loss1
        case _:
            loss = loss1
    return loss



# =================================================================
# Parent class used for hooking functionality on all objectives
class Common:
    def __init__(self, layer) -> None:
        self.hook = layer.register_forward_hook(self.hook_function)
        self.layer = layer
        self.scaler = 1.0
        self.output = {}

    def hook_function(self, layer, input, output) -> None:
        self.output[layer] = output

    def close(self):
        self.hook.remove()

    def __rmul__(self, other):
        self.scaler = other
        return self

    def __neg__(self):
        self.scaler = -1
        return self


class DeepDream_Obj(Common):
    def __init__(self, layer) -> None:
        super().__init__(layer)

    def __call__(self):
        if self.output[self.layer] is None:
            exit("Object callable only after forward pass!")
        dream = self.output[self.layer]**2  # like mse loss
        loss_local = -dream.mean()
        return self.scaler * loss_local

class Channel_Obj(Common):
    def __init__(self, layer, channel) -> None:
        super().__init__(layer)
        self.channel = channel

    def __call__(self) -> torch.Tensor:
        if self.output[self.layer] is None:
            exit("Object callable only after forward pass!")
        channel_tensor = self.output[self.layer][:, self.channel]
        loss_local = -channel_tensor.mean()
        return self.scaler * loss_local

class Neuron_Obj(Common):
    def __init__(self, layer, channel,
                 spatial_x=None, spatial_y=None):
        super().__init__(layer)
        self.channel = channel
        self.x = spatial_x
        self.y = spatial_y

    def __call__(self) -> torch.Tensor:
        if self.output[self.layer] is None:
            exit("Object callable only after forward pass!")
        if self.x is None:
            self.x = self.output[self.layer].shape[2] // 2
        if self.y is None:
            self.y = self.output[self.layer].shape[3] // 2
        neuron_tensor = self.output[self.layer][:, :, self.x:self.x+1, self.y:self.y+1]
        channel_tensor = neuron_tensor[:, self.channel]
        loss_local = -channel_tensor.mean()
        return self.scaler * loss_local


class Channel_Weight(Common):
    """ Linearly weighted channel activation as objective.

    Args:
        weight: a torch.Tensor vector of same lenght as channels in
                the specified layer
                (e.g weight = torch.rand(256, device=device)

    """
    def __init__(self, layer, weight) -> None:
        super().__init__(layer)
        self.weight = weight

    def __call__(self) -> torch.Tensor:
        loss_local = -(self.output[self.layer] * self.weight.view(1, -1, 1, 1)).mean()
        return self.scaler * loss_local


class Direction(Common):
    """ Visualize a direction

    Args:
        layer: specific layer in model (passed as type and not string
                as of 2022-12-01)
        direction: direction to visualize. torch.Tensor of shape
                (num_channels,)

    Returns:
        loss. torch.Tensor

    """
    def __init__(self, layer, direction) -> None:
        super().__init__(layer)
        self.direction = direction

    def __call__(self) -> torch.Tensor:
        # after reshaping the random direction to represent a tensor of 4
        # dimensions with the second one being the channels, it returns
        # the negative of the mean of the cosine similarity between them
        loss_local = -nn.CosineSimilarity(dim=1)(self.direction.reshape((1,
                                                 -1, 1, 1)),
                                                 self.output[self.layer]).mean()
        return self.scaler * loss_local


class Channel_Interpolate(Common):
    def __init__(self, layer, channel, layer_2, channel_2):
        super().__init__(layer)
        self.channel = channel
        self.layer_2 = layer_2
        self.hook_2 = layer_2.register_forward_hook(self.hook_function)
        self.channel_2 = channel_2

    def __call__(self) -> torch.Tensor:
        if self.output[self.layer] is None or self.output[self.layer_2] is None:
            exit("Object callable only after forward pass!")
        batch_number = list(self.output[self.layer].shape)[0]
        channel_tensor_1 = self.output[self.layer][:, self.channel]
        channel_tensor_2 = self.output[self.layer_2][:, self.channel_2]
        weights = np.arange(batch_number) / (batch_number - 1)
        weights = torch.from_numpy(weights)
        sum_loss = torch.zeros(1).to(device)
        for n in range(batch_number):
            sum_loss -= (1 - weights[n]) * channel_tensor_1[n].mean()
            sum_loss -= weights[n] * channel_tensor_2[n].mean()
        return self.scaler * sum_loss

class WRT_Classes(Common):
    """This class displays 10 images (thus needs a batched input
        with a shape of [num_classes, 3, H, W]) adding the logits of the
        diagonal of the [num_classes, num_classes] output tensor (i.e.
        the fc linear layer).

        Output is 10 images, each maximizing the specified channel and
        one of the available output classes, since the logits of that class
        are to be maximized to induce a greater loss.
        
        The loop advances sum_loss accessing the channel tensor for batch
        number n (the first image is indexed as batch_num=0)
        and adding the diagonal element corresponding to each class.
        By doing so, the upper leftmost element of the fc layer's output (0,0)
        referring to the logit of the first class of the classification task is
        added to the first image of the batch (the value of which has already
        been reduced to a scalar).

        1.5e-3 seems to yield the most legible results, since the logits need
        to be scaled, otherwise they overrule the visualization, obfuscating
        any other objective. 
        """
    def __init__(self, layer, channel, model):
        super().__init__(layer)
        self.channel = channel
        self.layer_2 = model.fc
        self.hook_2 = self.layer_2.register_forward_hook(self.hook_function)
        self.classes_num = model.fc.out_features

    def __call__(self) -> torch.Tensor:
        if self.output[self.layer] is None or self.output[self.layer_2] is None:
            exit("Object callable only after forward pass!")
        # batch_number = list(self.output[self.layer].shape)[0]  # must always be 10
        channel_tensor = self.output[self.layer][:, self.channel]
        sum_loss = torch.zeros(1).to(device)
        for n in range(self.classes_num):
            sum_loss -= channel_tensor[n].mean() + (1.5e-3 * self.output[self.layer_2][n, n])
        return self.scaler * sum_loss



class Diversity_Obj(Common):
    def __init__(self, layer, channel) -> None:
        super().__init__(layer)
        self.channel = channel

    def __call__(self)-> torch.Tensor:
        if self.output[self.layer] is None:
            exit("Object callable only after forward pass!")
        batch, channels, _, _ = self.output[self.layer].shape
        channel_tensor = self.output[self.layer][:, self.channel]
        flattened = self.output[self.layer].view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        loss_local = -channel_tensor.mean()
        return loss_local - self.scaler * (-sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch)

class Diversity_Obj_2(Common):
    def __init__(self, layer) -> None:
        super().__init__(layer)

    def __call__(self)-> torch.Tensor:
        if self.output[self.layer] is None:
            exit("Object callable only after forward pass!")
        batch, channels, _, _ = self.output[self.layer].shape
        flattened = self.output[self.layer].view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return - self.scaler * (-sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch)