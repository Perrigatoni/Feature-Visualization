from __future__ import absolute_import, print_function

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34
from torch.utils.data import DataLoader
from tqdm import tqdm
import Feature_Visualization.models.scratch_tiny_plain

def module_fill(model):
    module_dict = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                # underscored_name = name.replace('.', '_')
                module_dict[name] = mod
    return module_dict


class Hook_Layer:
    def __init__(self, layer) -> None:
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.output = None

    def hook_fn(self, layer, input, output):
        self.output = output

    def __call__(self):
        return self.output


# Create a dataset class that extends ImageFolder while
# simultaneously returning a 3 way Tuple, instead of the
# original that contains 2 elements.
# For that reason we must define a new __getitem__ method.
class ImageFolderWithPaths(ImageFolder):
    """Dataset class extending ImageFolder dataset,
    returning Tuple.

    Returns:
            Tuple[img[torch.Tensor],
                  label[int],
                  path[str]]
    """

    def __getitem__(self, index: int):
        # Super the __getitem__ of base class
        img, label = super().__getitem__(index)
        # Extract the path of each image in the dataset
        path = self.imgs[index][0]
        # Return new tuple with 3 elements
        return (img, label, path)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    weights = None
    model = resnet34(weights=weights)
    # model = ResNet10()

    # reshape last layer.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    model.load_state_dict(
        torch.load(
            r"C:\Users\Noel\Documents\THESIS\Feature Visualization\Weights\resnet34_torchvision\test72_epoch446.pth"
        )
    )
    # Set model to evaluation mode and send to device
    model.to(device).eval()

    module_dict = module_fill(model)

    layer_activations = {}
    for name, layer in module_dict.items():
        layer_activations[name] = Hook_Layer(layer)

    batch_size = 16
    transforms = T.Compose(
        [
            T.Resize(224),
            # T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.5162, 0.4644, 0.3975], [0.2724, 0.2640, 0.2574]),
        ]
    )

    dataset = ImageFolderWithPaths(
        root=r"C:\Users\Noel\Documents\THESIS\Data\artbench-10-imagefolder-split\train",
        transform=transforms,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    print("Dataloader Initialized. Workers Warming up I guess ;)")
    # ================================================================
    data = []

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, total=len(dataloader)):
            # Send stuff to GPU if available.
            images = images.to(device)
            labels = labels.to(device)
            # Make Forward Pass.
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            # path_list = []
            # for path in paths:
            #     path_list.append(path)

            for i, image in enumerate(images):
                private_dict = {}
                # Three entries regarding the image identification.
                # private_dict['path'] = path_list[i]
                private_dict["path"] = paths[i]
                private_dict["class_label"] = labels[i].item()
                private_dict["prediction"] = preds[i].item()
                # Iterate over all available layers.
                for key, hook_object in layer_activations.items():
                    tensor_out = hook_object()  # .output  # modified
                    # from original script to accommodate objects
                    if key == "fc":
                        # The array to store is a 32 by 10 array, each batch
                        output = torch.unbind(tensor_out, dim=0)
                    else:
                        # The array 'll have a final shape of 32*num_channels
                        # in specific layer
                        b, c, _, _ = tensor_out.shape
                        output = torch.unbind(tensor_out.view(b, c, -1).mean(2), dim=0)
                    private_dict[key] = output[i].cpu().numpy()
                data.append(private_dict)

    df = pd.DataFrame(data, copy=False)
    df.to_parquet(
        r"C:\Users\Noel\Documents\THESIS\Feature Visualization\Dataframes\test72_activations.parquet"
    )


if __name__ == "__main__":
    main()
