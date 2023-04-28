from __future__ import division, print_function
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from torchvision import datasets
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


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


def main():
    print("PyTorch Version: ", torch.__version__)
    print("TorchVision Version: ", torchvision.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    test_num = 65
    num_classes = 10
    input_size = 224
    batch_size = 32
    data_dir = r"C:\Users\Noel\Documents\THESIS\Data\artbench-10-imagefolder-split\test"
    # data_dir = "/home/periclesstamatis/artbench-10-imagefolder-split/test"
    # Load model with no weights.
    model = models.resnet18(weights=None)
    # Reshape last layer.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load("C://Users//Noel//Documents//THESIS"
                                    "//Feature Visualization//Weights"
                                    "//resnet18_torchvision"
                                    "//test65_epoch598.pth"))
    # model.load_state_dict(torch.load("/home/periclesstamatis/"
    #                                  "saved_model_parameters/test65"
    #                                  "/test65_epoch598.pth"))
    model.to(device)
    # Conversion of ReLU activation function to LeakyReLU.
    module_convertor(model,
                     nn.ReLU,
                     nn.LeakyReLU(inplace=True))

    data_transforms = T.Compose([T.Resize(input_size),
                                T.ToTensor(),
                                T.Normalize([0.5162, 0.4644, 0.3975],
                                            [0.2724, 0.2640, 0.2574])
                                 ])

    dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            )

    """ This function handles training and validation.
        After the total calculation, saves the best performing
        model weights and returns the newly parameterized model
        and a list of previously achieved accuracies
        that were overshadowed after a better one was hit.
        """
    # Evaluation mode.
    model.eval()
    # Amount of corrects.
    running_corrects = 0
    # Init Confusion Matrix
    confusion_accumulator = np.zeros((num_classes, num_classes))

    # Iterate over the data.
    # The dataloaders dictionary has two entries, one for the
    # phase of training and one for validation
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            """ Confusion Matrix """
            batch_confmat = confusion_matrix(labels.detach().cpu().numpy(),
                                            preds.detach().cpu().numpy(),
                                            labels=[0, 1, 2, 3, 4,
                                                    5, 6, 7, 8, 9])
            # print(batch_confmat)
            confusion_accumulator = np.add(confusion_accumulator,
                                        batch_confmat).astype(np.int32)  # !

        """PERFORMANCE METRICS"""
        running_corrects += torch.sum(preds == labels.data)

    indices = ['art_nouveau', 'baroque', 'expressionism', 'impressionism',
               'post_impressionism.', 'realism', 'renaissance', 'romanticism',
               'surrealism', 'ukiyo_e']
    # print(confusion_accumulator)
    confmat = ConfusionMatrixDisplay(confusion_accumulator,
                                     display_labels=indices
                                     )
    confmat.plot(xticks_rotation='vertical',
                 colorbar=False)
    # plt.show()

    # Calculate epoch accuracy
    acc = running_corrects.double() / len(dataset)
    print(f'Mean Accuracy across classes: {acc.item() * 100 :.4f}%')

    df = pd.DataFrame(data=confusion_accumulator,
                      index=indices,
                      columns=indices)
    print(df)
    # path = f'/home/periclesstamatis/stats/dataframe_test{test_num}'
    # df.to_parquet(path=path)

    # Save the confmat as figure too.
    confmat.figure_.savefig(f'bestconfmat_test{test_num}.png',
                            bbox_inches='tight')


if __name__ == "__main__":
    main()
