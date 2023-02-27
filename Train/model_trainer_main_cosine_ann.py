from __future__ import division, print_function

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model_initializer import initialize_model
from train import train_model

# from scratch_tiny_resnet import ResNet10, ResNet18


def main():

    print("PyTorch Version: ", torch.__version__)
    print("TorchVision Version: ", torchvision.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Hyper Parameters
    test_num = 45
    log_with_TB = True
    parameters_saved_as = "test"+str(test_num)+"_epoch{}.pth"
    model_name = "resnet"
    num_classes = 10
    batch_size = 384
    num_epochs = 400
    feature_extract = False
    pretrained = False
    change_activ_func = True
    starting_lr = 0.008
    weight_decay = 0.01

    # Specify directories
    data_dir = r"/home/periclesstamatis/artbench-10-imagefolder-split"
    # data_dir = r"C:\Users\Noel\Documents\THESIS STUFF\Data\artbench-10-imagefolder-split"

    param_path = os.path.join(r"/home/periclesstamatis/saved_model_parameters",
                              "test"+str(test_num))

    # Initializing the model for our run
    model_ft, input_size = initialize_model(model_name, num_classes,
                                            feature_extract,
                                            use_pretrained=pretrained,
                                            change_AF=change_activ_func)
    # Send model to the GPU
    model_ft = model_ft.to(device)
    print(model_ft._modules)

    # Data augmentation and normalization for training/validation/testing
    data_transforms = {
        'train': T.Compose([T.RandomResizedCrop(input_size),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.RandomAffine(degrees=0,
                                           translate=(0.1, 0.1),
                                           scale=(0.7, 1.3),
                                           interpolation=InterpolationMode.BILINEAR),
                            T.ToTensor(),
                            T.Normalize([0.5162, 0.4644, 0.3975],
                                        [0.2724, 0.2640, 0.2574]),
                            T.RandomErasing()
                            # T.Normalize([0.4897, 0.4370, 0.3717],
                            #             [0.2850, 0.2713, 0.2587]),
                            ]),
        'test': T.Compose([T.Resize(input_size),
                           T.CenterCrop(input_size),
                           T.ToTensor(),
                           T.Normalize([0.5162, 0.4644, 0.3975],
                                       [0.2724, 0.2640, 0.2574])
                           ]),
    }

    # Create training and validation datasets.
    print('Initializing Datasets and Dataloaders for both phases...')
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}

    # Create training and validation dataloaders.
    dataloaders_dict = {'train':
                        DataLoader(image_datasets['train'],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4),
                        'test':
                        DataLoader(image_datasets['test'],
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=4),
                        }

    # print(example_inputs[0].shape)
    log_path = None
    writer = SummaryWriter(log_path)
    if log_with_TB:
        log_path = os.path.join('log_artbench', 'test'+str(test_num))
        writer = SummaryWriter(log_path)
        # View some basic loading information
        examples = iter(dataloaders_dict['train'])
        example_inputs, _ = next(examples)
        # img_grid = torchvision.utils.make_grid(example_inputs)
        # writer.add_image('First batch of train dataloader.', img_grid)
        writer.add_graph(model_ft, example_inputs.to(device))

    # Gather parameters to update. reminder that feature extraction and
    # finetuning have different requirements for parameter updates.
    # Pass the execution of model's parameters method to a variable
    # for later use.
    params_to_update = model_ft.parameters()
    print('Params to learn:')
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad is True:
                print("\t", name)

    # Optimizer and Scheduler
    optimizer_ft = optim.Adam(params_to_update,
                             lr=starting_lr,
                             weight_decay=weight_decay)

    # scheduler_ft = optim.lr_scheduler.MultiStepLR(optimizer_ft,
    #                                               milestones=[30, 50, 60],
    #                                               gamma=0.3,
    #                                               verbose=True)
    scheduler_ft = CosineAnnealingWarmRestarts(optimizer_ft,
                                               T_0=50,
                                               T_mult=1,
                                               eta_min=1e-9,
                                               verbose=True)

    if log_with_TB:
        writer.add_text('test'+str(test_num),
                        f'Initial Lr :{starting_lr}\n \
                        Batch Size :{batch_size}\n \
                        Weight Decay :{weight_decay}\n \
                        Pretrained :{pretrained}\n \
                        Optimizer :{optimizer_ft.__class__.__name__}\n \
                        Scheduler :{scheduler_ft.__class__.__name__}\n')
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Training and Evaluation

    # path = r"C:\Users\Noel\Documents\THESIS STUFF"
    # parameters_saved_as = f"lr{starting_lr}_optimizer{optimizer_ft.__class__.__name__}_scheduler{}_pretrained{}_epoch{}.pth"
    if os.path.exists(param_path) is False:
        os.mkdir(param_path)
    save_path = os.path.join(param_path, parameters_saved_as)
    # save_path = r"C:\Users\Noel\Documents\THESIS STUFF\PYTHON-THINGIES\
    #               Saved Model Parameters\
    #               resnet34_torchvision_stepLR_SGD\testing_unit.pth"
    model_ft, hist = train_model(model_ft,
                                 dataloaders_dict,
                                 optimizer_ft,
                                 scheduler_ft,
                                 criterion,
                                 num_classes=num_classes,
                                 num_epochs=num_epochs,
                                 save_path=save_path,
                                 is_inception=(model_name == ('inception')),
                                 logging=log_with_TB,
                                 log_path=log_path)

    # Cumulative save of best weights.
    # torch.save(hist, r"/home/periclesstamatis/saved_model_parameters/
    #                    2023_01_27/resnet34torchvision_artbench_test_logging")
    if log_with_TB:
        writer.close()


if __name__ == "__main__":
    main()
