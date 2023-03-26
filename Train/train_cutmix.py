from __future__ import division, print_function

import time
import tracemalloc
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_cutmix(model,
                dataloaders,
                optimizer,
                scheduler,
                criterion,
                num_classes,
                num_epochs=25,
                save_path=None,
                is_inception=False,
                logging=False,
                log_path=None):
    """ This function handles training and validation.
        After the total calculation, saves the best performing
        model weights and returns the newly parameterized model
        and a list of previously achieved accuracies
        that were overshadowed after a better one was hit.
        """
    
    if log_path is None and logging:
        log_path = 'default_logs'
    writer = SummaryWriter(log_path)
    # Specify the confmat
    # confmat = ConfusionMatrix(task='binary',
    #                           num_classes=num_classes).to(device)
    # Get the time of start.
    since = time.time()
    # Create an empty list where the validation accuracy
    # of each epoch is to be stored for future reference.
    # test_acc_history = []
    # Deepcopy the model's state dictionary as best model weights.
    # ---------> state_dict is a python dictionary object
    #  mapping each layer to its parameter tensor.
    # ---------> copy.deeepcopy means that we create a new compound
    #  object with copies of the objects found in the original,
    #  which is different from shallow copying which inserts references
    #  to the objects of the original.
    best_model_wts = copy.deepcopy(model.state_dict())
    # Initialize best accuracy at zero.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Note: This training function implements cutmix.')
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        tracemalloc.start()  # !
        # Reminder! Each epoch has both a training and a validation step.
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Setting model to training mode.
            else:
                model.eval()   # Setting model to evaluation mode.

            # Define the starting loss for each epoch @ 0.0.
            running_loss = 0.0
            # Amount of corrects.
            running_corrects = 0
            confusion_accumulator = np.zeros((num_classes, num_classes))
            # Iterate over the data.
            # The dataloaders dictionary has two entries, one for the
            # phase of training and one for validation
            for inputs, labels in tqdm(dataloaders[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out the gradient before each forward step.
                # In fact this zeros the parameters' gradients.
                optimizer.zero_grad()

                # Calculate forward pass, keeping track history
                # only when in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calsulate loss.
                    # Define a special case for inception, accomodating
                    # the aux output during training phase.
                    # Accomodate any other case of model.
                    # Forward pass, equal to calling model.forward(inputs)
                    # but avoid actually calling the forward
                    # method like this!
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)  # actual forward
                        # pass, exactly as is, returning the output prediction.
                        # Loss calculation based on the predetermined loss
                        # function.
                        # This loss is calculated from the difference
                        # between the prediction and the actual output.
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        if phase == 'train':
                            if np.random.rand(1) < 0.5:
                                beta = 1
                                lam = np.random.beta(beta, beta)
                                rand_index = torch.randperm(inputs.size()[0]).cuda()
                                target_a = labels
                                target_b = labels[rand_index]
                                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                                # images = inputs
                                # display_out(images)
                                # input()
                                # adjust lambda to exactly match pixel ratio
                                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                                outputs = model(inputs)
                                """ DO NOT CALL THE FORWARD METHOD,
                                    IT CAN AFFECT THE HOOKS STORED
                                    DURING THE FORWARD PASS!"""
                                # loss = criterion(outputs, labels)
                                loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                            else:
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        # change this dim to simply ,1 if it doesn't comply
                    _, preds = torch.max(outputs, dim=1)
                    """ Confusion Matrix """
                    if phase == 'test':
                        # Confusion matrix for that batch
                        b_cm = confusion_matrix(labels.detach().cpu().numpy(),
                                                preds.detach().cpu().numpy(),
                                                labels=[0, 1, 2, 3, 4,
                                                        5, 6, 7, 8, 9])
                        # print(batch_confmat)
                        confusion_accumulator = np.add(confusion_accumulator,
                                                       b_cm).astype(np.int16)  # ! DO I NEED IT TO BE LARGER?

                    # Calculate the Backward pass, only when in training phase
                    if phase == 'train':
                        # Calc backward graph on the loss.
                        loss.backward()
                        # Optimizer's step.
                        optimizer.step()

                """PERFORMANCE METRICS"""
                # Advance the running loss
                running_loss += loss.item() * inputs.size(0)
                # Why multiply by the inputs?
                # --> seems like this is because we are passing a batch,
                #     so we take the total loss on it.

                # Advance the correct prediction counters
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'test':
                # print(confusion_accumulator)
                confmat = ConfusionMatrixDisplay(confusion_accumulator,
                                                 display_labels=['art_n.',
                                                                 'bar.',
                                                                 'expres.',
                                                                 'impres.',
                                                                 'p_impres.',
                                                                 'real.',
                                                                 'ren.',
                                                                 'rom.',
                                                                 'sur.',
                                                                 'ukiyo.'])
                confmat.plot(xticks_rotation='vertical',
                             colorbar=False)
                # plt.show()
            
            # Calculate total epoch loss
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # Calculate epoch accuracy
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if logging:
                writer.add_scalar('{} Loss:'.format(phase), epoch_loss, epoch)
                writer.add_scalar('{} Acc:'.format(phase), epoch_acc, epoch)
                
            # Print the above results to console
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,
                                                       epoch_loss,
                                                       epoch_acc))

            """If model accuracy improves during current validation phase,
                we must deepcopy the new model's state_dict() to match the
                parameter tensor to each layer. That way we ensure that
                the weights are getting updated on our model with the best
                available, produced of course by the training step prior
                to validation.
                """

            # Deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                # Display confusion matrix in Tensorboard/figures tab.
                if logging:
                    writer.add_figure("Best Confusion Matrix",
                                      figure=confmat.figure_, global_step=epoch)
                print(confusion_accumulator)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the confmat as figure too.
                # confmat.figure_.savefig('bestconfmat.png')   # SAVE CONFUSION MATRIX TO ROOT DIRECTORY
                plt.close()
                # Check if save_path is defined to save state dictionary
                # after each advancing epoch.
                if save_path is not None:
                    torch.save(model.state_dict(), save_path.format(epoch))
            # Append to history list.
            if phase == 'test':
                # test_acc_history.append(epoch_acc)
                # Scheduler
                last_lr = optimizer.param_groups[0]['lr']
                # if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                #     scheduler.step(epoch_loss)
                #     last_lr = scheduler._last_lr[0]
                if scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts':
                    scheduler.step()
                    last_lr = scheduler.get_last_lr()[0]
                if logging:
                    writer.add_scalar("Learning Rate:", last_lr, epoch)
        if logging:
            mem_trace = tracemalloc.get_traced_memory()  # ! tuple[current, peak]
            writer.add_scalar('Memory Usage', mem_trace[1], epoch)  # !
        tracemalloc.stop()
    # Total Runtime
    time_elapsed = time.time() - since
    # Print the total runtime calculated above.
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    # Print the best accuracy achieved during a validation step.
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights.
    model.load_state_dict(best_model_wts)
    return model  #, test_acc_history


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


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
    return Image.fromarray(image).show()