from __future__ import division, print_function

import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model,
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
    if log_path is None:
        log_path = 'default_logs'
    writer = SummaryWriter(log_path)
    # Specify the confmat
    # confmat = ConfusionMatrix(task='binary',
    #                           num_classes=num_classes).to(device)
    # Get the time of start.
    since = time.time()
    # Create an empty list where the validation accuracy
    # of each epoch is to be stored for future reference.
    test_acc_history = []
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
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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
                    else:  # Accomodate any other case of model.
                        # Forward pass, equal to calling model.forward(inputs)
                        # but avoid actually calling the forward
                        # method like this!
                        outputs = model(inputs)
                        """ DO NOT CALL THE FORWARD METHOD,
                            IT CAN AFFECT THE HOOKS STORED
                            DURING THE FORWARD PASS!"""
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
                                                       b_cm).astype(np.int64)  # !

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
                                                 display_labels=['art_nouveau',
                                                                 'baroque',
                                                                 'expressionism',
                                                                 'impressionism',
                                                                 'post_impressionism',
                                                                 'realism',
                                                                 'renaissance',
                                                                 'romanticism',
                                                                 'surrealism',
                                                                 'ukiyo-e'])
                confmat.plot(xticks_rotation='vertical',
                             colorbar=False)
                plt.show()

            # Calculate total epoch loss
            epoch_loss = running_loss / \
                len(dataloaders[phase].dataset)
            # Calculate epoch accuracy
            epoch_acc = running_corrects.double() / \
                len(dataloaders[phase].dataset)

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
                # Check if save_path is defined to save state dictionary
                # after each advancing epoch.
                if save_path is not None:
                    torch.save(model.state_dict(), save_path.format(epoch))
            # Append to history list.
            if phase == 'test':
                test_acc_history.append(epoch_acc)
                # Scheduler
                last_lr = optimizer.param_groups[0]['lr']
                if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    scheduler.step(epoch_loss)
                    last_lr = scheduler._last_lr[0]
                if scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts':
                    scheduler.step()
                    last_lr = scheduler.get_last_lr()[0]
                if logging:
                    writer.add_scalar("Learning Rate:", last_lr, epoch)

    # Total Runtime
    time_elapsed = time.time() - since
    # Print the total runtime calculated above.
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    # Print the best accuracy achieved during a validation step.
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights.
    model.load_state_dict(best_model_wts)
    return model, test_acc_history
