import time
import copy

import torch
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import scipy

import torch.optim as optim
from torch.optim import lr_scheduler


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time()
    model.cuda()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0
    early = 0

    for epoch in range(num_epochs):
        time_epoch = time.time()

        # cyclical learning rate
        if epoch % 200 == 0:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)

        print('Epoch {}/{} {}'.format(epoch, num_epochs - 1, early))
        print('-' * 10)
        if early >= 250:
            model.load_state_dict(best_model_wts)
            return model

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            epoch_losses = 0.0
            deno = 0.0
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                labels = labels.type(torch.FloatTensor)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    aa = time.time()
                    outputs = model(inputs)
                    preds = outputs.squeeze(1)
                    preds = preds.type(torch.FloatTensor)
                    loss = criterion(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        aa = time.time()
                        loss.backward()
                        optimizer.step()

                del inputs;
                del outputs
                epoch_losses += loss.item() * len(preds)
                #                epoch_losses += loss.data[0] * len(preds)
                deno += len(preds)
                del preds

            epoch_loss = epoch_losses / deno
            print('{} Loss: {:.4f} {}'.format(phase, epoch_loss, deno))
            del deno

            # torch.cuda.empty_cache()

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early = 0
            if phase == 'val' and epoch_loss > best_loss:
                early += 1

            # stop if there is no convergence....
            # if phase == 'val' and best_loss > 2 and epoch >= 50:
            #    model.load_state_dict(best_model_wts)
            #    return model

            # now predict for test set
            if phase == 'val':
                pred = []
                obs = []
                for inputs, labels in dataloaders['test']:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    for i in range(len(labels)):
                        pred.append(outputs.data[i])
                        obs.append(labels.data[i])
                    del labels, outputs, inputs
                pred = np.array([x.item() for x in pred])
                obs = np.array([x.item() for x in obs])
                rms = sqrt(mean_squared_error(pred, obs))
                r2 = scipy.stats.pearsonr(pred, obs)
                print('test Loss: {:.4f} {}'.format(rms, r2[0]))
                del pred, obs, rms, r2

        print('Epoch complete in {:.0f}m {:.0f}s'.format((time.time() - time_epoch) // 60,
                                                         (time.time() - time_epoch) % 60))

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model
