#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import datetime
import copy
from param_loader import ParamLoader


def smoother(x, y_rough, kind_of_filter, l_win, r_win, b):
    # x is the input of the NN (coord x and y of the past trajectory)
    # y_rough is the x-y coordinate of the future trajectory
    y_smooth = copy.deepcopy(y_rough)
    # GAUSSIAN FILTER
    if kind_of_filter == 0:
        idx = np.arange(x.shape[1] + y_rough.shape[1])
        data = np.append(x, y_rough, axis=0)
        rng = np.arange(-l_win, r_win+1, step=1)
        gk = np.exp(- ((rng ** 2) / (2 * (b ** 2))))
        gk_std = gk / gk.sum()
        plt.figure("Gaussian kernel")
        plt.plot(gk, label='gk')
        plt.plot(gk_std, label='gk_std')
        plt.legend()
        for i in np.arange(y_rough.shape[0] - r_win):
            start = x.shape[0] + i - l_win
            end = x.shape[0] + i + r_win
            debug = (data[start:end+1, 0] * gk_std).sum()
            y_smooth[i, 0] = (data[start:end+1, 0] * gk_std).sum()
            y_smooth[i, 1] = (data[start:end+1, 1] * gk_std).sum()
    return y_smooth


class TrainTestEva:
    def __init__(self, model, loss_fn, optimizer, optimizer_tl, device, tl):
        self.model        = model
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.optimizer_tl = optimizer_tl
        self.train_losses = []
        self.test_losses  = []
        self.device       = device
        self.tl           = tl

    def train_step(self, x, y):                     # single train step: x=input, y=known output, y_pred=NN pred
        self.model.train()                          # set the model in train mode
        y_pred = self.model(x)[0]                   # make the prediction, not taking (hn, cn) since not used
        y_pred = y_pred.reshape(y.shape)
        # from (n_of_batches, headers * len_ out) -> (n_of_batches, len_ out, headers) = y.shape
        loss = self.loss_fn(y, y_pred)              # compute loss
        loss.backward()                             # compute gradients
        self.optimizer.step()                       # update param according to the optimizer
        self.optimizer.zero_grad()                  # zeroing the grad
        return loss.item()                          # return the loss as a number

    def train_step_tl(self, x, y):
        # Some layer are frozen

        # self.model.train()                        # set the model in train mode
        y_pred = self.model(x)[0]                   # make the prediction, not taking (hn, cn) since not used
        y_pred = y_pred.reshape(y.shape)
        loss = self.loss_fn(y, y_pred)              # compute loss
        loss.backward()                             # compute gradients
        self.optimizer_tl.step()                    # update param according to the optimizer
        self.optimizer.zero_grad()  # zeroing the grad
        return loss.item()

    def train_loop(self, train_loader, test_loader, n_epochs, n_features, scheduler):
        # At each epoch, iterate on batches; at each batch train step (update weights)
        # Then validate on the test set
        self.train_losses = []
        self.test_losses = []

        if self.tl:
            print(f"Training phase (lstm frozen!)...")
            for param in self.model.lstm.parameters():
                param.requires_grad = False
        else:
            print(f"Training phase...")
            for param in self.model.lstm.parameters():        # probably True by default
                param.requires_grad = True
        print("qq4")

        # at each epoch train  on all the trains_loader dataset, train_loader dataset is organized in batches
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            batch_losses = []
            # TRAIN
            print("loading...")
            # print(train_loader)

            for x_batch, y_batch in train_loader:

                x_batch = x_batch.view([x_batch.shape[0], -1, n_features]).to(self.device).float()

                y_batch = y_batch.to(self.device).float()

                if self.tl:
                    loss = self.train_step_tl(x_batch, y_batch)  # loss of single train example
                else:
                    loss = self.train_step(x_batch, y_batch)  # loss of single train example
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)           # mean-losses on the batch
            self.train_losses.append(training_loss)         # append the value

            # VALIDATION
            with torch.no_grad():                           # (~ disable gradient computation to speed up)
                batch_test_losses = []

                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.view([x_batch.shape[0], -1, n_features]).to(self.device).float()
                    y_batch = y_batch.to(self.device).float()
                    self.model.eval()
                    y_pred_batch, hidden = self.model(x_batch)
                    y_pred_batch = y_pred_batch.reshape(y_batch.shape)
                    # from (n_of_batches, headers * len_ out) -> (n_of_batches, len_ out, headers) = y_batch.shape
                    eval_batch_loss = self.loss_fn(y_batch, y_pred_batch).item()
                    batch_test_losses.append(eval_batch_loss)
                test_loss = np.mean(batch_test_losses)      # mean-losses on the batch
                self.test_losses.append(test_loss)          # append the value

            if epoch % 2 == 0 or epoch % 5 == 0 or epoch == 1 or epoch == 2 or epoch == 3 or epoch == 4 or epoch == 5:
                time_per_batch = time.time() - t0
                eta = str(datetime.timedelta(seconds=(n_epochs - epoch) * time_per_batch)).split('.', 2)[0]
                print(
                    f'Epoch: {epoch}/{n_epochs} ---> Train Loss: {training_loss:.8f}  -  Test Loss: {test_loss:.8f}'
                    f'  -  lr = {scheduler.get_last_lr()[0]:.8f}  -  time elapsed [s] = {time_per_batch:.0f}'
                    '  -  ETA [h:m:s] = ' + eta)

            if epoch > n_epochs:
                print('\n')
            scheduler.step()
        return self.model

    def evaluate(self, dataloader, num_samples, n_features, array_of_scaler):
        pl = ParamLoader()
        print('Evaluation...')
        # NB! Here 'x', 'y' means input and output of the NN, NOT COORDINATE!
        x_sample, y_sample = next(iter(dataloader))

        x_sample = x_sample.view([num_samples, -1, n_features]).to(
            self.device).float()                            # reshape since "Batch_first=True"
        y_sample = y_sample.to(self.device)                 # .float() in order to same data type

        with torch.no_grad():                               # (~ disable gradient computation to speed up)
            self.model.eval()
            y_eva, hidden_eva = self.model(x_sample)        # compute the prediction
            y_eva = y_eva.reshape(y_sample.shape)
            # from (n_of_batches, headers * len_ out) -> (n_of_batches, len_ out, headers) = y_sample.shape
            y_eva = y_eva.data.cpu().squeeze().numpy()      # move to cpu and numpy in order to plot!
            x_sample = x_sample.data.cpu().squeeze().numpy()
            y_sample = y_sample.data.cpu().squeeze().numpy()

        # SMOOTH THE NN's OUTPUT:
        y_eva_smoothed = np.empty_like(y_eva)
        for i in np.arange(num_samples):
            # debug_x_sample = x_sample[i, :, 0:2]
            # debug_y_eva = y_eva[i, :, :]
            y_tmp = smoother(x_sample[i, :, 0:2], y_eva[i, :, :], kind_of_filter=0, l_win=5, r_win=2, b=2)
            y_eva_smoothed[i, :, 0] = y_tmp[:, 0]
            y_eva_smoothed[i, :, 1] = y_tmp[:, 1]
            print('\n')

        plt_test = True         # plot the final tests
        if plt_test:
            for i in np.arange(num_samples):
                # plt.subplot(2, num_samples // 2 + num_samples % 2, i + 1)
                # Normalized output
                plt.figure(f"Plot test {i + 1}/{num_samples}")
                plt.plot(x_sample[i, :, 0], x_sample[i, :, 1], label="Input")
                plt.plot(y_sample[i, :, 0], y_sample[i, :, 1], label="Correct")
                plt.plot(y_eva[i, :, 0],    y_eva[i, :, 1],    label="Prediction")
                plt.legend()
                plt.grid(False)
                plt.title(f"Normalized: XY plane - Prediction n°: {i + 1}/{num_samples}")
                # Assuming header (0,1) of topic (0) are x,y of my desired predicted output
                file_name = "normal_pred_n_" + str(i + 1) + "_of_" + str(num_samples) + ".pdf"
                # print(file_name)
                # print(pl.data_path)
                plt.savefig(pl.data_path + '/plots/' + file_name)
                # print("problemi qua")
                # print(array_of_scaler)

                # DENORMALIZED:
                plt.figure(f"DENORM Plot test {i + 1}/{num_samples}")
                plt.plot(array_of_scaler[0, 0].inverse_transform(x_sample[i, :, 0].reshape(-1, 1)),
                         array_of_scaler[0, 1].inverse_transform(x_sample[i, :, 1].reshape(-1, 1)),
                         label="Input_DENORM")
                plt.plot(array_of_scaler[0, 0].inverse_transform(y_sample[i, :, 0].reshape(-1, 1)),
                         array_of_scaler[0, 1].inverse_transform(y_sample[i, :, 1].reshape(-1, 1)),
                         label="Correct_DENORM")
                plt.plot(array_of_scaler[0, 0].inverse_transform(y_eva[i, :, 0].reshape(-1, 1)),
                         array_of_scaler[0, 1].inverse_transform(y_eva[i, :, 1].reshape(-1, 1)),
                         label="Prediction_DENORM")
                plt.legend()
                plt.grid(False)
                plt.title(f"DENORM XY plane - Prediction n°: {i + 1}/{num_samples}")
                # Assuming header (0,1) of topic (0) are x,y of my desired predicted output
                file_name = "denorm_pred_n_" + str(i + 1) + "_of_" + str(num_samples) + ".pdf"
                plt.savefig(pl.data_path + '/plots/' + file_name)
                # # SMOOTHED:
                plt.figure(f"SMOOTHED Plot test {i + 1}/{num_samples}")
                plt.plot(x_sample[i, :, 0], x_sample[i, :, 1], label="Input_SMOOTH")
                plt.plot(y_sample[i, :, 0], y_sample[i, :, 1], label="Correct_SMOOTH")
                plt.plot(y_eva_smoothed[i, :, 0], y_eva_smoothed[i, :, 1], label="Prediction_SMOOTH")
                plt.legend()
                plt.grid(False)
                plt.title(f"SMOOTHED : XY plane - Prediction n°: {i + 1}/{num_samples}")
                # Assuming header (0,1) of topic (0) are x,y of my desired predicted output
                file_name = "SMOOTH_pred_n_" + str(i + 1) + "_of_" + str(num_samples) + ".pdf"
                plt.savefig(pl.data_path + '/plots/' + file_name)

    def plot_losses(self, pl = ParamLoader()):
        plt.figure("Training and test losses")
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(np.arange(1, len(self.test_losses) + 1), self.test_losses, label="Test loss")
        plt.legend()
        plt.yscale("log")
        if self.tl:
            plt.title("Transfer Learning case: Losses")
        else:
            plt.title("Losses")
        print("stampa qua")
        print(pl.data_path)
        plt.savefig(pl.data_path + '/plots/' + 'Loss_vs_epoch.pdf')