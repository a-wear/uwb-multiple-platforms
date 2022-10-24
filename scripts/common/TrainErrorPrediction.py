'''
This file is loosely based on the training method for Deep Imbalanced Regression
found at: https://github.com/YyzHarry/imbalanced-regression

We use the focal L1 loss implemented there but not (yet) the LDS and FDS methods.
The dataset parsers were adjusted to fit our dataset.
'''
import time
import argparse
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

from common.FCNet import FCNet
from common.loss import *
from common.CirErrorDataset import CirErrorDataset
from common.utils_torch import *

import os

def plot_true_pred_pdf(y_true, y_pred, filename=""):
    global file_root_run
    try:
        leny = y_true.size
    except AttributeError:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

    print("test size: ", y_true.shape)

    # Plot distribution of predicted and true y
    miny = np.min(np.hstack((y_true, y_pred)))
    maxy = np.max(np.hstack((y_true, y_pred)))

    fig, ax = plt.subplots()
    bins = np.linspace(miny, maxy, 100)

    y = y_true
    hist, _ = np.histogram(y, bins, density=False)
    hist = hist / y.size
    ax.bar(bins[:-1], hist, width=bins[1] - bins[0], alpha=0.3, label="y test")

    y = y_pred
    hist, _ = np.histogram(y, bins, density=False)
    hist = hist / y.size
    ax.bar(bins[:-1], hist, width=bins[1] - bins[0], alpha=0.3, label="y pred")

    ax.legend()
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("PDF")

    if len(filename) > 0:
        filedir = "./images/train_results_images/"
        if not os.path.exists(filedir): # Create dir if it doesn't exist
            os.makedirs(filedir)
        filepath = os.path.join(filedir, filename)
        fig.savefig(filepath, format="pdf")
        print("Saved ", filepath)
    # plt.show()

def plot_cdf_before_after(y_true, y_pred, filename=""):
    try:
        leny = y_true.size
    except AttributeError:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

    y_corr = np.abs(y_true - y_pred)
    y_true = np.abs(y_true)

    # Plot distribution of predicted and true y
    miny = np.min(np.hstack((y_true, y_corr)))
    maxy = np.max(np.hstack((y_true, y_corr)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(miny, maxy, 100)

    y = y_true
    hist, _ = np.histogram(y, bins, density=False)
    hist = hist / y.size
    cdf = np.cumsum(hist)
    ax.plot(bins[:-1], cdf, label="Initial errors, m = {:.2f} m, std = {:.2f} m".format(
        np.mean(y), np.std(y)))

    y = y_corr
    hist, _ = np.histogram(y, bins, density=False)
    hist = hist / y.size
    cdf = np.cumsum(hist)
    ax.plot(bins[:-1], cdf, label="Corrected errors, m = {:.2f} m, std = {:.2f} m".format(
        np.mean(y), np.std(y)))

    ax.legend()
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("CDF")
    ax.set_ylim([0, 1])
    ax.grid("--")
    fig.suptitle("Absolute errors")
    fig.tight_layout()

    if len(filename) > 0:
        filedir = "./images/train_results_images/"
        if not os.path.exists(filedir): # Create dir if it doesn't exist
            os.makedirs(filedir)
        filepath = os.path.join(filedir, filename)
        fig.savefig(filepath, format="pdf")
        print("Saved ", filepath)
    plt.show()

class TrainErrorPrediction():
    def __init__(self, train_device, dataset_name, args):
        self.train_device = train_device
        self.root_dataset = args.root_dataset
        self.dataset_name = dataset_name
        self.root_checkpoints = args.root_checkpoints
        self.use_smogn = args.use_smogn
        self.nr_layers = args.layers
        self.optimizer_type = args.optimizer
        self.loss = args.loss
        self.lr = args.lr
        self.epoch = args.epoch
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.schedule = args.schedule
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.workers = args.workers
        self.resume = args.resume
        self.patience = args.patience
        self.start_epoch = 0
        self.best_loss = 1e5
        self.file_root_run = args.file_root_run

        self.root_dataset = os.path.join(self.root_dataset, self.dataset_name)

        print("File root for this run: ", self.file_root_run)

        if self.use_smogn:
            self.smogn_file_root = "_smogn"
        else:
            self.smogn_file_root = ""

        # Create dir for checkpoints
        if args.resume:
            # Initialize checkpoint dir based on resume
            resume_dirs = self.resume.split("/")
            # Last field is the file name, 2nd to last is dir_checkpoint, the rest is root
            self.dir_checkpoint = resume_dirs[-2]
            self.root_checkpoints = "/".join(resume_dirs[:-2])
            print("resume: ", self.resume)
            print("Dir checkpoint: ", self.dir_checkpoint)
            print("Root checkpoint: ", self.root_checkpoints)
        else:
            self.dir_checkpoint = "train_{}{}_{}".format(
                self.dataset_name, self.smogn_file_root, self.file_root_run)
            if not os.path.exists(os.path.join(self.root_checkpoints, self.dir_checkpoint)):
                os.makedirs(os.path.join(self.root_checkpoints, self.dir_checkpoint))
            print("Checkpoint dir: ", self.dir_checkpoint)

        self.train_loader = None; self.test_loader = None; self.val_loader = None
        self.in_data_size = 0
        self.y_true = None; self.y_pred = None

    def load_datasets(self):
        if self.use_smogn:
            try:
                self.train_dataset = CirErrorDataset(
                    data_dir=os.path.join(self.root_dataset, "train_smogn.csv"),
                    split="train")
            except FileNotFoundError:
                print("Warning! SMOGN augmented train set not found! Continuing with regular one.")
                self.train_dataset = CirErrorDataset(
                    data_dir=os.path.join(self.root_dataset, "train.csv"),
                    split="train")
                self.smogn_file_root = ""
        else:
            self.train_dataset = CirErrorDataset(
                data_dir=os.path.join(self.root_dataset, "train.csv"),
                split="train")

        self.val_dataset = CirErrorDataset(
            data_dir=os.path.join(self.root_dataset, "val.csv"),
            split="val")
        self.test_dataset = CirErrorDataset(
            data_dir=os.path.join(self.root_dataset, "test.csv"),
            split="test")

        self.in_data_size = self.train_dataset[0][0].size
        self.train_labels = self.train_dataset._get_labels()

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True, drop_last=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True, drop_last=False)

        print("Training set size: ", len(self.train_dataset))
        print("Validation set size: ", len(self.val_dataset))
        print("Test set size: ", len(self.test_dataset))

    def init_model(self):
        if self.in_data_size == 0 or self.train_loader is None:
            raise ValueError("First load the datasets!")

        layers = [self.nr_layers for i in range(3)]
        # layers = [128, 256, 128]
        self.model = FCNet(layers=layers, in_size=self.in_data_size)
        summary(self.model, input_size=(self.batch_size, 1, self.in_data_size))

        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                momentum=self.momentum, weight_decay=self.weight_decay)

        # If checkpoint provided, resume model training
        if self.resume:
            if os.path.isfile(self.resume):
                print(f"===> Loading checkpoint '{self.resume}'")
                self.checkpoint = torch.load(self.resume)
                self.start_epoch = self.checkpoint['epoch']
                self.best_loss = self.checkpoint['best_loss']
                self.model.load_state_dict(self.checkpoint['state_dict'])
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                print("===> Loaded checkpoint '{}' (Epoch {})".format(
                    self.resume, self.checkpoint['epoch']))
            else:
                print("===> No checkpoint found at '{}'".format(self.resume))

    def train_all(self):
        curr_patience = 0
        for epoch in range(self.start_epoch, self.epoch):
            adjust_learning_rate(self.optimizer, epoch, self.lr, self.schedule)
            train_loss = self.train(
                self.train_loader, self.model, self.optimizer, epoch)
            _, _, results = self.validate(
                self.val_loader, self.model, train_labels=self.train_labels)

            val_loss_mse = results["mse"]
            val_loss_l1 = results["l1"]
            val_loss_gmean = results["gmean"]
            loss_metric = val_loss_mse if self.loss == 'mse' else val_loss_l1
            is_best = loss_metric < self.best_loss
            self.best_loss = min(loss_metric, self.best_loss)

            if is_best:
                curr_patience = 0
            else:
                curr_patience += 1

            if "l1" in self.loss:
                print("Best L1 Loss: {:.3f}".format(self.best_loss))
            else:
                print("Best MSE Loss: {:.3f}".format(self.best_loss))

            state = {
                'epoch': epoch + 1,
                'model': "fcnet1",
                'best_loss': self.best_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            self.save_checkpoint(
                save_path=os.path.join(self.root_checkpoints, self.dir_checkpoint),
                state=state, is_best=is_best)

            print("Epoch #{}: Train loss: [{:.4f}]; Val loss: MSE [{:.4f}], L1 [{:.4f}], G-Mean [{:.4f}]".format(
                epoch, train_loss, val_loss_mse, val_loss_l1, val_loss_gmean))

            if curr_patience >= self.patience:
                print("Early stopping")
                break

    def train(self, train_loader, model, optimizer, epoch):
        batch_time = AverageMeter('Time', ':6.2f')
        data_time = AverageMeter('Data', ':6.4f')
        losses = AverageMeter(f'Loss ({self.loss.upper()})', ':.3f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch)
        )

        model.train()
        end = time.time()
        for idx, (inputs, targets, weights) in enumerate(train_loader):
            data_time.update(time.time() - end)
            outputs = model(inputs, targets, epoch)

            loss = globals()[f"weighted_{self.loss}_loss"](outputs, targets, weights)
            assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

            losses.update(loss.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % self.print_freq == 0:
                progress.display(idx)

        return losses.avg

    def test(self):
        print("=" * 120)
        print("Test best model on testset...")
        self.checkpoint = torch.load(
            os.path.join(self.root_checkpoints, self.dir_checkpoint,
                         "ckpt.best.pth.tar"))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        print("Loaded best model, epoch {}, best val loss: {:.4f}".format(
            self.checkpoint['epoch'], self.checkpoint['best_loss']))

        self.y_true, self.y_pred, results = \
            self.validate(
                val_loader=self.test_loader, model=self.model,
                train_labels=self.train_labels, prefix='Test',
                do_plot=True, do_save_pred=True)

        print("Test loss: MSE [{:.4f}], L1 [{:.4f}], G-mean [{:.4f}]".format(
            results["mse"], results["l1"], results["gmean"]))

        return results


    def validate(self, val_loader, model, train_labels=None, prefix='Val',
        do_plot=False, do_save_pred=False):

        batch_time = AverageMeter('Time', ':6.3f')
        losses_mse = AverageMeter('Loss (MSE)', ':.3f')
        losses_l1 = AverageMeter('Loss (L1)', ':.3f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses_mse, losses_l1],
            prefix=f'{prefix}: '
        )

        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()
        criterion_gmean = nn.L1Loss(reduction='none')

        model.eval()
        losses_all = []
        preds, labels = [], []
        with torch.no_grad():
            end = time.time()
            for idx, (inputs, targets, _) in enumerate(val_loader):
                outputs = model(inputs)

                preds.extend(outputs.data.cpu().numpy())
                labels.extend(targets.data.cpu().numpy())

                loss_mse = criterion_mse(outputs, targets)
                loss_l1 = criterion_l1(outputs, targets)
                loss_all = criterion_gmean(outputs, targets)
                losses_all.extend(loss_all.cpu().numpy())

                losses_mse.update(loss_mse.item(), inputs.size(0))
                losses_l1.update(loss_l1.item(), inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                if idx % self.print_freq == 0:
                    progress.display(idx)

            loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
            print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")

        abs_err_before = np.mean(np.abs(labels))
        abs_err_after = np.mean(np.abs(np.array(labels) - np.array(preds)))
        print("Mean abs error before: {:.4f} m".format(abs_err_before))
        print("Mean abs error after correction: {:.4f} m".format(abs_err_after))

        if do_plot:
            filename = "cdf_before_after_{}_{}.pdf".format(
                self.file_root_run, self.dataset_name)
            plot_cdf_before_after(y_true=labels, y_pred=preds, filename=filename)

        if do_save_pred:
            df = pd.DataFrame()
            df["y_pred"] = np.array(preds).reshape((-1,))
            df["y_true"] = np.array(labels).reshape((-1,))
            filename = "pred_true_dist_err.csv"
            df.to_csv(
                os.path.join(self.root_checkpoints, self.dir_checkpoint, filename),
                index=False)
            print("Wrote ", filename)

        results = {
            "mse": losses_mse.avg,
            "l1": losses_l1.avg,
            "gmean": loss_gmean,
            "abs_err_before": abs_err_before,
            "abs_err_after": abs_err_after,
            "std_before": np.std(np.abs(labels)),
            "std_after": np.std(np.abs(np.array(labels) - np.array(preds))),
        }
        return labels, preds, results

    def save_checkpoint(self, save_path, state, is_best):
        filepath = os.path.join(save_path, "ckpt.pth.tar")
        torch.save(state, filepath)
        if is_best:
            print("===> Saving current best checkpoint...")
            shutil.copyfile(filepath,
                os.path.join(save_path, "ckpt.best.pth.tar"))

    def get_predictions(self):
        return self.y_true, self.y_pred

    def save_results(self):
        pass # TODO
