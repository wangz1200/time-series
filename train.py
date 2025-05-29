import argparse
import copy
import os
import random
import time
from functools import partial
from pathlib import Path
import math
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Trainer(object):

    def __init__(
            self,
            args,
    ):
        super().__init__()
        self.args = args
        self.device = self.acquire_device()
        self.model = None

    def acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def get_data(
            self,
            flag: str = "train",
            mean: list | None = None,
            std: list | None = None,
            shuffle_flag: bool | None = None,
    ):
        data = FutureDataset(
            args=self.args,
            flag=flag,
            mean=mean,
            std=std,
        )
        loader = FutureDataloader(
            args=self.args,
            dataset=data,
            shuffle_flag=shuffle_flag
        )
        return data, loader

    def get_optim(
            self,
            model: nn.Module,
    ):
        return torch.optim.RAdam(
            model.parameters(),
            lr=self.args.learning_rate,
        )

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def cal_accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)

    def save_model(
            self,
            name: str,
    ):
        if not self.model:
            raise ValueError("model is None")
        torch.save(
            self.model.state_dict(),
            self.args.output + "/" + name,
        )

    def train(
            self,
            eval: bool = False
    ):
        path = Path(self.args.output)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        mean = std = None
        if self.args.mean:
            mean = [float(v) for v in self.args.mean.split(',')]
        if self.args.std:
            std = [float(v) for v in self.args.std.split(',')]
        data, loader = self.get_data(
            flag="train", mean=mean, std=std
        )
        print(f"train data len: {len(data)}")
        eval_data, eval_loader = None, None
        if eval:
            eval_data, eval_loader = self.get_data(
                flag="eval", mean=mean, std=std
            )
            print(f"eval data len: {len(eval_data)}")
        self.model = Model(
            seq_len=data.seq_len,
            e_layers=args.e_layers,
            embed=args.embed,
            freq=args.freq,
            enc_in=data.dims,
            d_model=args.d_model,
            d_ff=args.d_ff,
            top_k=args.top_k,
            num_kernels=args.num_kernels,
            dropout=args.dropout,
            num_class=len(data.names),
            mean=data.mean,
            std=data.std,
        )
        self.model = self.model.to(self.device)
        optim = self.get_optim(self.model)
        criterion = self.get_criterion()
        time_now = time.time()
        train_steps = len(loader)
        global_loss = 10000
        patience = args.patience
        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            preds = []
            trues = []
            for i, (batch_x, label, padding_mask) in enumerate(loader):
                iter_count += 1
                optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())
                preds.append(outputs.detach())
                trues.append(label)
                if (i + 1) % self.args.iter == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print(f"[train] - epoch: {epoch + 1} | iters: {i + 1} | loss: {loss.item():.8f} | speed: {speed:.2f}s/iter | left time: {left_time:.2f}s")
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optim.step()
            train_loss = np.average(train_loss)
            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            probs = F.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
            predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
            trues = trues.flatten().cpu().numpy()
            accuracy = np.mean(predictions == trues)
            cost_time = time.time() - epoch_time
            print(f"[train] - epoch: {epoch + 1} | loss: {train_loss:.8f} | accuracy: {accuracy:.4f} | patience: {args.patience - patience} | cost time: {cost_time:.2f}s")
            self.save_model(f"checkpoint_{epoch}.pth")
            with Path(self.args.output, "config.yaml").open(
                    "w+", encoding="utf8"
            ) as f:
                data = {
                    "model": self.model.config()
                }
                yaml.dump(data, f)
            train_loss = np.average(train_loss)
            if train_loss < global_loss:
                global_loss = train_loss
                patience = 10
                self.save_model(f"checkpoint_best.pth")
            else:
                patience -= 1
            if eval:
                self._eval(self.model, eval_loader)
            if patience == 0:
                break
            # eval_loss, eval_accuracy = self.eval(model, eval_data, eval_loader)
            # print(
            #     f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.8f} Vali Loss: {eval_loss:.8f} Vali Acc: {eval_accuracy:.4f}"
            # )

    def _eval(
            self,
            model: nn.Module,
            loader
    ):
        total_loss = []
        preds = []
        trues = []
        criterion = self.get_criterion()
        model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = model(batch_x, padding_mask, None, None)
                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)
                preds.append(outputs.detach())
                trues.append(label)
        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = F.softmax(preds, dim=0)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = np.mean(predictions == trues)
        print(f"[eval] - loss: {total_loss:.8f} | accuracy: {accuracy:.4f}")
        return total_loss, accuracy

    def eval(
            self,
            model: nn.Module | None = None
    ):
        model = model or self.model
        data, loader = self.get_data(
            flag="eval",
            mean=model.mean,
            std=model.std,
            shuffle_flag=False,
        )
        return self._eval(model, loader)


def init_args():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='Future')
    # basic config
    parser.add_argument('--training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='future', help='model id')
    # data loader
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--output', type=str, default='./output/', help='location of model checkpoints')
    parser.add_argument('--shuffle_flag', type=bool, default=True, help="")
    parser.add_argument('--drop_last', type=bool, default=False, help="")
    parser.add_argument('--mean', type=str, default="", help="")
    parser.add_argument('--std', type=str, default="", help="")
    parser.add_argument('--iter', type=int, default=100, help="")
    # model define
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--num_kernels', type=int, default=6, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', default=True, action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    args = init_args()
    args.use_gpu = True
    args.gpu = 3
    # v-m5-768
    args.data_dir = "./future/dataset/v-m5-768"
    args.output = "./output/v-m5-768"
    args.mean = "5975.44101517153,5975.378367559345,5979.6195508586525,5971.161282574757,3286.795044233618,220217.93443016693"
    args.std = "379.76050964706246,379.8207488747327,379.3841481409951,380.1733797817888,7782.143808369144,325615.355034679"
    # v-m15-256
    # args.data_dir = "./dataset/v-m15-256"
    # args.output = "./output/v-m15-256"
    # args.mean = "6049.488226340764, 12101.578923806966, 241726.94941355538"
    # args.std = "324.8078921058797, 27176.4525441869, 335655.1490411683"
    args.batch_size = 16
    args.epochs = 30
    args.iter = 100
    args.top_k = 3
    args.d_model = 64
    args.d_ff = 64
    args.e_layers = 3
    args.num_kernels = 6
    args.learning_rate = 0.0001
    trainer = Trainer(args)
    training = 1
    eval_ = 1
    if training:
        trainer.train(
            eval=eval_ > 0
        )
    else:
        model = Model.from_config(
            model_dir=args.output,
        )
        trainer.eval(model)
