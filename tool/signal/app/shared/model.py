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


__all__ = (
    "Model",
)


class FutureDataset(Dataset):

    def __init__(
            self,
            args,
            flag: str = "train",
            mean: list | None = None,
            std: list | None = None,
    ):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.data_dir = Path(args.data_dir)
        self.flag = flag
        self.seq_len = 0
        self.names = None
        self.dims = None
        self.data = None
        self.labels = None
        self.ids = None
        self.mean = mean
        self.std = std
        self.load()

    def subsample(
            self, y,
            limit=256,
            factor=2,
    ):
        return (
            y[::factor].reset_index(drop=True)
            if len(y) > limit else y
        )

    def interpolate_missing(
            self,
            y
    ):
        """
        Replaces NaN values in pd.Series `y` using linear interpolation
        """
        if y.isna().any():
            y = y.interpolate(method="linear", limit_direction="both")
        return y

    def load(self):
        from sktime.datasets import load_from_tsfile_to_dataframe
        data_list = []
        label_list = []
        for file in [
            file.name
            for file in self.data_dir.rglob("*.ts")
            if self.flag.lower() in file.name.lower()
        ]:
            file_path = Path(self.data_dir, file).absolute().__str__()
            data, labels = load_from_tsfile_to_dataframe(
                full_file_path_and_name=file_path,
                return_separate_X_and_y=True,
                replace_missing_vals_with="NaN",
            )
            data_list.append(data)
            label_list.append(pd.Series(
                labels,
                dtype="category",
            ))
        if not data_list or not label_list:
            raise Exception("no data *.ts file found")
        self.labels = pd.concat(label_list, axis=0, ignore_index=True)
        self.data = pd.concat(data_list, axis=0, ignore_index=True)
        self.names = self.labels.cat.categories
        self.labels = pd.DataFrame(
            self.labels.cat.codes,
            dtype=np.int8,
        )
        # file_path = Path(self.data_dir, files[0]).absolute().__str__()
        # self.data, self.labels = load_from_tsfile_to_dataframe(
        #     full_file_path_and_name=file_path,
        #     return_separate_X_and_y=True,
        #     replace_missing_vals_with="NaN",
        # )
        # self.labels = pd.Series(
        #     self.labels,
        #     dtype="category",
        # )
        # self.names = self.labels.cat.categories
        # self.labels = pd.DataFrame(
        #     self.labels.cat.codes,
        #     dtype=np.int8,
        # )
        lengths = self.data.map(lambda x: len(x)).values
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            self.data = self.data.map(self.subsample)
        lengths = self.data.map(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.seq_len = int(np.max(lengths[:, 0]))
        else:
            self.seq_len = int(lengths[0, 0])
        self.data = pd.concat(
            (
                pd.DataFrame(
                    {col: self.data.loc[row, col] for col in self.data.columns}
                )
                .reset_index(drop=True)
                .set_index(
                    pd.Series(lengths[row, 0] * [row])
                ) for row in range(self.data.shape[0])
            ),
            axis=0
        )
        grp = self.data.groupby(by=self.data.index)
        self.data = grp.transform(self.interpolate_missing)
        self.ids = self.data.index.unique()
        self.dims = len(self.data.columns)
        if not self.mean:
            self.mean = self.data.mean().tolist()
        if not self.std:
            self.std = self.data.std().tolist()
        self.data = (self.data - self.mean) / (self.std + np.finfo(float).eps)

    def __getitem__(
            self,
            id_: int,
    ):
        batch_x = self.data.loc[self.ids[id_]].values
        labels = self.labels.loc[self.ids[id_]].values
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.ids)


class FutureDataloader(DataLoader):

    def __init__(
            self,
            args,
            dataset: FutureDataset,
            shuffle_flag: bool | None = None,
    ):
        self.args = args
        if shuffle_flag is None:
            shuffle_flag = args.shuffle_flag
        super().__init__(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            collate_fn=partial(self.collate_fn, max_len=dataset.seq_len)
        )

    def padding_mask(
            self,
            lengths,
            max_len=None,
    ):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
        return torch.arange(
            0,
            max_len,
            device=lengths.device
        ).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))

    def collate_fn(
            self,
            data,
            max_len=None,
    ):
        """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
        Args:
            data: len(batch_size) list of tuples (X, y).
                - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
                - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                    (for classification or regression, respectively). num_labels > 1 for multi-task models
            max_len: global fixed sequence length. Used for architectures requiring fixed length input,
                where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
        Returns:
            X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
            targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
            target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
                0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
            padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        """
        batch_size = len(data)
        features, labels = zip(*data)
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [x.shape[0] for x in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)
        x = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            x[i, :end, :] = features[i][:end, :]
        targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)
        padding_masks = self.padding_mask(
            torch.tensor(lengths, dtype=torch.int16),
            max_len=max_len,
        )  # (batch_size, padded_length) boolean tensor, "1" means keep
        return x, targets, padding_masks


class PositionalEmbedding(nn.Module):

    def __init__(
            self,
            d_model,
            max_len=5000
    ):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):

    def __init__(
            self,
            c_in, d_model,
    ):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(
            x.permute(0, 2, 1)
        ).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):

    def __init__(
            self,
            c_in, d_model,
    ):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):

    def __init__(
            self,
            d_model,
            embed_type="fixed",
            freq="h"
    ):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):

    def __init__(
            self,
            d_model,
            embed_type="timeF",
            freq="h",
    ):
        super().__init__()
        freq_map = {
            "h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3,
        }
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):

    def __init__(
            self,
            c_in, d_model,
            embed_type="fixed",
            freq="h",
            dropout=0.1,
    ):
        super().__init__()
        self.value_embedding = TokenEmbedding(
            c_in=c_in,
            d_model=d_model,
        )
        self.position_embedding = PositionalEmbedding(
            d_model=d_model,
        )
        self.temporal_embedding = (
            TimeFeatureEmbedding(
                d_model=d_model,
                embed_type=embed_type,
                freq=freq
            ) if embed_type.lower() == 'timef'
            else TemporalEmbedding(
                d_model=d_model,
                embed_type=embed_type,
                freq=freq
            )
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            x, x_mark
    ):
        x = self.value_embedding(x) + self.position_embedding(x)
        if x_mark:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)


class InceptionBlockV1(nn.Module):

    def __init__(
            self,
            in_channels, out_channels,
            num_kernels=6,
            init_weight=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                )
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class InceptionBlockV2(nn.Module):

    def __init__(
            self,
            in_channels, out_channels,
            num_kernels=6,
            init_weight=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 2 * i + 3, ),
                    padding=(0, i + 1, )
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(2 * i + 3, 1, ),
                    padding=(i + 1, 0, )
                ),
            ])
        kernels.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):

    def __init__(
            self,
            seq_len: int = 96,
            d_model: int = 32,
            d_ff: int = 32,
            top_k: int = 3,
            num_kernels: int = 6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def fft(
            self,
            x,
            k: int = 3,
    ):
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = self.fft(x, self.top_k)
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = (
                     (self.seq_len // period) + 1
                ) * period
                padding = torch.zeros(
                    [x.shape[0], (length - self.seq_len), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):

    @classmethod
    def from_config(
            cls,
            model_dir: str | Path,
    ):
        model_dir = Path(model_dir)
        config = []
        model_file = []
        for file in model_dir.iterdir():
            if file.name.endswith(".yaml"):
                config.append(file.name)
            if file.name.endswith(".pth"):
                model_file.append(file.name)
        if len(config) > 1 or len(model_file) > 1 or len(config) == 0 or len(model_file) == 0:
            raise Exception("config file or model file error")
        with Path(model_dir, config[0]).open(
                mode="r", encoding="utf8"
        ) as f:
            config = yaml.safe_load(f)
            model = cls(**config["model"])
            if model.use_gpu:
                device = torch.device(f"cuda:{model.gpus}")
            else:
                device = torch.device("cpu")
            model.load_state_dict(torch.load(
                Path(model_dir, model_file[0]),
                map_location=device,
                weights_only=True,
            ))
            model.to(device)
        return model

    def __init__(
            self,
            seq_len: int = 96,
            e_layers: int = 3,
            embed: str = "timeF",
            freq: str = "t",
            enc_in: int = 6,
            d_model: int = 32,
            d_ff: int = 32,
            top_k: int = 3,
            num_kernels: int = 6,
            dropout: float = 0.1,
            num_class: int = 2,
            mean: list | None = None,
            std: list | None = None,
            use_gpu: bool = True,
            gpus: str = "0",
    ):
        super().__init__()
        self.use_gpu = use_gpu
        self.gpus = gpus
        self.seq_len = seq_len
        self.e_layers = e_layers
        self.embed = embed
        self.freq = freq
        self.enc_in = enc_in
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout = dropout
        self.dropout_ = nn.Dropout(dropout)
        self.num_class = num_class
        self.mean = mean
        self.std = std
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout
        )
        self.backbone = nn.ModuleList([
            TimesBlock(
                seq_len=seq_len,
                num_kernels=num_kernels,
                d_model=d_model,
                d_ff=d_ff,
                top_k=top_k,
            ) for _ in range(self.e_layers)
        ])
        # self.projection = nn.Linear(
        #     configs.d_model * configs.seq_len, configs.num_class
        # )
        mid_dim = int(d_model * seq_len * 0.618)
        self.header = nn.Sequential(
            nn.Linear(
                d_model * seq_len, mid_dim,
            ),
            nn.GELU(),
            self.dropout_,
            nn.Linear(
                mid_dim, mid_dim,
            ),
            nn.GELU(),
            self.dropout_,
            nn.Linear(
                mid_dim, self.num_class,
            ),
        )
    def config(self):
        return {
            "use_gpu": self.use_gpu,
            "gpus": self.gpus,
            "seq_len": self.seq_len,
            "e_layers": self.e_layers,
            "embed": self.embed,
            "freq": self.freq,
            "enc_in": self.enc_in,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "top_k": self.top_k,
            "num_kernels": self.num_kernels,
            "dropout": self.dropout,
            "num_class": self.num_class,
            "mean": self.mean,
            "std": self.std,
        }

    def load(
            self,
            model_path: str | Path,
    ):
        self.load_state_dict(torch.load(
            Path(model_path).absolute().__str__(),
        ))

    def forward(
            self,
            x_enc,
            x_mark_enc = None,
            **kwargs,
    ):
        # embedding
        enc_out = self.embedding(x_enc, None)  # [B,T,C]
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.backbone[i](enc_out))
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = F.gelu(enc_out)
        output = self.dropout_(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1) if x_mark_enc else output
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.header(output)  # (batch_size, num_classes)
        return output
