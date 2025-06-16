import copy
import datetime
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import tushare as ts
import sqlalchemy as sa


class TuShare(object):

    def __init__(
            self,
            token: str = '',
    ):
        super().__init__()
        self.token = token
        self.client = ts.pro_api(
            token=self.token,
        )

    def fetch(
            self,
            product: str,
            exchange: str,
            period: str,
            start_date: str,
            end_date: str,
            save_dir: str | Path | None = None,
    ):
        data = pd.DataFrame()
        end_ = end_date
        while True:
            df = self.client.ft_mins(
                ts_code=f"{product.upper()}.{exchange.upper()}",
                freq=period,
                start_date=start_date,
                end_date=end_,
            )
            if df.empty:
                break
            data = pd.concat([data, df], ignore_index=True)
            last = df.tail(1)
            end_ = last['trade_time'].values[0]
            end_ =(
                datetime.datetime.strptime(end_, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(seconds=1)
            ).strftime('%Y-%m-%d %H:%M:%S')
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = f'{product.lower()}.{exchange.lower()}_{period.lower()}_{start_date.replace(":", "-")}-{end_date.replace(":", "-")}.csv'
            file_path = save_dir / file_name
            data.to_csv(file_path, index=False)
        return data

    @classmethod
    def create_engine(
            cls,
            host='localhost',
            port=3306,
            user="root",
            password="root",
            name="future",
    ):
        engine = sa.create_engine(sa.URL.create(
            drivername="mysql+pymysql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=name,
        ))
        return engine

    def sync(
            self,
            product: str,
            exchange: str,
            period: str,
            start_date: str,
            end_date: str,
            engine: sa.Engine,
            batch_size: int = 1000,
    ):
        data = self.fetch(
            product=product,
            exchange=exchange,
            period=period,
            start_date=start_date,
            end_date=end_date,
        )
        values = []
        for _, d in data.iterrows():
            values.append({
                "product": product.lower(),
                "exchange": exchange.lower(),
                "period": period.lower(),
                "open": d["open"],
                "close": d["close"],
                "high": d["high"],
                "low": d["low"],
                "volume": d["vol"],
                "amount": d["amount"],
                "oi": d["oi"],
                # "datetime": datetime.datetime.strptime(d["trade_time"], "%Y-%m-%d %H:%M:%S"),
                "datetime": d["trade_time"],
            })
        if not values:
            raise Exception("数据为空。")
        stmt = (
            "INSERT INTO detail (product, exchange, period, open, high, low, close, volume, amount, oi, datetime) "
            "VALUES (:product, :exchange, :period, :open, :high, :low, :close, :volume, :amount, :oi, :datetime) "
            "ON DUPLICATE KEY UPDATE "
            "open = VALUES(open), "
            "high = VALUES(high), "
            "low = VALUES(low), "
            "close = VALUES(close), " 
            "volume = VALUES(volume), "
            "amount = VALUES(amount), "
            "oi = VALUES(oi)"
        )
        stmt = sa.text(stmt)
        with engine.begin() as tx:
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                tx.execute(stmt, batch)

    def calculate_daily_high_low(
            self,
            period: str,
            product: str,
            host: str = "127.0.0.1",
            port: int = 3306,
            user: str = "root",
            password: str = "root",
            name: str = "future",
    ):
        engine = self.create_engine(
            host=host,
            port=port,
            user=user,
            password=password,
            name=name,
        )
        stmt = (
            "select product, opwn, close, high, low from detail "
            f"where product = '{product}' and period = '{period}' order by datetime"
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
        return torch.tensor(batch_x), torch.tensor(labels)

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


if __name__ == "__main__":
    with Path("./config/config.yaml").open(
            mode="r",
            encoding="utf8",
    ) as f:
        config = yaml.safe_load(f)
    token = config["tushare"]['token']
    dataset = TuShare(
        token=token,
    )
    # dataset.fetch(
    #     product="rb2510",
    #     exchange="shf",
    #     period="1min",
    #     start_date="2024-09-30 00:00:00",
    #     end_date="2025-12-31 00:00:00",
    #     save_dir="./data"
    # )
    dataset.sync(
        product="rb2510",
        exchange="shf",
        period="",
        start_date="2024-09-30 00:00:00",
        end_date="2025-12-31 00:00:00",
        engine=TuShare.create_engine()
    )
