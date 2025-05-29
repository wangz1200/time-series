import numpy as np
import torch
from torch.nn import functional as F
from .base import *
from .state import state


def query(
        kline: list[dict],
):
    kline = shared.util.list_(kline)
    seq_len = state.model.seq_len
    if len(kline) < state.model.seq_len:
        raise Exception(f"K线长度少于{seq_len}根")
    data = []
    for k in kline[-seq_len:]:
        data.append([
            float(k["close"]),
            float(k["volume"]),
            float(k["open_interest"]),
        ])
    with torch.no_grad():
        x = np.array(data)
        x = (x - state.model.mean) / state.model.std
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
        out = state.model.forward(x)
        out = F.softmax(out, dim=-1)
        out = out.cpu().numpy()
        return out.tolist()[0]


class Signal(object):

    def __init__(
            self,
            state: shared.state.State | None = None,
    ):
        super().__init__()
        self.state = state

    def query(self):
        pass