from .base import *


class Item(BaseModel):

    open: float = 0
    close: float = 0
    high: float = 0
    low: float = 0
    volume: float = 0
    open_interest: float = 0


class QueryReq(BaseModel):

    data: Item | List[Item]

    def data_(self):
        data = shared.util.list_(self.data)
        ret = []
        for d in data:
            ret.append({
                "open": d.open,
                "close": d.close,
                "high": d.high,
                "low": d.low,
                "volume": d.volume,
                "open_interest": d.open_interest,
            })
        return ret
