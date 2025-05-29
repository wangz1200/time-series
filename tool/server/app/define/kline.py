from .base import *


class Item(BaseModel):

    id: str | int | None = None
    exchange: str | None = None
    product: str | None = None
    period: str | None = None
    open: float | None = None
    close: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    datetime: str = ""
    timestamp: int = 0


class AddReq(BaseModel):

    data: Item | list[Item]
    update: bool | None = None

    def data_(self):
        rows = []
        data = shared.util.list_(self.data)
        for d in data:
            rows.append({
                "id": f"{d.product}-{d.period}-{d.timestamp}",
                "exchange": d.exchange or "",
                "product": d.product or "",
                "period": d.period or "",
                "open": d.open or 0,
                "close": d.close or 0,
                "high": d.high or 0,
                "low": d.low or 0,
                "volume": d.volume or 0,
                "open_interest": d.open_interest or 0,
                "datetime": d.datetime or "1999-12-31 23:59",
                "timestamp": d.timestamp or 0,
            })
        return rows


