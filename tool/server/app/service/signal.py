import requests
from .base import *


class Signal(object):

    def __init__(
            self,
            host: str = "127.0.0.1",
            port: str | int = 9000,
    ):
        super().__init__()
        self.host = None
        self.port = None
        self.url = None
        self.init(
            host=host,
            port=port,
        )

    def init(
            self,
            host: str = "127.0.0.1",
            port: str | int = 9000,
    ):
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}/signal"

    def query(
            self,
            product: str,
            period: str,
            begin_datetime: str | None = None,
            end_datetime: str | None = None,
            timestamp: int | None = None,
            limit: int = 96,
    ):
        t = shared.state.dao.table["kline"]
        stmt = shared.state.dao.select(
            t.c.open.label("open"),
            t.c.close.label("close"),
            t.c.high.label("high"),
            t.c.low.label("low"),
            t.c.volume.label("volume"),
            t.c.open_interest.label("open_interest"),
        ).where(
            t.c.product == product,
            t.c.period == period,
        ).limit(
            limit=limit,
        ).order_by(
            t.c.timestamp.desc(),
        )
        if begin_datetime:
            stmt = stmt.where(
                t.c.begin_date >= begin_datetime
            )
        if end_datetime:
            stmt = stmt.where(
                t.c.begin_date <= end_datetime
            )
        if timestamp:
            stmt = stmt.where(
                t.c.timestamp <= timestamp
            )
        rows = shared.state.dao.list_(
            shared.state.dao.execute(stmt),
        )
        rows = rows[::-1]
        id_ = rows[-1]["id"]
        ret = requests.post(
            url=self.url,
            json=rows,
        ).json()
        if ret["code"] != 0:
            raise Exception(ret["msg"])
        return {
            id_: ret["data"],
        }

    def sync(
            self,
            product: str,
            period: str,
            timestamp: int | None = None,
    ):
        pass

signal = Signal()
