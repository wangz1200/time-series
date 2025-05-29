from urllib.parse import urljoin
import requests
from .base import *


__all__ = (
    "kline",
)


class _Util(object):

    def __init__(self):
        super(_Util, self).__init__()


class _Signal(object):

    def __init__(self):
        super(_Signal, self).__init__()
        self.util = _Util()

    def query(
            self,
            product: str,
            period: str,
            timestamp: str | int,
            limit: int = 1024,
            score: float = 0.5,
    ):
        t = modal.db.Kline(dao=S.dao).t
        stmt = S.dao.select(t).where(
            t.c.product == product,
            t.c.period == period,
        ).order_by(
            t.c.timestamp.desc()
        )
        flag = False
        if limit:
            stmt = stmt.limit(limit)
            flag = True
        if timestamp:
            stmt = stmt.where(
                t.c.timestamp <= int(timestamp),
            )
            flag = True
        if not flag:
            raise Exception("参数错误")
        list_ = S.dao.list_(
            rows=S.dao.execute(stmt)
        )
        list_ = list_[::-1]
        timestamp = int(list_[-1]["timestamp"])
        url = urljoin(S.signal.url, "signal")
        data = [{
            "open": l["open"],
            "close": l["close"],
            "high": l["high"],
            "low": l["low"],
            "volume": l["volume"],
            "open_interest": l["open_interest"],
        } for l in list_]
        pred = requests.post(
            url=url,
            json={
                "product": product,
                "period": period,
                "data": data,
            },
        ).json()
        if pred["code"] != 0:
            raise Exception(pred["msg"])
        return {
            "id": f"{product}-{period}-{timestamp}",
            "timestamp": timestamp,
            **pred["data"]["pred"]
        }

    def add(
            self,
            data: dict | list[dict],
            update: bool | None = None,
            tx: sa.Connection | None = None,
    ):
        data = shared.util.list_(data)
        if len(data) == 0:
            raise Exception("数据为空")
        modal.db.KlineSignal(dao=S.dao).insert(
            data=data,
            update=update,
            tx=tx,
        )

    def remove(
            self,
            id_: str | list[str],
            tx: sa.Connection | None = None,
    ):
        id_ = shared.util.list_(id_)
        modal.db.KlineSignal(dao=S.dao).delete(
            id_=id_,
            tx=tx,
        )

    def sync(
            self,
            data: dict | list[dict],
    ):
        data = shared.util.list_(data)
        if not data:
            return
        insert_ = []
        delete_ = []
        for d in data:
            if d["signal"] == 0:
                delete_.append(d["id"])
            else:
                insert_.append(d)
        if not insert_ and not delete_:
            return
        with S.dao.trans() as tx:
            if delete_:
                modal.db.KlineSignal(dao=S.dao).delete(
                    id_=delete_,
                    tx=tx
                )
            if insert_:
                modal.db.KlineSignal(dao=S.dao).insert(
                    data=insert_,
                    update=True,
                    tx=tx,
                )


class Kline(object):

    def __init__(self):
        super().__init__()
        self.signal = _Signal()

    def query(
            self,
            product: str,
            period: str,
            begin_datetime: str | None = None,
            end_datetime: str | None = None,
            timestamp: int | None = None,
            limit: int = 500,
    ):
        if not product and not period:
            raise Exception("product or period is empty")
        t_kline = modal.db.Kline(dao=S.dao).t
        t_signal = modal.db.KlineSignal(dao=S.dao).t
        stmt = S.dao.select(
            t_kline.c.id.label("id"),
            t_kline.c.exchange.label("exchange"),
            t_kline.c.product.label("product"),
            t_kline.c.period.label("period"),
            t_kline.c.open.label("open"),
            t_kline.c.close.label("close"),
            t_kline.c.high.label("high"),
            t_kline.c.low.label("low"),
            t_kline.c.volume.label("volume"),
            t_kline.c.open_interest.label("open_interest"),
            t_kline.c.datetime.label("datetime"),
            t_kline.c.timestamp.label("timestamp"),
            sa.func.coalesce(t_signal.c.signal, 0).label("signal"),
            sa.func.coalesce(t_signal.c.score, 0).label("score"),
        ).select_from(
            t_kline.outerjoin(t_signal, t_kline.c.id == t_signal.c.id)
        ).where(
            t_kline.c.product == product,
            t_kline.c.period == period,
        ).order_by(t_kline.c.timestamp)
        if limit:
            stmt = stmt.limit(limit)
        if begin_datetime:
            stmt = stmt.where(
                t_kline.c.datetime >= begin_datetime,
            )
        if end_datetime:
            stmt = stmt.where(
                t_kline.c.datetime <= end_datetime,
            )
        list_ = S.dao.list_(
            rows=S.dao.execute(stmt)
        )
        return list_

    def add(
            self,
            data: dict | list[dict],
            update: bool | None = None,
    ):
        data = shared.util.list_(data)
        modal.db.Kline(dao=S.dao).insert(
            data=data,
            update=update,
        )

    def remove(
            self,
            id_: int | str = 0,
            exchange: str = "",
            product: str = "",
            period: str = "",
    ):
        pass


kline = Kline()
