import datetime
from pathlib import Path
import requests
from pythongo.base import BaseParams, Field
from pythongo.ui import BaseStrategy
from pythongo.classdef import KLineData, OrderData, TickData, TradeData, Position
from pythongo.utils import KLineGenerator


class Params(BaseParams):

    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="v2505", title="合约代码")
    period: str = Field(default="M5", title="K线周期")
    url: str = Field(default="http://47.101.153.166:8000", title="数据服务器")
    signal: bool = Field(default=False, title="更新信号")


class KLine(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.params_map = Params()
        self.kline_generator: KLineGenerator | None = None
        self.initializing = False
        self.file_path = None
        self.timestamp = 0
        self.rows = []

    def on_start(self):
        super().on_start()
        self.initializing = True
        self.kline_generator = KLineGenerator(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.period,
            callback=self.callback,
            real_time_callback=None,
            # real_time_callback=self.real_time_callback,
        )
        self.kline_generator.push_history_data()
        self.initializing = False
        if self.rows:
            res = self.upload_to_server(
                data=self.rows,
                update=True,
            )
            self.output("[initializing] upload data to server:", res)
            self.rows = []

    def on_stop(self):
        super().on_stop()

    def on_tick(
            self,
            tick: TickData,
    ):
        super().on_tick(tick)
        self.kline_generator.tick_to_kline(tick)

    def upload_to_server(
            self,
            data: dict | list,
            update: bool = False,
    ):
        res = requests.post(
            url=self.params_map.url + "/kline",
            json={
                "data": data,
                "update": update,
            }
        )
        return res.json()

    def sync_signal(
            self,
            product: str,
            period: str,
    ):
        res = requests.post(
            url=self.params_map.url + "/kline/signal/sync",
            json={
                "product": product,
                "period": period,
            }
        )
        return res.json()

    def callback(
            self,
            kline: KLineData
    ):
        data = kline.to_json()
        data["exchange"] = self.params_map.exchange.lower()
        data["product"] = self.params_map.instrument_id.lower()
        data["period"] = self.params_map.period.lower()
        data["timestamp"] = int(data["datetime"].timestamp() * 1000)
        data["datetime"] = data["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        if self.initializing:
            self.rows.append(data)
        else:
            self.upload_to_server(
                data=data,
                update=True,
            )
            if self.params_map.signal:
                res = self.sync_signal(
                    product=self.params_map.instrument_id,
                    period=self.params_map.period.lower(),
                )
                self.output(f"signal: {data['product']} - {data['period']} - {data['datetime']} - {res}")
            self.output(f"upload: {data['product']} - {data['period']} - {data['datetime']}")

    def real_time_callback(
            self,
            kline: KLineData
    ):
        self.widget.recv_kline({
            "kline": kline,
        })

