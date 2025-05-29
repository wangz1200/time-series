import datetime
import numpy as np
import talib
from pythongo.base import BaseParams, Field
from pythongo.ui import BaseStrategy
from pythongo.classdef import KLineData, OrderData, TickData, TradeData, Position
from pythongo.utils import KLineGenerator


class Params(BaseParams):

    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="v2409", title="合约代码")
    period: str = Field(default="M15", title="K线周期")
    indicator: str = Field(default="ma:ema,55;macd:13,21,8", title="指标参数")
    lep: str = Field(default="loss:-0.005;even:0.005;profit:0.015,0.24", title="止盈止损")
    order_volume: int = Field(default=1, title="下单手数", ge=1)
    pay_up: float | int = Field(default=0, title="超价", ge=0)


class State(object):

    def __init__(
            self,
            indicator: str = "ma:ema,55;macd:13,21,8",
            lep: str = "loss:-0.004;even:0.002;profit:0.01,0.24",
            pay_up: int | float = 1
    ):
        super().__init__()
        self._prices = np.array([])
        self._ma = []
        self._macd = []
        self.ma_type = "ema"
        self.ma_period = 55
        self.diff = 13
        self.dea = 21
        self.m = 8
        self.loss = -0.004
        self.even = 0.002
        self.profit = 0.01
        self.back = 0.24
        self.position = Position()
        self.klines = []
        self.signal = 0
        self.count = 0
        self.order: OrderData | None = OrderData()
        self.tick: TickData | None = TickData()
        self.pay_up: float | int = pay_up
        self.opening = True
        self.high = 0.0
        self.low = 99999999.0
        self.init(
            indicator=indicator,
            lep=lep,
        )

    def init(
            self,
            indicator: str = "ma:ma,55;macd:13,21,8",
            lep: str = "loss:-0.004;even:0.002;profit:0.01,0.24",
    ):
        indicator = indicator.replace("：", ":").replace("，", ",").replace(" ", "")
        lep = lep.replace("：", ":").replace("，", ",").replace(" ", "")
        for v1 in indicator.split(";"):
            v2 = v1.split(":")
            match v2[0].lower():
                case "ma":
                    v = v2[1].split(",")
                    self.ma_type = v[0].lower()
                    self.ma_period = int(v[1])
                case "macd":
                    v = v2[1].split(",")
                    self.diff = int(v[0])
                    self.dea = int(v[1])
                    self.m = int(v[2])
                case _:
                    raise ValueError("指标参数错误")
        for v1 in lep.split(";"):
            v2 = v1.split(":")
            match v2[0].lower():
                case "loss":
                    v = v2[1].split(",")
                    self.loss = float(v[0])
                case "even":
                    v = v2[1].split(",")
                    self.even = float(v[0])
                case "profit":
                    v = v2[1].split(",")
                    self.profit = float(v[0])
                    self.back = float(v[1])

    # @property
    # def high(self):
    #     v = [k.high for k in self.klines]
    #     if self.position.long.position > 0:
    #         v.append(self.position.long.open_avg_price)
    #     elif self.position.short.position > 0:
    #         v.append(self.position.short.open_avg_price)
    #     return max(v) if v else 0
    #
    # @property
    # def low(self):
    #     v = [k.low for k in self.klines]
    #     if self.position.long.position > 0:
    #         v.append(self.position.long.open_avg_price)
    #     elif self.position.short.position > 0:
    #         v.append(self.position.short.open_avg_price)
    #     return min(v) if v else 0

    @property
    def price(self):
        p = 0
        if self.position.long.position > 0:
            p = self.position.long.open_avg_price
        elif self.position.short.position > 0:
            p = self.position.short.open_avg_price
        return p

    @property
    def status(self):
        return {
            "opening": self.opening,
            "signal": self.signal,
            "count": self.count,
            "price": self.price,
            "high": self.high,
            "low": self.low,
            "ma": self._ma[-1],
            "macd": self._macd[-1][-1],
        }

    def ma_call(
            self,
    ):
        match self.ma_type:
            case "ema":
                v = talib.EMA(
                    self._prices,
                    timeperiod=self.ma_period
                )[-1]
            case "llt":
                if len(self._ma) <= 2:
                    v = self._prices[-1]
                else:
                    alpha = 2 / (self.ma_period + 1)
                    v = (alpha - alpha ** 2 / 4) * self._prices[-1] + (alpha ** 2 / 2) * self._prices[-2] - (alpha - 3 * (alpha ** 2) / 4) * self._prices[-3] + 2 * (1 - alpha) * self._ma[-1] - (1 - alpha) ** 2 * self._ma[-2]
            case _:
                v = talib.MA(
                    self._prices,
                    timeperiod=self.ma_period
                )[-1]
        return v

    def macd_call(
            self,
    ):
        diff, dea, macd = talib.MACD(
            self._prices,
            fastperiod=self.diff,
            slowperiod=self.dea,
            signalperiod=self.m,
        )
        return diff[-1], dea[-1], macd[-1] * 2

    def push(
            self,
            kline: KLineData,
    ):
        # price = (kline.high + kline.low + kline.open + kline.close * 3) / 6
        price = kline.close
        self._prices = np.append(self._prices, price)
        v = self.ma_call()
        self._ma.append(v)
        v = self.macd_call()
        self._macd.append(v)
        first = False
        close = price
        if close > self._ma[-1] and self._macd[-1][-1] > 0:
            first = (
                True if self.signal <= 0 else False
            )
            self.signal = 1
        if close < self._ma[-1] and self._macd[-1][-1] < 0:
            first = (
                True if self.signal >= 0 else False
            )
            self.signal = -1
        if first:
            self.count = 1
            self.opening = True
            self.klines = [kline, ]
        else:
            self.count += 1
            self.klines.append(kline)

    def opened(self):
        if self.position.position > 0:
            return ""
        elif (
                self.opening and
                self.signal > 0 and
                self.count <= 3
        ):
            return "buy"
        elif (
                self.opening and
                self.signal < 0 and
                self.count <= 3
        ):
            return "sell"
        return ""

    def closed(self):
        if self.position.position == 0:
            return ""
        elif self.position.long.position > 0:
            if self.signal < 0:
                return "sell"
            if (
                (self.tick.last_price - self.position.long.open_avg_price) / self.position.long.open_avg_price <= self.loss
            ):
                return "sell"
            elif (
                (self.high - self.position.long.open_avg_price) / self.position.long.open_avg_price >= self.even
                and self.tick.last_price <= self.position.long.open_avg_price + self.pay_up
            ):
                return "sell"
            elif (
                self.tick.last_price <= self.high
                and (self.high - self.position.long.open_avg_price) / self.position.long.open_avg_price >= self.profit
                and (self.high - self.tick.last_price) / (self.high - self.position.long.open_avg_price) >= self.back
            ):
                return "sell"
        elif self.position.short.position > 0:
            if self.signal > 0:
                return "buy"
            if (
                (self.position.short.open_avg_price - self.tick.last_price) / self.position.short.open_avg_price <= self.loss
            ):
                return "buy"
            elif (
                (self.position.short.open_avg_price - self.low) / self.position.short.open_avg_price >= self.even
                and self.tick.last_price >= self.position.short.open_avg_price - self.pay_up
            ):
                return "buy"
            elif (
                self.tick.last_price >= self.low
                and (self.position.short.open_avg_price - self.low) / self.position.short.open_avg_price >= self.profit
                and (self.low - self.tick.last_price) / (self.low - self.position.short.open_avg_price) >= self.back
            ):
                return "buy"
        return ""


class SuperMoney(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.params_map = Params()
        self.trading = False
        self.state: State | None = None
        self.kline_generator: KLineGenerator | None = None
        self.timestamp = 0

    def on_start(self):
        self.trading = False
        self.state = State(
            indicator=self.params_map.indicator,
            lep=self.params_map.lep,
            pay_up=self.params_map.pay_up,
        )
        self.kline_generator = KLineGenerator(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.period,
            callback=self.callback,
            # real_time_callback=self.real_time_callback,
            real_time_callback=None,
        )
        self.kline_generator.push_history_data()
        self.state.position = self.get_position(
            instrument_id=self.params_map.instrument_id,
        )
        if self.state.position.position > 0:
            self.state.opening = False
            self.state.high = max([k.high for k in self.state.klines])
            self.state.low = min([k.low for k in self.state.klines])
            self.output(self.state.high, self.state.low)
        self.trading = True
        super().on_start()

    def on_stop(self):
        super().on_stop()

    def on_tick(
            self,
            tick: TickData,
    ):
        super().on_tick(tick)
        self.state.tick = tick
        self.kline_generator.tick_to_kline(tick)
        timestamp = datetime.datetime.now().timestamp()
        delta = timestamp - self.timestamp
        if delta < 1:
            return
        if int(timestamp) % 5 == 0:
            if self.state.order and self.state.order.status == "未成交":
                self.cancel_order(self.state.order.order_id)
            self.state.position = self.get_position(
                instrument_id=self.params_map.instrument_id,
            )
            if self.state.position.position > 0:
                p = self.state.position.long.open_avg_price or self.state.position.short.open_avg_price
                self.state.high = max(self.state.high, p, tick.last_price)
                self.state.low = min(self.state.low, p, tick.last_price)
            else:
                self.state.high = 0.0
                self.state.low = 99999999.0
            if (d := self.state.opened()) != "":
                self.open(direction=d)
            elif (d := self.state.closed()) != "":
                self.close(direction=d)
        self.timestamp = datetime.datetime.now().timestamp()

    def on_order(
            self,
            order: OrderData,
    ):
        super().on_order(
            order=order
        )
        self.state.order = order
        self.output(f"挂单：{order.exchange}:{order.instrument_id} | 价格[{order.price}]")

    def on_order_cancel(
            self,
            order: OrderData,
    ):
        super().on_order_cancel(order)
        self.state.order = None
        self.output(f"撤单：{order.exchange}:{order.instrument_id} | 价格[{order.price}]")

    def on_trade(
            self,
            trade: TradeData,
            log: bool = False,
    ):
        super().on_trade(trade, log)
        self.state.order = None
        if "开" in trade.offset:
            o = "开"
            self.state.opening = False
        else:
            o = "平"
            self.state.high = 0.0
            self.state.low = 99999999.0
        self.output(f"成交：{trade.exchange}:{trade.instrument_id} | 价格[{trade.price}] | {trade.direction+o}")

    def open(
            self,
            direction: str,
    ):
        self.state.position = self.get_position(
            instrument_id=self.params_map.instrument_id,
        )
        if direction == "buy":
            self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=self.params_map.order_volume,
                price=self.state.tick.last_price + self.state.pay_up,
                order_direction="buy"
            )
        elif direction == "sell":
            self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=self.params_map.order_volume,
                price=self.state.tick.last_price - self.state.pay_up,
                order_direction="sell"
            )

    def close(
            self,
            direction: str,
    ):
        if direction == "sell":
            self.auto_close_position(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=self.params_map.order_volume,
                price=self.state.tick.last_price - self.state.pay_up,
                order_direction="sell"
            )
        elif direction == "buy":
            self.auto_close_position(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=self.params_map.order_volume,
                price=self.state.tick.last_price + self.state.pay_up,
                order_direction="buy"
            )

    def callback(
            self,
            kline: KLineData
    ):
        self.state.push(
            kline=kline,
        )
        self.output(self.state.status)

    def real_time_callback(
            self,
            kline: KLineData
    ):
        self.widget.recv_kline({
            "kline": kline,
        })

