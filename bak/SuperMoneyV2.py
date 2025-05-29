import datetime
import numpy as np
import talib

from pythongo.base import BaseParams, Field
from pythongo.ui import BaseStrategy
from pythongo.classdef import KLineData, OrderData, TickData, TradeData, Position
from pythongo.utils import KLineGenerator


class Params(BaseParams):
    exchange: str = Field(default="DCE", title="交易所代码")
    instrument_id: str = Field(default="v2501", title="合约代码")
    period: str = Field(default="M5", title="K线周期")
    ma_length: int = Field(default=60, title="均线")
    macd_short_length: int = Field(default=21, title="MACD快线")
    macd_long_length: int = Field(default=55, title="MACD慢线")
    macd_m_length: int = Field(default=13, title="MACD")
    atr_length: int = Field(default=60, title="ATR")
    atr_loss: int = Field(default=2, title="ATR止损")
    atr_profit: int = Field(default=4, title="ATR止盈")
    back_ratio: float = Field(default=0.2, title="回撤比例")
    order_volume: int = Field(default=1, title="下单手数", ge=1)
    pay_up: float = Field(default=0, title="超价", ge=0)
    signal: bool = Field(default=True, title="多空止损")


class SuperMoneyV2(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.params_map = Params()
        self.trading = False
        self.kline_generator: KLineGenerator | None = None
        self.order: OrderData | None = None
        self.position: Position | None = None
        self.timestamp = 0
        self.klines = []
        self.kline_size = 500
        self.signal = 0
        self.count = 0
        self.ma = 0.0
        self.macd = 0.0
        self.atr = 0.0
        self.high = 0.00
        self.low = 10000000.00
        self.close = 0

    @property
    def status(self):
        if self.position.long.position > 0:
            price = self.position.long.open_avg_price
        elif self.position.short.position > 0:
            price = self.position.short.open_avg_price
        else:
            price = 0.00
        info = {
            "signal": self.signal,
            "count": self.count,
            "price": "%.2f" % price,
            "close": "%.2f" % self.close,
            "ma": "%.2f" % self.ma,
            "macd": "%.2f" % self.macd,
            "atr": "%.2f" % self.atr,
            "high": "%.2f" % self.high,
            "low": "%.2f" % self.low,
        }
        return info

    def push(
            self,
            kline: KLineData,
    ):
        self.klines.append(kline)
        if len(self.klines) > self.kline_size:
            begin = len(self.klines) - self.kline_size
            self.klines = self.klines[begin :]
        self.close = kline.close
        price = []
        atr = []
        for k in self.klines:
            price.append(k.close)
            atr.append(k.high - k.low)
        price = np.array(price)
        atr = np.array(atr)
        self.ma = talib.EMA(
            price,
            timeperiod=self.params_map.ma_length,
        )[-1]
        _, _, macd = talib.MACD(
            price,
            fastperiod=self.params_map.macd_short_length,
            slowperiod=self.params_map.macd_long_length,
            signalperiod=self.params_map.macd_m_length,
        )
        self.macd = macd[-1]
        self.atr = talib.EMA(
            atr,
            timeperiod=self.params_map.atr_length,
        )[-1] + 0.017
        if self.signal <= 0 and kline.open > self.ma and kline.close > self.ma and self.macd > 0:
            self.signal = 1
            self.count = 0
        elif self.signal >= 0 and kline.open < self.ma and kline.close < self.ma and self.macd < 0:
            self.signal = -1
            self.count = 0
        self.count += 1
        if self.position.long.position > 0:
            self.high = max(kline.high, self.high, self.position.long.open_avg_price)
        elif self.position.short.position > 0:
            self.low = min(kline.low, self.low, self.position.short.open_avg_price)
        else:
            self.high = 0.00
            self.low = 10000000.00

    def on_start(self):
        self.trading = False
        self.kline_generator = KLineGenerator(
            exchange=self.params_map.exchange,
            instrument_id=self.params_map.instrument_id,
            style=self.params_map.period,
            callback=self.callback,
            # real_time_callback=self.real_time_callback,
            real_time_callback=None,
        )
        self.position = self.get_position(
            instrument_id=self.params_map.instrument_id,
        )
        self.kline_generator.push_history_data()
        self.trading = True
        super().on_start()

    def on_stop(self):
        super().on_stop()

    def on_tick(
            self,
            tick: TickData,
    ):
        super().on_tick(tick)
        self.kline_generator.tick_to_kline(tick)
        timestamp = datetime.datetime.now().timestamp()
        if int(timestamp) % 5 != 0:
            return
        if self.order and self.order.status == "未成交":
            self.cancel_order(self.order.order_id)
            return
        self.position = self.get_position(
            instrument_id=self.params_map.instrument_id,
        )
        if self.position.long.position > 0:
            p = self.position.long.open_avg_price
            self.high = max(self.high, p)
            if (
                    (self.signal < 0 and self.params_map.signal)
                    or p - tick.last_price >= self.atr * self.params_map.atr_loss
                    or (
                        self.high - p >= self.atr * self.params_map.atr_profit
                        and (self.high - tick.last_price) / (self.high - p) >= self.params_map.back_ratio
                    )
            ):
                self.auto_close_position(
                    exchange=self.params_map.exchange,
                    instrument_id=self.params_map.instrument_id,
                    volume=self.params_map.order_volume,
                    price=tick.last_price - self.params_map.pay_up,
                    order_direction="sell"
                )
        elif self.position.short.position > 0:
            p = self.position.short.open_avg_price
            self.low = min(self.low, p)
            if (
                    (self.signal > 0 and self.params_map.signal)
                    or tick.last_price - p >= self.atr * self.params_map.atr_loss
                    or (
                        p - self.low >= self.atr * self.params_map.atr_profit
                        and (tick.last_price - self.low) / (p - self.low) >= self.params_map.back_ratio
                    )
            ):
                self.auto_close_position(
                    exchange=self.params_map.exchange,
                    instrument_id=self.params_map.instrument_id,
                    volume=self.params_map.order_volume,
                    price=tick.last_price + self.params_map.pay_up,
                    order_direction="buy"
                )
        elif self.signal > 0 and self.count <= 2:
            self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=self.params_map.order_volume,
                price=tick.last_price + self.params_map.pay_up,
                order_direction="buy"
            )
        elif self.signal < 0 and self.count <= 2:
            self.send_order(
                exchange=self.params_map.exchange,
                instrument_id=self.params_map.instrument_id,
                volume=self.params_map.order_volume,
                price=tick.last_price - self.params_map.pay_up,
                order_direction="sell"
            )

    def on_order(
            self,
            order: OrderData,
    ):
        super().on_order(
            order=order
        )
        self.order = order
        self.output({
            "type": "挂单",
            "product": f"{order.exchange}_{order.instrument_id}",
            "price": order.price,
        })

    def on_order_cancel(
            self,
            order: OrderData,
    ):
        super().on_order_cancel(order)
        self.order = None
        self.output({
            "type": "撤单",
            "product": f"{order.exchange}_{order.instrument_id}",
            "price": order.price,
        })

    def on_trade(
            self,
            trade: TradeData,
            log: bool = False,
    ):
        super().on_trade(trade, log)
        self.order = None
        if "开" in trade.offset:
            o = "开"
        else:
            o = "平"
        self.output({
            "type": trade.direction + o,
            "product": f"{trade.exchange}_{trade.instrument_id}",
            "price": trade.price,
        })

    def callback(
            self,
            kline: KLineData
    ):
        self.push(
            kline=kline,
        )
        self.output(self.status)

    def real_time_callback(
            self,
            kline: KLineData
    ):
        self.widget.recv_kline({
            "kline": kline,
        })

