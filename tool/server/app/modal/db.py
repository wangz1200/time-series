from .base import *
from app import shared

__all__ = (
    "Kline",
    "KlineSignal"
)


class Kline(Table):

    __tablename__ = "kline"

    def __init__(
            self,
            name: str | None = None,
            dao: shared.dao.DAO | None = None,
    ):
        super().__init__(
            name=name,
            dao=dao,
        )

    @classmethod
    def columns(cls):
        return (
            sa.Column("id", sa.VARCHAR(64), nullable=False, comment="K线ID"),
            sa.Column("exchange", sa.VARCHAR(128), nullable=False, default="", comment="交易所"),
            sa.Column("product", sa.VARCHAR(128), nullable=False, default="", comment="品种"),
            sa.Column("period", sa.VARCHAR(128), nullable=False, default="", comment="周期"),
            sa.Column("open", sa.FLOAT, nullable=False, default=0, comment="开盘价"),
            sa.Column("close", sa.FLOAT, nullable=False, default=0, comment="收盘价"),
            sa.Column("high", sa.FLOAT, nullable=False, default=0, comment="最高价"),
            sa.Column("low", sa.FLOAT, nullable=False, default=0, comment="最低价"),
            sa.Column("volume", sa.FLOAT, nullable=False, default=0, comment="成交量"),
            sa.Column("open_interest", sa.FLOAT, nullable=False, default=0, comment="持仓量"),
            sa.Column("datetime", sa.DATETIME, nullable=False, default="", comment="日期时间"),
            sa.Column("timestamp", sa.BIGINT, nullable=False, default=0, comment="毫秒时间戳"),
            sa.PrimaryKeyConstraint("id"),
        )


class KlineSignal(Table):

    __tablename__ = "kline_signal"

    def __init__(
            self,
            name: str | None = None,
            dao: shared.dao.DAO | None = None,
    ):
        super().__init__(
            name=name,
            dao=dao,
        )

    @classmethod
    def columns(cls):
        return (
            sa.Column("id", sa.VARCHAR(64), nullable=False, comment="K线ID"),
            sa.Column("signal", sa.INT, nullable=False, default=0, comment="信号"),
            sa.Column("score", sa.INT, nullable=False, default=0, comment="阈值"),
            sa.PrimaryKeyConstraint("id"),
        )
