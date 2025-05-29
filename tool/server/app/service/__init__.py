from app import shared, modal
from .base import State, S
from .kline import kline
from .signal import signal


def init():
    db_config = shared.config.args["database"]
    S.dao = shared.dao.MySQL(
        host=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"],
        name=db_config["name"],
    )
    signal_config = shared.config.args["signal"]
    S.signal.url = f"http://{signal_config['host']}:{signal_config['port']}"
    modal.db.Kline.create(
        dao=S.dao,
    )
    modal.db.KlineSignal.create(
        dao=S.dao,
    )

