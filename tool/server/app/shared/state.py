from .base import *
from .dao import DAO


__all__ = (
    "State",
    "state",
)


class State(object):

    def __init__(
            self,
            dao: DAO | None = None,
            signal_url: str | None = None,
    ):
        super().__init__()
        self.dao = dao
        self.signal_url = signal_url


state = State()
