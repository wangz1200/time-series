from .base import *
from .dao import DAO, MySQL

__all__ = (
    "State",
    "state",
)


class State(object):

    def __init__(
            self,
            dao: DAO | None = None,
    ):
        super().__init__()
        self.dao = dao


state = State()
