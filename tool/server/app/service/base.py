from typing import Tuple, List, Dict, Any
import copy
import sqlalchemy as sa
from app import shared
from app import modal


class _Signal(object):

    def __init__(
            self,
            url: str | None = None
    ):
        super().__init__()
        self.url = url


class State(object):

    def __init__(
            self,
            dao: shared.dao.DAO | None = None,
            signal_url: str | None = None,
    ):
        super().__init__()
        self.dao = dao
        self.signal = _Signal(url=signal_url)


S = State()
