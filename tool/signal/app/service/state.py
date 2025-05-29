from .base import *


__all__ = (
    "State",
    "state",
)


class State(object):

    def __init__(
            self,
            model: shared.model.Model | None = None,
    ):
        super().__init__()
        self.model = model


state = State()
