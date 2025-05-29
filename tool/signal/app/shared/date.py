from .base import *
import datetime as DT


class Date(object):

    def __init__(self):
        super().__init__()

    def now(self):
        return DT.datetime.now()

    def ms(self):
        return DT.datetime.now().microsecond // 1000

    def format(
            self,
            date: str,
    ):
        pass


date = Date()
