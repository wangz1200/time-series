from .base import *
import datetime
import time
from threading import Lock



__all__ = (
    "snow",
)


class Snow(object):

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__()
        self._kwargs = copy.deepcopy(kwargs)
        self.lock = Lock()
        self.year = self._kwargs.get("year", 2013)
        self.month = self._kwargs.get("month", 12)
        self.day = self._kwargs.get("day", 31)
        self.hour = self._kwargs.get("hour", 12)
        self.minute = self._kwargs.get("minute", 59)
        self.second = self._kwargs.get("second", 59)
        self.epoch = int(
            datetime.datetime(
                year=self.year,
                month=self.month,
                day=self.day,
                hour=self.hour,
                minute=self.minute,
                second=self.second,
            ).timestamp() * 1000
        )
        self.current = self.now()
        self.index = 1

    def now(self):
        return int(time.time() * 1000)

    def sid(self):
        with self.lock:
            now = self.now()
            if now == self.current:
                self.index += 1
            else:
                self.current = now
                self.index = 1
            sid = ((self.current - self.epoch) << 12) | self.index
            return sid


snow = Snow()

