from .base import *
import hashlib


__all__ = (
    "util",
)


class Util(object):

    def __init__(self):
        super().__init__()

    def list_(
            self,
            data: Any,
            separator: str = ","
    ):
        if not data:
            return []
        if isinstance(data, str):
            data = data.split(separator)
        if not isinstance(data, list):
            data = [data, ]
        return data

    def md5(
            self,
            content: bytes
    ):
        return hashlib.md5(content).hexdigest()


util = Util()

