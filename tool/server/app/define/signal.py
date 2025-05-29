from .base import *


class Item(BaseModel):

    id: str | None = None
    signal: int = 0
    score: float = 1

class QueryReq(BaseModel):

    product: str
    period: str
    begin_datetime: str | None = None
    end_datetime: str | None = None
    timestamp: int | None = None
    limit: int = 96


class UpdateReq(BaseModel):

    data: Item | List[Item]

    def data_(self):
        data = shared.util.list_(self.data)
        ret = []
        for d in data:
            ret.append({
                "id": d.id,
                "signal": d.signal,
                "score": d.score,
            })
        return ret

class SyncReq(BaseModel):

    product: str
    period: str
    timestamp: int | None = None
    score: float = 0.5
