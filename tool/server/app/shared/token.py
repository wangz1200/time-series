from .base import *
import datetime
import jwt
import fastapi as fa


__all__ = (
    "token",
)


class Token(object):
    SECRET_KEY = "wangz1200"
    ALGORITHM = "HS256"
    TIMEDELTA = datetime.timedelta(hours=24)

    def __init__(self):
        super(Token, self).__init__()

    @classmethod
    def build(cls,
              id_: str | int,
              user: str,
              timedelta=None, ):
        timedelta = timedelta or cls.TIMEDELTA
        playload = {
            "id": id_,
            "user": user,
        }
        token_str = jwt.encode(playload,
                               key=cls.SECRET_KEY,
                               algorithm=cls.ALGORITHM, )
        return token_str

    @classmethod
    def parse(
            cls,
            ts: str | None = None,
    ):
        ret = (
            None if ts is None
            else jwt.decode(ts, cls.SECRET_KEY, algorithms=[cls.ALGORITHM])
        )
        return ret

    def request(
            self, req: fa.Request,
    ):
        try:
            ts = req.headers.get("authorization", None)
            t = self.parse(ts.split(" ")[1])
            req.session.update(t)
        except Exception:
            raise fa.HTTPException(
                status_code=fa.status.HTTP_401_UNAUTHORIZED,
                detail="Parse Token Error",
            )


token = Token()

