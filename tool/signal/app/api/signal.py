from .base import *
from app import service


router = fa.APIRouter(
    prefix="/signal",
)


@router.post("")
async def query_signal(
        r: fa.Request,
        req: define.signal.QueryReq,
):
    res = define.Result()
    try:
        data = req.data_()
        pred = service.signal.query(
            kline=data,
        )
        res.data = {
            "pred": pred
        }
    except Exception as ex:
        res.code = -1
        res.msg = str(ex)
    return res

shared.router.include_router(
    router=router,
)
