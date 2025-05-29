from .base import *


router = fa.APIRouter(
    prefix="/kline",
)


@router.get("")
async def query_kline(
        product: str,
        period: str,
        begin_datetime: str | None = None,
        end_datetime: str | None = None,
        limit: int = 500,
):
    res = define.Result()
    try:
        list_ = service.kline.query(
            product=product,
            period=period,
            begin_datetime=begin_datetime,
            end_datetime=end_datetime,
            limit=limit,
        )
        res.data = {
            "list": list_
        }
    except Exception as ex:
        res.code = -1
        res.msg = str(ex)
    return res


@router.post("")
async def add_kline(
        r: fa.Request,
        req: define.kline.AddReq,
):
    res = define.Result()
    try:
        data = req.data_()
        service.kline.add(
            data=data,
            update=req.update,
        )
    except Exception as ex:
        res.code = -1
        res.msg = str(ex)
    return res


@router.get("/signal")
async def query_kline_signal(
        product: str,
        period: str,
        timestamp: int | None = None,
        limit: int = 1024,
):
    res = define.Result()
    try:
        pred = service.kline.signal.query(
            product=product,
            period=period,
            timestamp=timestamp,
            limit=limit,
        )
        res.data = pred
    except Exception as ex:
        res.code = -1
        res.msg = str(ex)
    return res


@router.post("/signal/sync")
async def sync_kline_signal(
        r: fa.Request,
        req: define.signal.SyncReq,
):
    res = define.Result()
    try:
        pred = service.kline.signal.query(
            product=req.product,
            period=req.period,
            timestamp=req.timestamp,
        )
        if float(pred["score"]) < req.score:
            return
        id_ = pred["id"]
        service.kline.signal.add(
            data={
                "id": id_,
                "signal": pred["signal"],
                "score": pred["score"],
            },
            update=True,
        )
    except Exception as ex:
        res.code = -1
        res.msg = str(ex)
    return res


@router.put("/signal")
async def update_kline_signal(
        r: fa.Request,
        req: define.signal.UpdateReq
):
    res = define.Result()
    try:
        service.kline.signal.add(
            data=req.data_(),
            update=True
        )
    except Exception as ex:
        res.code = -1
        res.msg = str(ex)
    return res


shared.router.include_router(
    router=router,
)
