from .base import *
import fastapi as fa
from starlette.middleware.cors import CORSMiddleware


__all__ = (
    "router",
)

router = fa.FastAPI()
router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

