from typing import Tuple, List, Dict, Any
from pydantic import BaseModel
from app import shared


class Result(BaseModel):

    code: int = 0
    msg: str = ""
    data: Any = None
