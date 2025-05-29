from .base import *


class State(object):

    def __init__(
            self,
            dao: shared.dao.DAO | None = None,
    ):
        super().__init__()
        self.dao = dao


db_config = shared.config.args["database"]
state = State(
    dao=shared.dao.MySQL(
        host=str(db_config["host"]),
        port=int(db_config["port"]),
        user=str(db_config["user"]),
        passwd=str(db_config["password"]),
        name=str(db_config["name"]),
    )
)
