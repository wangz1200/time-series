from app import shared
from app import modal
from app import service
from app import api
import uvicorn


if __name__ == "__main__":
    shared.config.load(
        config_path="./config/config.yaml"
    )
    service.init()
    uvicorn.run(
        shared.router,
        host=str(shared.config.args["server"]["host"]),
        port=int(shared.config.args["server"]["port"]),
    )
