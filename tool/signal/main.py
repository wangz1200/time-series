from pathlib import Path
from app import shared
from app import service
from app import api
import uvicorn


def _init_model(
        model_dir: str | Path,
):
    model_dir = Path(model_dir)
    model = shared.model.Model.from_config(
        model_dir=model_dir,
    )
    model = model.eval()
    return model


if __name__ == "__main__":
    shared.config.load(
        config_path="./config/config.yaml"
    )
    model_dir = Path(shared.config.args["root"]["path"], "public", "model")
    service.state.model = _init_model(
        model_dir=model_dir,
    )
    uvicorn.run(
        shared.router,
        host=str(shared.config.args["server"]["host"]),
        port=int(shared.config.args["server"]["port"]),
    )
