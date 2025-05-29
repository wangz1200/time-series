from pathlib import Path
import yaml


__all__ = (
    "args",
    "load",
)


args = {}


def load(
        config_path: str | Path,
):
    global args
    config_path = Path(config_path)
    with config_path.open(
            mode="r", encoding="utf8"
    ) as f:
        args = yaml.safe_load(f)
        root_path = Path(config_path).parent.parent.absolute()
        args["root"]["path"] = root_path.__str__()
