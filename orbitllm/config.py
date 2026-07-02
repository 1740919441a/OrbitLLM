from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    resolve_project_path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_project_path(cfg["figures_dir"]).mkdir(parents=True, exist_ok=True)

