from __future__ import annotations

import contextlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from . import __version__, engine, fetch
from . import config as cfg_mod

PYPI_URL = "https://pypi.org/pypi/whatisit/json"
DAY = 86_400


def _state_path() -> Path:
    return cfg_mod.data_dir() / "update.json"


def _load() -> dict:
    try:
        state = json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _save(**changes) -> bool:
    path = _state_path()
    state = _load()
    state.update(changes)
    tmp = path.with_name(f"{path.name}.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(state))
        tmp.replace(path)
        return True
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return False


def _version(v) -> tuple:
    if not isinstance(v, str) or not re.fullmatch(r"\d{1,6}(\.\d{1,6}){0,5}", v):
        return ()
    parts = [int(x) for x in v.split(".")]
    while parts and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _elapsed(since, now: float) -> float:
    try:
        return now - float(since)
    except (TypeError, ValueError):
        return float("inf")


def installed_model() -> tuple[Path, str] | None:
    if cfg_mod.env("MODEL"):
        return None
    models = cfg_mod.data_dir() / "models"
    path = (models / cfg_mod.MODEL_NAME).resolve()
    if path.parent != models.resolve():
        return None
    for size, spec in fetch.MODELS.items():
        if path.name == spec["file"] and path.is_file():
            return path, size
    return None


def _stat_key(path: Path) -> str:
    st = path.stat()
    return f"{path}:{st.st_size}:{st.st_mtime_ns}"


def model_sha(path: Path, compute: bool = True) -> str | None:
    try:
        key = _stat_key(path)
        cached = _load().get("model")
        if (isinstance(cached, dict) and cached.get("key") == key
                and isinstance(cached.get("sha256"), str)):
            return cached["sha256"]
        if not compute:
            return None
        sha = fetch.sha256_file(path)
    except OSError:
        return None
    _save(model={"key": key, "sha256": sha})
    return sha


def remember_sha(path: Path, sha: str) -> None:
    with contextlib.suppress(OSError):
        _save(model={"key": _stat_key(path), "sha256": sha})


def is_pinned(sha: str | None) -> bool:
    return sha in {spec["sha256"] for spec in fetch.MODELS.values()}


def refresh() -> None:
    try:
        req = urllib.request.Request(PYPI_URL, headers={"User-Agent": f"whatisit/{__version__}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            latest = json.loads(r.read().decode())["info"]["version"]
        if _version(latest):
            _save(latest=latest)
    except Exception:
        pass
    if cfg_mod.remote_config(cfg_mod.load_config()) is None and (found := installed_model()):
        model_sha(found[0])


def notices(cfg: dict, now: float | None = None) -> list[str]:
    if not cfg.get("update_check", True):
        return []
    now = time.time() if now is None else now
    state = _load()
    if not 0 <= _elapsed(state.get("checked"), now) < DAY and _save(checked=now):
        engine.spawn_detached("whatisit.update")

    msgs = []
    latest = state.get("latest")
    if _version(latest) > _version(__version__):
        msgs.append(f"whatisit {latest} is out (you have {__version__}). "
                    "pip install -U whatisit")
    found = None if cfg_mod.remote_config(cfg) else installed_model()
    if found:
        sha = model_sha(found[0], compute=False)
        if sha and not is_pinned(sha):
            flag = "" if found[1] == "1.5b" else f" --size {found[1]}"
            msgs.append(f"a newer model is out. run: whatisit setup{flag}")

    if not msgs or 0 <= _elapsed(state.get("notified"), now) < DAY:
        return []
    _save(notified=now)
    return msgs


if __name__ == "__main__":
    refresh()
