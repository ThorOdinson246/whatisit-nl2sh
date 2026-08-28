"""Config + model resolution, following the XDG spec.

Locations (overridable by env, which is what makes the tool testable):
  config  $XDG_CONFIG_HOME/whatisit/config.json   (~/.config/whatisit)
  models  $XDG_DATA_HOME/whatisit/models          (~/.local/share/whatisit)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# The published filename on Hugging Face, not our identifier.
MODEL_NAME = "nl2sh-1.5b-Q4_K_M.gguf"

# OpenAI-compatible remote endpoint settings. Env wins over config.json so a
# secret API key can live only in the environment, never on disk. Each of the
# three keys is independent: setting only a base URL + model selects a keyless
# backend (a local LAN llama-server), which is the common case for whatisit.
# Suffixes for the WHATISIT_* / NL2SH_* env vars, per the env() helper.
ENDPOINT_ENV = {
    "openai_base_url": "OPENAI_BASE_URL",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model": "OPENAI_MODEL",
    "openai_timeout": "OPENAI_TIMEOUT",
    "openai_max_tokens": "OPENAI_MAX_TOKENS",
}

APP_NAME = "whatisit"
LEGACY_NAME = "nl2sh"


def remote_config(cfg: dict, timeout_default: float = 120.0) -> dict | None:
    """Resolve the OpenAI-compatible endpoint, env over config.json.

    Returns None when no base URL is configured (the fully-local default).
    When a base URL is set it returns a dict with keys:
      base_url  the configured endpoint (raw; engine normalizes it)
      api_key   bearer token, or '' if unauthenticated (LAN llama-server)
      model     the model name to send in the body, or None if not chosen
      timeout    request timeout in seconds
      max_tokens generation budget in tokens (512 default: remote endpoints
                 often run reasoning models that spend tokens 'thinking' before
                 emitting the command, so the local 64-token budget is too small)
    An empty openai_base_url (or empty env value) means local mode. An empty
    env value wins over a stored config value, so `WHATISIT_OPENAI_BASE_URL=`
    turns remote mode off without editing config.json.
    """
    raw = env(ENDPOINT_ENV["openai_base_url"])
    base = cfg.get("openai_base_url") if raw is None else raw
    if base is None:
        return None
    base = str(base).strip()
    if not base:
        return None
    raw_key = env(ENDPOINT_ENV["openai_api_key"])
    key = cfg.get("openai_api_key") if raw_key is None else raw_key
    key = "" if key is None else str(key)
    raw_model = env(ENDPOINT_ENV["openai_model"])
    model = cfg.get("openai_model") if raw_model is None else raw_model
    if model is not None:
        model = str(model).strip() or None
    max_tokens = _openai_int(ENDPOINT_ENV["openai_max_tokens"], cfg, "openai_max_tokens",
                             512)
    timeout = _openai_int(ENDPOINT_ENV["openai_timeout"], cfg, "openai_timeout",
                          timeout_default)
    return {"base_url": str(base).strip(), "api_key": str(key),
            "model": str(model) if model else None, "timeout": timeout,
            "max_tokens": max_tokens}


def _openai_int(env_suffix: str, cfg: dict, key: str, default):
    """Parse an int config value (env over cfg), falling back on garbage."""
    raw = env(env_suffix)
    if raw is None:
        raw = cfg.get(key)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


SYSTEM_PROMPT = (
    "You are a shell command generator. Output exactly one line: a single "
    "POSIX/bash command that accomplishes the user's request. No prose, no "
    "markdown fences, no explanation."
)

DEFAULTS = {
    "threads": 0,          # 0 => auto (see resolve_threads)
    "max_tokens": 64,
    "temperature": 0.0,    # greedy: same question -> same command
    "confirm_execute": True,
    # OFF by default: -e output goes straight to a shell, so Enter must not run
    # commands unless the user opted in. Set confirm_default=true to get [Y/n]
    # (empty answer means yes) on the confirmation prompt.
    "confirm_default": False,
    # OFF by default. It is cheap (prefix-cached) but MEASURED HARMFUL on the
    # only benchmark available: 54.2% -> 45.1% pass, p=0.0004. See hostctx.py.
    "host_context": False,
    # OFF by default (opt-in). Keeps the full host-facts prompt block out of the
    # way of the shipped model's measured regression, while still fixing the
    # wrong-distro failure: when on, install prompts are GBNF-constrained to
    # the host package manager on the local backends (llama-server /
    # llama-cli), and every candidate command is regex-rewritten from foreign
    # syntax as a backstop everywhere -- remote endpoints included, though they
    # get no grammar field on the wire. Costs one short prefix-cached line.
    # Enable if whatisit suggests `apt install` on Arch, `pacman -S` on Debian,
    # etc.
    "distro_guidance": False,
    # GBNF grammar constraint for install prompts. When on (default) and the
    # host's package manager is known, the model is constrained to that manager's
    # install syntax. --no-grammar disables only the constraint; distro-aware
    # postprocessing remains active as a backstop.
    "use_grammar": True,
    # Serve over a TCP port instead of a UNIX socket. Set by `setup` when it
    # installs a runtime whose llama-server cannot bind a socket path. A fixed
    # value forces that port; None keeps the current auto/free-port behaviour.
    "force_tcp": False,
    "server_port": None,       # fixed TCP port for the resident server
    "ctx_size": 2048,          # context size passed to llama-server/llama-cli
}


def env(suffix: str, default: str | None = None) -> str | None:
    """WHATISIT_<suffix>, falling back to NL2SH_<suffix>.

    The fallback is permanent. Dropping it would fail silently: callers that
    set only the old name would get the defaults instead of an error.
    """
    return os.environ.get(f"WHATISIT_{suffix}",
                          os.environ.get(f"NL2SH_{suffix}", default))


def config_dir() -> Path:
    return Path(env("CONFIG_DIR") or
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME)


def data_dir() -> Path:
    return Path(env("DATA_DIR") or
                Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / APP_NAME)


def legacy_dir(kind: str) -> Path:
    base = {"config": (os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")),
            "data": (os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))}[kind]
    return Path(base) / LEGACY_NAME


def migrate_legacy_dirs(echo=print) -> list[str]:
    """Move ~/.config/nl2sh and ~/.local/share/nl2sh to their new names, once.

    Fires only when the new directory is absent and the old one is present, so
    it is idempotent and never overwrites. Skipped when the location came from
    the environment: that path was chosen deliberately.
    """
    msgs = []
    for kind, explicit, new in (("config", env("CONFIG_DIR"), config_dir()),
                                ("data", env("DATA_DIR"), data_dir())):
        old = legacy_dir(kind)
        if explicit or new.exists() or not old.is_dir():
            continue
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
        except OSError as e:
            msgs.append(f"whatisit: could not move {old} to {new}: {e}")
            continue
        msgs.append(f"whatisit: moved {old} -> {new}")
        # setup registers the model and binaries as symlinks. Absolute ones
        # survive the move; a relative one pointing outside now dangles.
        for p in sorted(new.rglob("*")):
            if p.is_symlink() and not p.exists():
                msgs.append(f"whatisit: warning: {p} points at "
                            f"{os.readlink(str(p))}, which no longer resolves")
    for m in msgs:
        echo(m)
    return msgs


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    p = config_path()
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass  # a corrupt config must not brick the tool
    return cfg


def save_config(cfg: dict) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    p = config_path()
    # Create with 0600 from the start rather than write-then-chmod: the latter
    # leaves a window where the file sits at the umask default -- group-readable
    # on the shared-NFS-home clusters this targets -- readable by a co-tenant.
    if os.name == 'nt':
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    else:
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    # The 0o600 above applies only at CREATION; a pre-existing file keeps
    # whatever mode it had. fchmod on the open fd closes that gap, race-free
    # because it acts on the fd rather than the path.
    try:
        os.fchmod(fd, 0o600)
    except (OSError, AttributeError):
        pass
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(cfg, indent=2) + "\n")
    return p


def resolve_threads(cfg: dict) -> int:
    """Default to half the cores, capped at 4.

    Decode is memory-bandwidth-bound (measured: ~31 GB/s saturated by both the
    1.5B and the 4B), so throughput stops scaling with cores well before the
    core count -- and on a laptop, spending every core makes the fans audible
    for no speedup. Half-cores capped at 4 is the knee of that curve.
    """
    t = int(cfg.get("threads") or 0)
    if t > 0:
        return t
    return max(1, min(4, (os.cpu_count() or 4) // 2))


def find_model() -> Path | None:
    if v := env("MODEL"):
        p = Path(v)
        return p if p.exists() else None
    for cand in (data_dir() / "models" / MODEL_NAME,
                 Path.cwd() / MODEL_NAME):
        if cand.exists():
            return cand
    return None


def find_llama_cli() -> Path | None:
    if v := env("LLAMA_CLI"):
        p = Path(v)
        return p if p.exists() else None
    if w := shutil.which("llama-cli"):
        return Path(w)
    cand = data_dir() / "bin" / "llama-cli"
    return cand if cand.exists() else None
