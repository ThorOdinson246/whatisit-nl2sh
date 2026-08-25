"""Cross-invocation memory for follow-up requests ("execute them").

The tool is a cold-start process: every invocation is a fresh interpreter, so
"whatisit list three python files here" followed by "whatisit execute them"
reaches the model with no sense of the first request, and the 1.5B rambles.
This module persists the last few turns on disk so the CLI can fold a compact
context block into the next prompt when (and only when) the new request
actually refers back to one of them.

Storage is NDJSON: one JSON turn per line, capped at MAX_TURNS lines. The cap
means a reader only ever needs the file tail and a writer only rewrites three
short lines -- no document parsing, no schema migration, no array bookkeeping.
A corrupt trailing byte costs one turn, never the file.

Two rules shaped by how this feature can go wrong:

NEVER STORE A DANGER COMMAND. Enforced by the caller (cmd_query): a command
the safety checker flagged must not reach this file, or a later "run it again"
would hand the model a stored dangerous command to replay past the checker.

LAST-WRITER-WINS concurrency. Two simultaneous invocations may interleave
rewrites; the loser loses one turn. A lockfile was rejected: it trades a
harmless forgotten turn for a stale-lock failure mode on a tool whose worst
case here is forgetting.

Privacy follows queries.jsonl: local disk only, owner-only (0700 dir, 0600
file, O_NOFOLLOW), wiped by `whatisit session clear`.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import config as cfg_mod

MAX_TURNS = 3
TTL_SECONDS = 15 * 60     # a follow-up minutes later relates; an hour later does not
MAX_NL_CHARS = 200        # request text, truncated before storage
MAX_CMD_CHARS = 300       # generated command, truncated before storage
_TAIL_BYTES = 4096        # orders of magnitude more than three turns occupy


def _path() -> Path:
    return cfg_mod.data_dir() / "session.jsonl"


def _norm_cwd(cwd) -> str:
    # realpath so a session started under /tmp survives macOS resolving it to
    # /private/tmp (and equivalent symlink aliasing elsewhere).
    try:
        return os.path.realpath(str(cwd))
    except OSError:
        return str(cwd)


# Two independent triggers, both conservative. Bare "that"/"this" appear
# nowhere: "find files THAT are bigger than 100MB" is a relative clause, not a
# follow-up, and wrongly injected history spends tokens on a model that
# measurably degrades with prompt noise (see hostctx.py). Plural
# demonstratives and "again" almost never occur outside a genuine follow-up,
# so they may appear anywhere in the sentence.
_STRONG_RE = re.compile(
    r"\b(them|those|these|that one|this one|the same|again)\b", re.IGNORECASE)

# Weak references ("it") count only under an imperative verb lead ("run it"),
# never free-standing: "what time is it" must stay clean.
_REF_RE = re.compile(
    r"\b(it|them|those|these|that one|this one|the same)\b", re.IGNORECASE)
_VERB_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+)?"
    r"(run|execute|delete|remove|kill|open|rename|move|copy|repeat|redo|stop)\b",
    re.IGNORECASE)


def looks_anaphoric(prompt: str) -> bool:
    """True when the request plausibly refers to a previous turn."""
    return bool(_STRONG_RE.search(prompt)
                or (_VERB_RE.match(prompt) and _REF_RE.search(prompt)))


def _read_turns() -> list[dict]:
    """Parse the file tail, skipping anything that is not a valid turn."""
    p = _path()
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    turns = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("nl"), str) \
                and isinstance(d.get("command"), str):
            turns.append(d)
    return turns


def _truncate() -> None:
    try:
        _path().unlink()
    except OSError:
        pass


def _rewrite(turns: list[dict]) -> None:
    """Rewrite all turns, creating owner-only (0600, O_NOFOLLOW).

    Same discipline as queries.jsonl and server.token: create at 0600 rather
    than write-then-chmod (the latter leaves a group-readable window on shared
    NFS homes), O_NOFOLLOW so a planted symlink in a user-controlled
    WHATISIT_DATA_DIR fails hard instead of redirecting the write, and fchmod
    on the open fd to repair a pre-existing wrongly-permissioned file.
    """
    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if os.name != "nt":
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(p), flags, 0o600)
    except OSError:
        return  # O_NOFOLLOW tripped: refuse to write through the symlink
    try:
        os.fchmod(fd, 0o600)
    except (OSError, AttributeError):
        pass  # Windows has neither meaningful chmod nor the POSIX-perm threat
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in turns))


def load_valid() -> list[dict]:
    """Turns the model may legitimately see, resetting the file otherwise.

    A session is dead when its TTL lapsed or the working directory moved;
    either way the stored context would resolve references against the wrong
    scene. Corruption and wrong permissions reset too: a session we cannot
    fully trust is a session we do not use.
    """
    turns = _read_turns()
    fresh = False
    if turns:
        last = turns[-1]
        if time.time() - float(last.get("ts", 0)) > TTL_SECONDS:
            fresh = True
        elif _norm_cwd(last.get("cwd")) != _norm_cwd(os.getcwd()):
            fresh = True
    elif _path().exists():
        fresh = True  # exists but parsed to nothing usable
    if os.name != "nt" and not fresh and _path().exists():
        try:
            if (_path().stat().st_mode & 0o777) != 0o600:
                fresh = True
        except OSError:
            fresh = True
    if fresh:
        _truncate()
        return []
    return turns


def record(nl: str, command: str) -> None:
    """Append one turn, enforcing MAX_TURNS. The caller only ever passes a
    non-DANGER command after successful generation (see cmd_query)."""
    turn = {
        "ts": time.time(),
        "cwd": _norm_cwd(os.getcwd()),
        "nl": nl[:MAX_NL_CHARS],
        "command": command[:MAX_CMD_CHARS],
        "executed": False,
        "exit_code": None,
    }
    turns = (_read_turns() + [turn])[-MAX_TURNS:]
    _rewrite(turns)


def update_executed(command: str, exit_code: int) -> None:
    """Overwrite the newest turn after -e runs it, recording what actually
    executed and how it fared. No-op when nothing was recorded -- e.g. the
    newest generation was DANGER-flagged and skipped."""
    turns = _read_turns()
    if not turns:
        return
    turns[-1]["command"] = command[:MAX_CMD_CHARS]
    turns[-1]["executed"] = True
    turns[-1]["exit_code"] = int(exit_code)
    _rewrite(turns)


def history_block(turns: list[dict]) -> str | None:
    """The injected context, in minimal tokens: one tight line per turn."""
    if not turns:
        return None
    return "\n".join(f'Prior: "{t["nl"]}" -> {t["command"]}' for t in turns)


def clear() -> bool:
    existed = _path().exists()
    _truncate()
    return existed


def show() -> list[dict]:
    """Raw turns for `session show`, without reset side effects."""
    return _read_turns()
