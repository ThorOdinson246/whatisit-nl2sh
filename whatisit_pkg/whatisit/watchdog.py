"""whatisit.watchdog - user-space idle timeout for the resident llama-server.

Spawned detached by engine.generate(); llama-server's native
--sleep-idle-seconds flag is never used (in builds b7492-b9060 it triggers
CVE-2026-43631). Unlike checking idleness only on the next query, this frees
the model's RAM at N seconds even when no further query ever arrives.

Contract with engine, all files inside the 0700 state dir:
  server.pid     written by start_server(); re-read every tick, so a
                 restart is followed, not treated as a stop
  server.watch   "<timeout>\\n", rewritten by every server-mode query while
                 the feature is on and removed when it is turned off
  server.last_use  touched before and after each server-mode query

A non-blocking flock makes this a singleton: extra spawns (one per query)
exit immediately, so repeated invocations cost nothing. Everything here is
best-effort and silent; engine.idle_stop() is the lazy fallback whenever this
process is not around.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

_TICK_MAX = 5.0
_RETRY = 1.0
_GONE_TICKS = 3


def _try_lock(path: Path):
    """fd holding an exclusive non-blocking lock, or None if one is held."""
    # O_NOFOLLOW, matching engine._write_private: WHATISIT_DATA_DIR is
    # user-controlled, so a planted symlink must fail loudly, not be locked.
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        fd = None
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_RDWR | nofollow, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if fd is not None:
                os.close(fd)     # flock lost the race: don't leak the fd
            return None
    try:
        import msvcrt
    except ImportError:              # neither primitive: duplicates would be
        return None                  # possible, so standing down is safer
    fd = None
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR | nofollow, 0o600)
        os.write(fd, b"\0")      # msvcrt locks a byte RANGE, so the file
        os.lseek(fd, 0, os.SEEK_SET)  # needs >= 1 byte, and every holder
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # must contend on offset 0
        return fd
    except OSError:
        if fd is not None:
            os.close(fd)
        return None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_ts(path: Path) -> float | None:
    try:
        v = float(path.read_text().strip())
    except (OSError, ValueError):
        return None
    # "nan" would disarm every deadline comparison; "inf" would fire at once.
    return v if math.isfinite(v) else None


def run(state_dir=None, now_fn=time.time, sleep_fn=time.sleep, stop_fn=None,
        max_ticks=None) -> None:
    """Watch the state dir until the idle deadline fires; injectable for tests."""
    try:
        if state_dir is None:
            from . import engine
            state_dir = engine._state_dir()
        sd = Path(state_dir)
        fd = _try_lock(sd / "watchdog.lock")
        if fd is None:
            return                   # a live singleton already watches this
        try:
            _loop(sd, now_fn, sleep_fn, stop_fn, max_ticks)
        finally:
            os.close(fd)
    except Exception:
        pass                         # never surface anything to the user


def _loop(sd: Path, now_fn, sleep_fn, stop_fn, max_ticks) -> None:
    pid_f, watch_f, use_f = (sd / "server.pid", sd / "server.watch",
                             sd / "server.last_use")
    started = now_fn()
    ticks = gone = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        pid, timeout = _read_int(pid_f), _read_int(watch_f)
        if pid is None or timeout is None or timeout <= 0:
            # The sub-second gap where stop_server has unlinked the pid and
            # start_server has not written the new one. We hold the lock, so a
            # replacement spawn already gave up: wait rather than leave nothing
            # watching. A full restart outlives this and is re-armed by the
            # post-ready _spawn_watchdog().
            gone += 1
            if gone >= _GONE_TICKS:
                return               # really stopped, or feature turned off
            sleep_fn(_RETRY)
            continue
        gone = 0
        seen = _read_ts(use_f)
        # Never queried: age from our own start, so a server that is launched
        # and then abandoned still gets unloaded.
        idle = max(0.0, now_fn() - (started if seen is None else seen))
        if idle < timeout:
            sleep_fn(min(timeout - idle, _TICK_MAX))
            continue
        sleep_fn(0.2)                # grace: a query may be touching now
        if _read_ts(use_f) != seen or _read_int(pid_f) != pid:
            continue                 # busy or superseded: keep watching
        if stop_fn is not None:
            stop_fn()
        else:
            from . import engine
            engine.stop_server()
        return


def main() -> None:
    run()


if __name__ == "__main__":
    main()
