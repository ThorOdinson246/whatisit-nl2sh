"""Tests for whatisit.watchdog: the user-space idle timeout that keeps
llama-server's CVE-2026-43631-triggering --sleep-idle-seconds flag out of the
picture entirely.

Hermetic throughout: the poll loop runs against a tmp_path state dir with an
injectable clock, a recording stop_fn, and a sleep_fn that returns
immediately. No test spawns a real detached process or waits on wall time.
"""
import os
import sys
import time

import pytest

from whatisit import watchdog as wd

NOW = 1_000_000.0


def make_sd(tmp_path):
    sd = tmp_path / "run"
    sd.mkdir(parents=True)
    return sd


def seed(sd, *, pid=4242, timeout=50, age=None):
    """A consistent-looking state dir; None omits the file."""
    if pid is not None:
        (sd / "server.pid").write_text(f"{pid}\n")
    if timeout is not None:
        (sd / "server.watch").write_text(f"{timeout}\n")
    if age is not None:
        (sd / "server.last_use").write_text(f"{NOW - age:.6f}\n")


class TestReadHelpers:
    def test_read_int(self, tmp_path):
        p = tmp_path / "f"
        p.write_text("300\n")
        assert wd._read_int(p) == 300

    @pytest.mark.parametrize("content", ["junk", "", "tomorrow"])
    def test_read_int_rejects_garbage(self, tmp_path, content):
        p = tmp_path / "f"
        p.write_text(content)
        assert wd._read_int(p) is None

    def test_read_int_missing_file(self, tmp_path):
        assert wd._read_int(tmp_path / "absent") is None

    def test_read_ts_valid(self, tmp_path):
        p = tmp_path / "f"
        p.write_text(f"{NOW:.6f}\n")
        assert wd._read_ts(p) == pytest.approx(NOW)

    @pytest.mark.parametrize("content", ["junk", "", "nan", "inf", "-inf"])
    def test_read_ts_rejects_garbage_and_non_finite(self, tmp_path, content):
        # nan would silently disarm every deadline comparison; inf would fire
        # instantly. Both must read as "no timestamp".
        p = tmp_path / "f"
        p.write_text(content)
        assert wd._read_ts(p) is None


class TestRun:
    def test_stops_a_stale_server_exactly_once(self, tmp_path):
        sd = make_sd(tmp_path)
        seed(sd, age=100.0, timeout=50)          # idle far past the deadline
        sleeps, stopped = [], []
        wd.run(state_dir=sd, now_fn=lambda: NOW,
               sleep_fn=sleeps.append, stop_fn=lambda: stopped.append(1))
        assert stopped == [1]
        assert 0.2 in sleeps                     # the grace re-read happened

    def test_spares_a_fresh_server(self, tmp_path):
        sd = make_sd(tmp_path)
        seed(sd, age=10.0, timeout=50)
        sleeps, stopped = [], []
        wd.run(state_dir=sd, now_fn=lambda: NOW,
               sleep_fn=sleeps.append, stop_fn=lambda: stopped.append(1),
               max_ticks=3)
        assert stopped == []
        assert sleeps and all(s > 0 for s in sleeps)

    def test_exits_when_the_server_was_stopped(self, tmp_path):
        sd = make_sd(tmp_path)
        seed(sd, pid=None, age=100.0, timeout=50)
        stopped, sleeps = [], []
        wd.run(state_dir=sd, now_fn=lambda: NOW,
               sleep_fn=sleeps.append, stop_fn=lambda: stopped.append(1),
               max_ticks=5)
        # Assert BOTH: no stop AND the loop gave up inside the restart grace
        # instead of running to max_ticks. A plain stopped==[] would pass even
        # with the guard gone -- run() swallows exceptions, so no sentinel.
        assert stopped == [] and sleeps == [wd._RETRY] * (wd._GONE_TICKS - 1)

    @pytest.mark.parametrize("timeout", [None, 0, -3])
    def test_exits_when_instructions_are_retracted(self, tmp_path, timeout):
        # Feature turned off mid-flight: a live watchdog must stand down
        # rather than fire with a deadline from an earlier window.
        sd = make_sd(tmp_path)
        seed(sd, timeout=timeout, age=100.0)
        stopped, sleeps = [], []
        wd.run(state_dir=sd, now_fn=lambda: NOW,
               sleep_fn=sleeps.append, stop_fn=lambda: stopped.append(1),
               max_ticks=5)
        assert stopped == [] and sleeps == [wd._RETRY] * (wd._GONE_TICKS - 1)

    def test_follows_the_server_across_a_restart(self, tmp_path):
        # We hold the singleton lock, so the restart's own spawn already gave
        # up. Exiting here would leave the new server with nothing watching.
        sd = make_sd(tmp_path)
        seed(sd, pid=1, age=100.0, timeout=50)
        stopped = []

        def nap_replaces_the_server(seconds):
            (sd / "server.pid").write_text("222\n")

        wd.run(state_dir=sd, now_fn=lambda: NOW,
               sleep_fn=nap_replaces_the_server,
               stop_fn=lambda: stopped.append(
                   (sd / "server.pid").read_text().strip()),
               max_ticks=5)
        assert stopped == ["222"]

    def test_waits_out_a_gap_in_the_pid_file(self, tmp_path):
        # stop_server unlinks server.pid before start_server writes the new
        # one. Standing down in that window is what stranded the replacement.
        sd = make_sd(tmp_path)
        seed(sd, pid=1, age=100.0, timeout=50)
        (sd / "server.pid").unlink()
        stopped = []

        def nap_starts_the_server(seconds):
            (sd / "server.pid").write_text("222\n")

        wd.run(state_dir=sd, now_fn=lambda: NOW,
               sleep_fn=nap_starts_the_server,
               stop_fn=lambda: stopped.append(
                   (sd / "server.pid").read_text().strip()),
               max_ticks=5)
        assert stopped == ["222"]

    def test_does_not_fire_on_a_pid_that_moved_during_the_grace_nap(
            self, tmp_path):
        # The pid we decided about is not the pid that is there now.
        sd = make_sd(tmp_path)
        seed(sd, pid=1, age=100.0, timeout=50)
        stopped = []
        wd.run(state_dir=sd, now_fn=lambda: NOW,
               sleep_fn=lambda s: (sd / "server.pid").write_text("222\n"),
               stop_fn=lambda: stopped.append(1), max_ticks=1)
        assert stopped == []

    @pytest.mark.parametrize("age,fires", [(49.0, False), (51.0, True)])
    def test_fires_only_past_the_deadline(self, tmp_path, age, fires):
        sd = make_sd(tmp_path)
        seed(sd, age=age, timeout=50)
        stopped = []
        wd.run(state_dir=sd, now_fn=lambda: NOW, sleep_fn=lambda s: None,
               stop_fn=lambda: stopped.append(1), max_ticks=3)
        assert bool(stopped) is fires

    def test_unloads_a_server_that_was_never_queried(self, tmp_path):
        # No last_use at all: age from our own start rather than polling for
        # a timestamp that may never arrive.
        sd = make_sd(tmp_path)
        seed(sd, age=0.0, timeout=50)
        (sd / "server.last_use").unlink()
        clock = [NOW]
        stopped = []
        wd.run(state_dir=sd, now_fn=lambda: clock[0],
               sleep_fn=lambda s: clock.__setitem__(0, clock[0] + s),
               stop_fn=lambda: stopped.append(1), max_ticks=50)
        assert stopped == [1]

    def test_grace_reread_cancels_kill_of_a_query_in_flight(self, tmp_path):
        # A query touched last_use between our staleness read and the grace
        # re-read: the toucher also spawned a fresher watchdog, so we exit
        # without stopping anything.
        sd = make_sd(tmp_path)
        seed(sd, age=100.0, timeout=50)
        stopped = []

        def grace_touches(seconds):
            if seconds == 0.2:                   # the grace sleep only
                (sd / "server.last_use").write_text(f"{time.time():.6f}\n")

        wd.run(state_dir=sd, now_fn=time.time, sleep_fn=grace_touches,
               stop_fn=lambda: stopped.append(1), max_ticks=4)
        assert stopped == []

    def test_lock_held_elsewhere_means_stand_down(self, tmp_path):
        sd = make_sd(tmp_path)
        seed(sd, age=100.0, timeout=50)
        fd = wd._try_lock(sd / "watchdog.lock")
        assert fd is not None                    # we are the other singleton
        try:
            sleeps, stopped = [], []
            wd.run(state_dir=sd, now_fn=lambda: NOW,
                   sleep_fn=sleeps.append,
                   stop_fn=lambda: stopped.append(1), max_ticks=5)
            assert stopped == [] and sleeps == []
        finally:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
            except ImportError:
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)     # unlock needs the locked pos
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)

    def test_try_lock_stands_down_when_neither_primitive_exists(
            self, monkeypatch, tmp_path):
        # Exercises the msvcrt fallback selection off Windows. With no locking
        # primitive at all, duplicate watchdogs would be possible, so the only
        # safe answer is to refuse the lock rather than take it unguarded.
        sd = make_sd(tmp_path)
        monkeypatch.setitem(sys.modules, "fcntl", None)
        monkeypatch.setitem(sys.modules, "msvcrt", None)
        assert wd._try_lock(sd / "watchdog.lock") is None

    def test_run_swallows_a_failing_stop(self, tmp_path):
        # run() is the top of a detached process with stderr at /dev/null;
        # letting anything escape would be an invisible crash that leaves the
        # lock released and nothing watching.
        sd = make_sd(tmp_path)
        seed(sd, age=100.0, timeout=50)

        def boom():
            raise RuntimeError("stop failed")

        wd.run(state_dir=sd, now_fn=lambda: NOW, sleep_fn=lambda s: None,
               stop_fn=boom, max_ticks=5)

    def test_never_raises_on_a_hostile_state_dir(self, tmp_path):
        sd = make_sd(tmp_path)
        (sd / "server.pid").write_text("not-a-pid\n")
        (sd / "server.watch").write_text("\x00\xff\n")
        wd.run(state_dir=sd, now_fn=lambda: "not a number",
               sleep_fn=lambda s: None, stop_fn=lambda: None, max_ticks=2)


class TestMainEntryPoint:
    def test_main_is_a_bare_call_into_run(self, monkeypatch):
        calls = []
        monkeypatch.setattr(wd, "run", lambda *a, **k: calls.append((a, k)))
        assert wd.main() is None
        assert calls == [((), {})]          # no args: state resolved inside
