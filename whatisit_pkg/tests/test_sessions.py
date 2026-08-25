"""Tests for whatisit.sessions: NDJSON turn storage, owner-only file
discipline, TTL/cwd invalidation, and anaphora detection.

Every test isolates itself via monkeypatch -- none may read or write the real
~/.local/share, and none may depend on the ambient working directory.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from whatisit import sessions


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated data dir + working directory. Returns the data dir path."""
    data = tmp_path / "data"
    monkeypatch.setenv("WHATISIT_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    return data


def _seed(iso, turns):
    """Write raw NDJSON directly, bypassing record(). Seeded owner-only: a
    wide-open session file is by definition an invalid one (see load_valid),
    so tests simulating a legitimate store must match its permissions."""
    iso.mkdir(parents=True, exist_ok=True)
    p = iso / "session.jsonl"
    p.write_text("".join(json.dumps(t) + "\n" for t in turns))
    if os.name != "nt":
        os.chmod(p, 0o600)
    return p


def _turn(nl="list files", command="ls", **over):
    t = {"ts": time.time(), "cwd": os.getcwd(), "nl": nl,
         "command": command, "executed": False, "exit_code": None}
    t.update(over)
    return t


# ---------------------------------------------------------------- NDJSON I/O

class TestStorage:
    def test_record_writes_one_json_object_per_line(self, iso):
        sessions.record("list files", "ls")
        sessions.record("show disk usage", "df -h")
        lines = (iso / "session.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["nl"] == "list files"
        assert json.loads(lines[1])["command"] == "df -h"

    def test_fourth_turn_drops_the_oldest(self, iso):
        for i in range(4):
            sessions.record(f"request {i}", f"cmd {i}")
        turns = sessions.show()
        assert [t["nl"] for t in turns] == ["request 1", "request 2", "request 3"]

    def test_corrupt_line_is_skipped_not_fatal(self, iso):
        p = _seed(iso, [_turn(nl="a")])
        with open(p, "a") as f:
            f.write("{this is not json\n")
            f.write(json.dumps(_turn(nl="b")) + "\n")
        assert [t["nl"] for t in sessions.show()] == ["a", "b"]
        # And recording on top of a corrupt line must keep the valid ones.
        sessions.record("c", "cmd-c")
        assert [t["nl"] for t in sessions.show()] == ["a", "b", "c"]

    def test_blank_lines_are_ignored(self, iso):
        p = _seed(iso, [_turn(nl="a")])
        with open(p, "a") as f:
            f.write("\n   \n")
            f.write(json.dumps(_turn(nl="b")) + "\n")
        assert [t["nl"] for t in sessions.show()] == ["a", "b"]

    def test_tail_read_survives_a_oversized_garbage_prefix(self, iso):
        p = _seed(iso, [])
        junk = "x" * (sessions._TAIL_BYTES * 3)
        p.write_text(junk + "\n" + json.dumps(_turn(nl="kept")) + "\n")
        # The junk line is far beyond the tail window; whatever fragment of it
        # is read must parse to nothing, never crash or resurrect itself.
        assert [t["nl"] for t in sessions.show()] == ["kept"]

    def test_non_dict_or_wrongly_typed_lines_are_skipped(self, iso):
        p = iso / "session.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('["a","list"]\n' + json.dumps(_turn()) + "\n")
        assert len(sessions.show()) == 1

    def test_missing_file_reads_as_empty(self, iso):
        assert sessions.show() == []
        assert sessions.load_valid() == []

    def test_nl_and_command_truncated_before_storage(self, iso):
        sessions.record("n" * 5000, "c" * 5000)
        t = sessions.show()[0]
        assert len(t["nl"]) == sessions.MAX_NL_CHARS
        assert len(t["command"]) == sessions.MAX_CMD_CHARS

    def test_show_has_no_reset_side_effects(self, iso, monkeypatch):
        # A stale file survives show() -- it is an audit view. load_valid is
        # what decides usability.
        old = time.time() - sessions.TTL_SECONDS - 60
        p = _seed(iso, [_turn(ts=old)])
        assert sessions.show()
        assert p.exists()


class TestUpdateExecuted:
    def test_overwrites_only_the_newest_turn(self, iso):
        sessions.record("first", "wrong")
        sessions.record("second", "ls -la")
        sessions.update_executed("ls -la /tmp", 3)
        turns = sessions.show()
        assert turns[0]["command"] == "wrong"
        assert turns[0]["executed"] is False
        assert turns[-1]["command"] == "ls -la /tmp"
        assert turns[-1]["executed"] is True
        assert turns[-1]["exit_code"] == 3

    def test_noop_on_empty_store(self, iso):
        sessions.update_executed("anything", 0)
        assert sessions.show() == []

    def test_exit_code_coerced_to_int(self, iso):
        sessions.record("x", "y")
        sessions.update_executed("y", "7")
        assert sessions.show()[-1]["exit_code"] == 7


# ------------------------------------------------------------- permissions

needs_posix = pytest.mark.skipif(os.name == "nt",
                                 reason="POSIX permission semantics")


class TestPermissions:
    @needs_posix
    def test_created_owner_only(self, iso):
        sessions.record("list files", "ls")
        p = iso / "session.jsonl"
        assert (p.stat().st_mode & 0o777) == 0o600
        assert (iso.stat().st_mode & 0o777) == 0o700

    @needs_posix
    def test_pre_existing_loose_file_is_repaired_on_rewrite(self, iso):
        p = _seed(iso, [_turn()])
        os.chmod(p, 0o644)
        sessions.record("another", "cmd")     # rewrite goes through fchmod
        assert (p.stat().st_mode & 0o777) == 0o600

    @needs_posix
    def test_wrong_permissions_reset_the_session(self, iso):
        p = _seed(iso, [_turn()])
        os.chmod(p, 0o666)
        assert sessions.load_valid() == []
        assert not p.exists()

    @needs_posix
    def test_planted_symlink_is_never_followed(self, iso, tmp_path):
        sentinel = tmp_path / "sentinel.txt"
        sentinel.write_text("do not touch")
        iso.mkdir(parents=True, exist_ok=True)
        (iso / "session.jsonl").symlink_to(sentinel)
        # Must neither raise nor write through the link; last-writer-wins
        # means silently doing nothing is the correct outcome here.
        sessions.record("list files", "ls")
        assert sentinel.read_text() == "do not touch"


# ------------------------------------------------------------ TTL and cwd

class TestInvalidation:
    def test_fresh_turn_is_kept(self, iso):
        _seed(iso, [_turn()])
        assert [t["nl"] for t in sessions.load_valid()] == ["list files"]

    def test_ttl_expiry_resets(self, iso):
        old = time.time() - sessions.TTL_SECONDS - 1
        p = _seed(iso, [_turn(ts=old)])
        assert sessions.load_valid() == []
        assert not p.exists()

    def test_turn_just_inside_ttl_is_kept(self, iso):
        edge = time.time() - sessions.TTL_SECONDS + 30
        _seed(iso, [_turn(ts=edge)])
        assert sessions.load_valid()

    def test_cwd_change_resets(self, iso, tmp_path, monkeypatch):
        _seed(iso, [_turn()])
        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
        p = iso / "session.jsonl"
        assert sessions.load_valid() == []
        assert not p.exists()

    @needs_posix
    def test_cwd_comparison_is_symlink_normalized(self, iso, tmp_path, monkeypatch):
        # A session recorded through one path to a directory stays valid when
        # the shell reaches the same directory through another name.
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        monkeypatch.chdir(real)
        _seed(iso, [{"ts": time.time(), "cwd": str(link), "nl": "a",
                     "command": "b", "executed": False, "exit_code": None}])
        assert sessions.load_valid()

    def test_entirely_corrupt_file_resets(self, iso):
        p = iso / "session.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(256))
        assert sessions.load_valid() == []
        assert not p.exists()


# ----------------------------------------------------------------- anaphora

class TestLooksAnaphoric:
    @pytest.mark.parametrize("prompt", [
        "execute them",
        "delete those",
        "open these",
        "run it again",
        "can you please remove them",
        "rename that one",
        "redo the same for logs",
        "copy this one over",
        "them again please",
        "Please Delete Them",
    ])
    def test_positives(self, prompt):
        assert sessions.looks_anaphoric(prompt) is True

    @pytest.mark.parametrize("prompt", [
        "find files that are bigger than 100MB",
        "show processes that use the most memory",
        "list python files in this directory",
        "what time is it",
        "install htop",
        "compress the logs folder into a tarball",
        "delete temporary files older than a week",
        "",
    ])
    def test_negatives(self, prompt):
        assert sessions.looks_anaphoric(prompt) is False


# ------------------------------------------------------------ block format

class TestHistoryBlock:
    def test_exact_tight_format(self, iso):
        block = sessions.history_block([
            {"nl": "list three python files here", "command": "ls *.py | head -3"}])
        assert block == 'Prior: "list three python files here" -> ls *.py | head -3'

    def test_multiple_turns_one_line_each(self, iso):
        block = sessions.history_block([{"nl": "a", "command": "b"},
                                        {"nl": "c", "command": "d"}])
        assert block == 'Prior: "a" -> b\nPrior: "c" -> d'

    def test_empty_history_gives_none(self, iso):
        assert sessions.history_block([]) is None


# ------------------------------------------------------------------ clear

class TestClear:
    def test_clear_removes_and_reports(self, iso):
        sessions.record("x", "y")
        assert sessions.clear() is True
        assert not (iso / "session.jsonl").exists()
        assert sessions.clear() is False

    def test_clear_when_nothing_stored(self, iso):
        assert sessions.clear() is False
