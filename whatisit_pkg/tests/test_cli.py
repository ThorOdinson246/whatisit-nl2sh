"""Tests for whatisit.cli argument parsing and dispatch.

These cover cases that were REAL bugs found by typing realistic requests:
  - unquoted natural language must not be swallowed by argparse subparsers
    ("look at queued tasks in slurm" once died with "invalid choice: slurm")
  - request text containing flag-like tokens ("find files -name test") must
    survive untouched
  - a subcommand only counts as one when it is the FIRST token
  - --quiet must still run the safety check and refuse a DANGER command

None of this starts a server or touches the network: engine.generate is
monkeypatched wherever cmd_query would otherwise call into it.
"""
import json
import os
import time

import pytest

from whatisit import cli
from whatisit import sessions as sessions_mod

# ------------------------------------------------------------------ QueryArgs

class TestQueryArgsHandParsing:
    def test_unflagged_natural_language_is_not_eaten(self):
        # This exact sentence used to die on argparse subparsers with
        # "invalid choice: 'slurm'" because "slurm" is not a subcommand but
        # looked like trailing positional noise to argparse.
        args = cli.QueryArgs(["look", "at", "queued", "tasks", "in", "slurm"])
        assert args.words == ["look", "at", "queued", "tasks", "in", "slurm"]
        assert args.num == 1
        assert args.execute is False

    def test_request_with_flag_like_text_survives(self):
        args = cli.QueryArgs(["find", "files", "-name", "test"])
        assert args.words == ["find", "files", "-name", "test"]

    def test_double_dash_separator_ends_flag_parsing(self):
        args = cli.QueryArgs(["--", "-n", "3", "do", "the", "thing"])
        assert args.words == ["-n", "3", "do", "the", "thing"]

    def test_dash_n_space_3_form(self):
        args = cli.QueryArgs(["-n", "3", "compress", "this", "folder"])
        assert args.num == 3
        assert args.words == ["compress", "this", "folder"]

    def test_dash_n3_glued_form(self):
        args = cli.QueryArgs(["-n3", "compress", "this", "folder"])
        assert args.num == 3
        assert args.words == ["compress", "this", "folder"]

    def test_leading_flags_before_request_are_all_consumed(self):
        args = cli.QueryArgs(["-e", "-q", "-t", "count", "lines"])
        assert args.execute is True
        assert args.quiet is True
        assert args.timing is True
        assert args.words == ["count", "lines"]

    def test_oneshot_flag(self):
        args = cli.QueryArgs(["--oneshot", "do", "a", "thing"])
        assert args.oneshot is True
        assert args.words == ["do", "a", "thing"]

    def test_missing_n_argument_raises(self):
        with pytest.raises(ValueError):
            cli.QueryArgs(["-n"])

    def test_no_flags_at_all(self):
        args = cli.QueryArgs(["show", "disk", "usage"])
        assert args.words == ["show", "disk", "usage"]
        assert args.num == 1

    def test_empty_argv(self):
        args = cli.QueryArgs([])
        assert args.words == []


# ------------------------------------------------------------------- routing

class TestSubcommandRoutingIsFirstTokenOnly:
    def test_subcommand_as_first_token_routes_to_subparser(self, monkeypatch):
        called = {}

        def fake_cmd_config(args, cfg):
            called["hit"] = True
            return 0

        monkeypatch.setattr(cli, "cmd_config", fake_cmd_config)
        # rebuild parser mapping so "config" dispatches to our stub
        parser = cli.build_parser()
        monkeypatch.setattr(cli, "build_parser", lambda: parser)
        rc = cli.main(["config"])
        assert rc == 0
        assert called.get("hit") is True

    def test_subcommand_word_buried_in_sentence_is_not_routed(self, monkeypatch):
        # "whatisit show me the git config" must stay a plain-English question,
        # not be treated as the `config` subcommand, because "config" is not
        # the first token.
        captured = {}

        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                          for_execution=False, extra_context=None):
            captured["prompt"] = prompt
            return (["git config --list"], 0.01, "server")

        monkeypatch.setattr(cli.engine, "generate", fake_generate)
        rc = cli.main(["show", "me", "the", "git", "config"])
        assert rc == 0
        assert captured["prompt"] == "show me the git config"

    def test_setup_doctor_stop_session_config_are_recognized_subcommands(self):
        assert cli.SUBCOMMANDS == {"setup", "doctor", "stop", "session", "config"}


# --------------------------------------------------------------- cmd_query

class TestCmdQueryQuietDangerRefusal:
    def test_quiet_stdout_carries_only_the_command(self, monkeypatch, capsys):
        """`eval "$(whatisit -q ...)"` runs whatever lands on stdout."""
        monkeypatch.setattr(
            cli.engine, "generate",
            lambda prompt, cfg, n=1, force_oneshot=False, quiet=False,
                    for_execution=False, extra_context=None:
                (["chmod 777 ./build"], 0.01, "server"))
        rc = cli.main(["-q", "fix", "permissions"])
        cap = capsys.readouterr()
        assert rc == 0
        assert cap.out == "chmod 777 ./build\n"
        assert "caution" in cap.err

    def test_quiet_refuses_danger_command_exit_6(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli.engine, "generate",
            lambda prompt, cfg, n=1, force_oneshot=False, quiet=False,
                    for_execution=False, extra_context=None:
                (["rm -rf /"], 0.01, "server"))
        rc = cli.main(["-q", "delete", "everything"])
        assert rc == 6
        out = capsys.readouterr()
        # stdout must stay bare even when refusing -- nothing should be
        # emitted there that a $(...) capture could pick up and run.
        assert out.out == ""

    def test_quiet_prints_bare_command_on_success(self, monkeypatch, capsys):
        captured = {}

        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                          for_execution=False, extra_context=None):
            captured["for_execution"] = for_execution
            return (["ls -la"], 0.01, "server")

        monkeypatch.setattr(cli.engine, "generate", fake_generate)
        rc = cli.main(["-q", "list", "files"])
        assert rc == 0
        assert captured["for_execution"] is True
        out = capsys.readouterr()
        assert out.out.strip() == "ls -la"

    def test_execute_marks_generation_as_execution_intended(
            self, monkeypatch, tmp_path):
        # Isolate config too: an ambient confirm_execute=false on a dev box
        # would skip the refusal this test exists to pin (latent, pre-dating
        # sessions -- surfaced here).
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        captured = {}

        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                          for_execution=False, extra_context=None):
            captured["for_execution"] = for_execution
            return (["ls -la"], 0.01, "server")

        monkeypatch.setattr(cli.engine, "generate", fake_generate)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
        # -e is refused outright on Windows (exit 7) before the no-tty
        # confirmation path (exit 6) is reached, so pin the platform.
        monkeypatch.setattr(cli, "_is_windows", lambda: False)
        assert cli.main(["-e", "list", "files"]) == 6
        assert captured["for_execution"] is True

    def test_no_model_found_reports_and_exits_3(self, monkeypatch):
        def raise_not_found(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                            for_execution=False, extra_context=None):
            raise FileNotFoundError("no model found -- run `whatisit setup`")
        monkeypatch.setattr(cli.engine, "generate", raise_not_found)
        rc = cli.main(["do", "something"])
        assert rc == 3

    def test_empty_request_after_flags_only(self):
        rc = cli.main(["-q"])
        assert rc == 0  # falls through to help, per main()'s "no words" branch

    def test_refuse_execute_in_windows(self, monkeypatch, capsys):
        # Patch the helper, not os.name: setting os.name="nt" on Linux makes
        # Path.home() (via load_config in main) raise RuntimeError.
        monkeypatch.setattr(cli, "_is_windows", lambda: True)
        monkeypatch.setattr(
            cli.engine, "generate",
            lambda prompt, cfg, n=1, force_oneshot=False, quiet=False,
                    for_execution=False, extra_context=None:
                (["ls -la"], 0.01, "server"))
        rc = cli.main(["-e", "list", "files"])
        assert rc == 7
        assert "disabled" in capsys.readouterr().err

    def test_execute_is_not_refused_off_windows(self, monkeypatch, capsys):
        # Without this, a _is_windows() stuck at True would pass the test above
        # and silently disable -e everywhere.
        monkeypatch.setattr(cli, "_is_windows", lambda: False)
        monkeypatch.setattr(
            cli.engine, "generate",
            lambda prompt, cfg, n=1, force_oneshot=False, quiet=False,
                    for_execution=False, extra_context=None:
                (["ls -la"], 0.01, "server"))
        rc = cli.main(["-e", "list", "files"])
        assert rc != 7
        assert "disabled" not in capsys.readouterr().err

# ------------------------------------------------------------------ parser

class TestBuildParser:
    def test_help_flag_does_not_crash(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["-h"])
        assert exc.value.code == 0

    def test_config_subparser_accepts_set(self):
        parser = cli.build_parser()
        args = parser.parse_args(["config", "--set", "threads=2"])
        assert args.sub == "config"
        assert args.set == ["threads=2"]

    def test_setup_subparser_flags(self):
        parser = cli.build_parser()
        args = parser.parse_args(["setup", "--model", "/tmp/m.gguf", "--copy"])
        assert args.model == "/tmp/m.gguf"
        assert args.copy is True


def test_trailing_flag_is_part_of_the_request_but_is_flagged():
    """`whatisit list files -e` sends "-e" to the model. That is deliberate
    (see QueryArgs) but silent, so it must at least be reported."""
    a = cli.QueryArgs(["list", "files", "-e"])
    assert a.execute is False
    assert a.words == ["list", "files", "-e"]
    assert a.stray_flags == ["-e"]


def test_leading_flag_still_acts_as_a_flag():
    a = cli.QueryArgs(["-e", "list", "files"])
    assert a.execute is True
    assert a.words == ["list", "files"]
    assert a.stray_flags == []


def test_double_dash_ends_flags_and_silences_the_note():
    """`--` means the user meant it literally, so no note."""
    a = cli.QueryArgs(["list", "files", "--", "-e"])
    assert a.words == ["list", "files", "-e"]
    assert a.stray_flags == []


def test_flag_shaped_words_that_are_not_our_flags_are_left_alone():
    """`find files -name test` must not warn: -name is not one of our flags."""
    a = cli.QueryArgs(["find", "files", "-name", "test"])
    assert a.words == ["find", "files", "-name", "test"]
    assert a.stray_flags == []


def test_subcommand_word_inside_a_question_stays_a_question():
    a = cli.QueryArgs(["how", "do", "I", "stop", "a", "stuck", "process"])
    assert a.words[0] == "how"
    assert a.stray_flags == []


class TestRemoteCli:
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        for suff in ("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"):
            for pref in ("WHATISIT_", "NL2SH_"):
                monkeypatch.delenv(pref + suff, raising=False)

    def test_config_redacts_api_key(self, monkeypatch, tmp_path, capsys):
        self._isolate(monkeypatch, tmp_path)
        rc = cli.main(["config", "--set",
                       "openai_base_url=http://h/v1",
                       "openai_api_key=sk-secret"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "sk-secret" not in out
        assert "********" in out

    def test_doctor_fails_without_model(self, monkeypatch, tmp_path, capsys):
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(cli.engine, "list_remote_models", lambda remote: [])
        rc = cli.main(["config", "--set", "openai_base_url=http://h/v1"])
        assert rc == 0
        rc = cli.main(["doctor"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "Not ready" in out
        assert "not set" in out

    def test_query_warns_on_stderr_only(self, monkeypatch, tmp_path, capsys):
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setenv("WHATISIT_OPENAI_BASE_URL", "http://192.0.2.8/v1")
        monkeypatch.setenv("WHATISIT_OPENAI_MODEL", "m")

        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                          for_execution=False, extra_context=None):
            return (["ls"], 0.01, "remote")

        monkeypatch.setattr(cli.engine, "generate", fake_generate)
        rc = cli.main(["-q", "list", "files"])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == "ls"
        assert "leaves this machine" in captured.err


# --------------------------------------------------------------- QueryArgs extras

def test_port_threads_model_flags_parse():
    a = cli.QueryArgs(["--port", "9000", "--threads", "8", "--model",
                       "/models/3b.gguf", "list", "files"])
    assert a.port == 9000
    assert a.threads == 8
    assert a.model == "/models/3b.gguf"
    assert a.words == ["list", "files"]


def test_ctx_size_flag_parse():
    a = cli.QueryArgs(["--ctx-size", "4096", "list", "files"])
    assert a.ctx_size == 4096
    assert a.words == ["list", "files"]


def test_no_host_context_and_no_grammar_flags():
    a = cli.QueryArgs(["--no-host-context", "--no-grammar", "list", "files"])
    assert a.host_context is False
    assert a.grammar is False
    assert a.words == ["list", "files"]


def test_flag_needs_value_raises():
    with pytest.raises(ValueError):
        cli.QueryArgs(["--port"])
    with pytest.raises(ValueError):
        cli.QueryArgs(["--threads"])
    with pytest.raises(ValueError):
        cli.QueryArgs(["--model"])


def test_debug_flag_is_set():
    a = cli.QueryArgs(["--debug", "list", "files"])
    assert a.debug is True
    assert a.words == ["list", "files"]


def test_yes_flag_and_enable_overrides_parse():
    a = cli.QueryArgs(["-y", "--host-context", "--grammar", "list", "files"])
    assert a.yes is True
    assert a.host_context is True
    assert a.grammar is True
    assert a.words == ["list", "files"]


def test_enable_and_disable_host_context_are_mutually_consistent():
    a = cli.QueryArgs(["--host-context", "list", "files"])
    assert a.host_context is True
    b = cli.QueryArgs(["--no-host-context", "list", "files"])
    assert b.host_context is False


def test_port_range_is_validated():
    with pytest.raises(ValueError, match="1\\.\\.65535"):
        cli.QueryArgs(["--port", "0", "list", "files"])
    with pytest.raises(ValueError, match="1\\.\\.65535"):
        cli.QueryArgs(["--port", "70000", "list", "files"])
    assert cli.QueryArgs(["--port", "65535", "list", "files"]).port == 65535


def test_ctx_size_and_threads_ranges_are_validated():
    with pytest.raises(ValueError, match="> 0"):
        cli.QueryArgs(["--ctx-size", "0", "list", "files"])
    with pytest.raises(ValueError, match="> 0"):
        cli.QueryArgs(["--ctx-size", "-1", "list", "files"])
    with pytest.raises(ValueError, match=">= 0"):
        cli.QueryArgs(["--threads", "-1", "list", "files"])


def test_value_taking_query_flags_are_flagged_when_stray():
    a = cli.QueryArgs(["list", "files", "--port", "9000"])
    assert a.port is None
    assert a.stray_flags == ["--port"]
    b = cli.QueryArgs(["list", "files", "--ctx-size", "4096"])
    assert b.stray_flags == ["--ctx-size"]


# --------------------------------------------------------------- cmd_query wiring

class TestQueryFlagsApply:
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))

    def test_port_thread_model_overrides_config(self, monkeypatch, tmp_path, capsys):
        self._isolate(monkeypatch, tmp_path)
        seen = {}
        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                          for_execution=False, extra_context=None):
            seen["cfg"] = dict(cfg)
            return (["ls -la"], 0.01, "server")
        monkeypatch.setattr(cli.engine, "generate", fake_generate)
        rc = cli.main(["--port", "9100", "--threads", "2",
                       "--model", "/m.gguf", "list", "files"])
        assert rc == 0
        assert seen["cfg"]["server_port"] == 9100
        assert seen["cfg"]["threads"] == 2
        assert os.environ["WHATISIT_MODEL"] == "/m.gguf"

    def test_no_grammar_disables_grammar(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        seen = {}
        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                          for_execution=False, extra_context=None):
            seen["cfg"] = dict(cfg)
            return (["ls"], 0.01, "server")
        monkeypatch.setattr(cli.engine, "generate", fake_generate)
        rc = cli.main(["--no-grammar", "list", "files"])
        assert rc == 0
        assert seen["cfg"]["use_grammar"] is False

    def test_debug_emits_prompt_to_stderr(self, monkeypatch, tmp_path, capsys):
        self._isolate(monkeypatch, tmp_path)
        # Only assert the debug output, not actual execution.
        monkeypatch.setattr(cli.engine, "generate",
                            lambda *a, **k: (["ls"], 0.01, "server"))
        monkeypatch.setattr(cli.engine.hostctx, "build",
                            lambda p, enabled=True, cwd=None, include_volatile=True,
                            extra_context=None: ("SYS", p))
        rc = cli.main(["--debug", "list", "files"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "--debug" in err
        assert "list files" in err

    def test_debug_shows_grammar_when_available(self, monkeypatch, tmp_path, capsys):
        self._isolate(monkeypatch, tmp_path)
        cli.main(["config", "--set", "host_context=true"])
        capsys.readouterr()
        monkeypatch.setattr(cli.engine, "generate",
                            lambda *a, **k: (["ls"], 0.01, "server"))
        monkeypatch.setattr(cli.engine.hostctx, "build",
                            lambda p, enabled=True, cwd=None, include_volatile=True,
                            extra_context=None: ("SYS", p))
        monkeypatch.setattr(cli.engine.hostctx, "stable_facts",
                            lambda *a, **k: {"pkg": "pacman"})
        # grammar_for_pkg may not exist (PR A vs PR B); if so, create a stub
        # so the debug path can derive a grammar through the hasattr guard.
        # raising=False restores the original after the test either way.
        monkeypatch.setattr(cli.engine.hostctx, "grammar_for_pkg",
                            lambda pkg: "grammar-blob", raising=False)
        rc = cli.main(["--debug", "list", "files"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "--debug" in err
        assert "grammar-blob" in err
        assert "list files" in err


# ---------------------------------------------------------- session memory

class TestSessionMemory:
    """Opt-in cross-invocation memory (sessions.py). Default OFF: every test
    here either passes --session or pre-sets config sessions=true."""

    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))

    @staticmethod
    def _fake_generate(monkeypatch, captured, cmds):
        def fake_generate(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                          for_execution=False, extra_context=None):
            captured["prompt"] = prompt
            captured["extra_context"] = extra_context
            return (cmds, 0.01, "server")
        monkeypatch.setattr(cli.engine, "generate", fake_generate)

    def _session_file(self, tmp_path):
        return tmp_path / "data" / "session.jsonl"

    def test_session_flag_parses_before_the_request(self):
        a = cli.QueryArgs(["--session", "find", "logs"])
        assert a.session is True
        assert a.words == ["find", "logs"]

    def test_off_by_default_records_nothing(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls"])
        rc = cli.main(["execute", "them"])
        assert rc == 0
        assert not self._session_file(tmp_path).exists()
        assert captured["extra_context"] is None

    def test_followup_injects_prior_turn(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls *.py | head -3"])
        cli.main(["--session", "list", "three", "python", "files", "here"])
        self._fake_generate(monkeypatch, captured, ["rm one.py"])
        rc = cli.main(["--session", "delete", "them"])
        assert rc == 0
        assert captured["extra_context"] == (
            'Prior: "list three python files here" -> ls *.py | head -3')

    def test_non_anaphoric_gets_no_context_but_still_records(
            self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls -la"])
        cli.main(["--session", "list", "files", "in", "this", "folder"])
        assert captured["extra_context"] is None
        lines = self._session_file(tmp_path).read_text().splitlines()
        assert len(lines) == 1

    def test_quiet_never_receives_context_but_records(
            self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls *.py"])
        cli.main(["--session", "list", "python", "files"])
        self._fake_generate(monkeypatch, captured, ["rm one.py"])
        rc = cli.main(["--session", "-q", "execute", "them"])
        assert rc == 0
        assert captured["extra_context"] is None
        assert len(self._session_file(tmp_path).read_text().splitlines()) == 2

    def test_enabled_via_config_not_flag(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config.json").write_text('{"sessions": true}')
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls"])
        cli.main(["list", "python", "files"])
        assert self._session_file(tmp_path).exists()

    def test_danger_command_is_never_recorded(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["rm -rf /"])
        rc = cli.main(["--session", "delete", "them"])
        assert rc == 0
        assert not self._session_file(tmp_path).exists()

    def test_caution_command_is_recorded(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["chmod 777 ./build"])
        rc = cli.main(["--session", "open", "permissions", "on", "it"])
        assert rc == 0
        lines = self._session_file(tmp_path).read_text().splitlines()
        turns = [json.loads(line) for line in lines]
        assert turns[-1]["command"] == "chmod 777 ./build"

    def test_stale_session_is_reset_before_injection(self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        data = tmp_path / "data"
        data.mkdir(parents=True)
        stale = time.time() - sessions_mod.TTL_SECONDS - 1
        turn = {"ts": stale, "cwd": os.getcwd(), "nl": "old request",
                "command": "old-cmd", "executed": False, "exit_code": None}
        seed = data / "session.jsonl"
        seed.write_text(json.dumps(turn) + "\n")
        # Seed at 0600 or the permission rule resets the file before the
        # stale timestamp is ever consulted -- this test pins the TTL rule,
        # not the perms rule (which has its own tests).
        if os.name != "nt":
            os.chmod(seed, 0o600)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["rm one.py"])
        cli.main(["--session", "delete", "them"])
        # Expired history must not reach the prompt. The file exists again
        # afterwards: this very run recorded a fresh turn over the reset.
        assert captured["extra_context"] is None
        lines = self._session_file(tmp_path).read_text().splitlines()
        turns = [json.loads(line) for line in lines]
        assert [t["nl"] for t in turns] == ["delete them"]
        assert all(t["ts"] > time.time() - 60 for t in turns)

    def test_execute_overwrites_with_actual_command_and_rc(
            self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config.json").write_text('{"confirm_execute": false}')

        class _R:
            returncode = 5

        monkeypatch.setattr(cli, "_is_windows", lambda: False)

        def fake_run(*a, **k):
            return _R()

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls -la"])
        rc = cli.main(["--session", "-e", "-y", "list", "files"])
        assert rc == 5
        lines = self._session_file(tmp_path).read_text().splitlines()
        turns = [json.loads(line) for line in lines]
        assert turns[-1]["executed"] is True
        assert turns[-1]["exit_code"] == 5

    def test_danger_top_candidate_never_rewrites_the_previous_turn(
            self, monkeypatch, tmp_path):
        # Regression: with -n 2 whose greedy candidate is DANGER, nothing is
        # recorded for this run -- so executing the safe alternative must
        # NOT rewrite turns[-1], which still belongs to the prior invocation.
        self._isolate(monkeypatch, tmp_path)
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config.json").write_text('{"confirm_execute": false}')
        monkeypatch.setattr(cli, "_is_windows", lambda: False)

        class _R:
            returncode = 0

        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _R())
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls *.py"])
        cli.main(["--session", "list", "python", "files"])

        # This run: candidate 1 flagged DANGER ("rm -rf /" is a critical
        # path), candidate 2 executable.
        def fake_generate2(prompt, cfg, n=1, force_oneshot=False, quiet=False,
                           for_execution=False, extra_context=None):
            captured["prompt"] = prompt
            return (["rm -rf /", "rm ./tmp.log"], 0.01, "server")
        monkeypatch.setattr(cli.engine, "generate", fake_generate2)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *a: "2")
        rc = cli.main(["--session", "-e", "clean", "this", "up"])
        assert rc == 0
        lines = self._session_file(tmp_path).read_text().splitlines()
        turns = [json.loads(line) for line in lines]
        assert len(turns) == 1
        assert turns[0]["nl"] == "list python files"
        assert turns[0]["command"] == "ls *.py"
        assert turns[0]["executed"] is False

    def test_session_subcommand_show_and_clear(self, monkeypatch, tmp_path,
                                               capsys):
        self._isolate(monkeypatch, tmp_path)
        assert cli.main(["session", "show"]) == 0
        assert "no session stored" in capsys.readouterr().out
        captured = {}
        self._fake_generate(monkeypatch, captured, ["ls"])
        cli.main(["--session", "list", "files"])
        capsys.readouterr()
        assert cli.main(["session", "show"]) == 0
        out = capsys.readouterr().out
        assert "list files" in out and "ls" in out
        assert cli.main(["session", "clear"]) == 0
        assert "cleared" in capsys.readouterr().out
        assert not self._session_file(tmp_path).exists()

    def test_request_containing_session_word_midway_is_a_query(self):
        a = cli.QueryArgs(["show", "my", "session", "history"])
        assert a.words[0] == "show"
