"""Tests for whatisit.engine, with the HTTP layer entirely mocked out.

No test here ever starts llama-server, touches the network, or requires a
model file: engine._post / engine._query_server / engine.start_server are all
monkeypatched, and "model" paths are just empty tmp_path files that only need
to exist.
"""
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from whatisit import config as cfg_mod
from whatisit import engine

# ---------------------------------------------------------- looks_degenerate

class TestLooksDegenerate:
    def test_short_command_is_never_degenerate(self):
        assert engine.looks_degenerate("ls -la") is False

    def test_ordinary_long_command_is_not_degenerate(self):
        cmd = "find . -type f -name '*.py' -newer ref.txt -exec grep -l TODO {} +"
        assert engine.looks_degenerate(cmd) is False

    def test_flag_spam_loop_is_degenerate(self):
        # Same shape as the reproducible real-world zip failure from the
        # docstring, extended so the repeated flag hits the default
        # min_repeats=4 threshold.
        cmd = "zip -r -9 -q -n -j -0 -9 -n -j -0 -9 -n -j -0 -9 -n -j -0 archive.zip ."
        assert engine.looks_degenerate(cmd) is True

    def test_repeated_flag_below_threshold_is_not_degenerate(self):
        cmd = "tar -czvf out.tar.gz -C dir1 a -C dir2 b -C dir3 c file.txt"
        # "-C" repeats only 3 times here, one below the default min_repeats=4.
        assert engine.looks_degenerate(cmd, min_repeats=4) is False

    def test_custom_min_repeats_threshold(self):
        cmd = "prog -x -x -x file1 file2 file3 file4 file5"
        assert engine.looks_degenerate(cmd, min_repeats=3) is True
        assert engine.looks_degenerate(cmd, min_repeats=4) is False


# --------------------------------------------------------------- CLI chrome

class TestStripCliChrome:
    def test_extracts_answer_between_echo_and_footer(self):
        out = (
            "llama-cli banner text\n"
            "loading model...\n"
            "> find files bigger than 100MB\n"
            "find . -size +100M\n"
            "[ Prompt: 12 tokens, 1.5 ms/token ]\n"
        )
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_uses_last_echo_line_when_several_present(self):
        # --no-display-prompt does not suppress the prompt echo, and in a
        # multi-turn-looking transcript there can be more than one "> " line;
        # the real answer follows the LAST one.
        out = (
            "> irrelevant earlier turn\n"
            "some old answer\n"
            "> find files bigger than 100MB\n"
            "find . -size +100M\n"
            "Exiting...\n"
        )
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_strips_ansi_codes(self):
        out = "> prompt\n\x1b[32mfind . -size +100M\x1b[0m\n[ Prompt: done ]\n"
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_no_footer_present_still_returns_body(self):
        out = "> prompt\nfind . -size +100M\n"
        assert engine._strip_cli_chrome(out) == "find . -size +100M"

    def test_carriage_returns_normalized(self):
        out = "> prompt\rfind . -size +100M\r[ Prompt: done ]\r"
        assert engine._strip_cli_chrome(out) == "find . -size +100M"


# ------------------------------------------------------------- private write

class TestWritePrivate:
    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Unix permission bits are not enforced on Windows")
    def test_creates_file_with_0600(self, tmp_path):
        p = tmp_path / "server.token"
        engine._write_private(p, "supersecret")
        assert p.read_text() == "supersecret"
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    def test_overwrites_existing_content(self, tmp_path):
        p = tmp_path / "server.pid"
        p.write_text("old")
        engine._write_private(p, "new")
        assert p.read_text() == "new"

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Unix permission bits are not enforced on Windows")
    def test_pre_existing_loosely_permissioned_file_is_tightened(self, tmp_path):
        p = tmp_path / "server.token"
        p.write_text("old")
        os.chmod(p, 0o644)
        engine._write_private(p, "new-secret-token")
        assert stat.S_IMODE(p.stat().st_mode) == 0o600

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="os.O_NOFOLLOW is not available on Windows")
    def test_refuses_to_follow_a_symlink(self, tmp_path):
        target = tmp_path / "outside_file.txt"
        target.write_text("do not touch me")
        link = tmp_path / "server.token"
        link.symlink_to(target)
        try:
            engine._write_private(link, "pwned")
        except OSError:
            pass  # ELOOP is the expected, safe outcome
        # Either way, the symlink target must be untouched.
        assert target.read_text() == "do not touch me"


# --------------------------------------------------------- TCP authentication

class TestTcpServerIdentity:
    @pytest.fixture(autouse=True)
    def _socket_inspection_available(self, monkeypatch):
        # These exercise the attribution logic itself, not whether the host can
        # inspect sockets. Without this they depend on /proc or lsof existing --
        # false on Windows, and in a minimal container.
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: True)

    @pytest.mark.parametrize(
        ("checker", "args", "state"),
        [
            ("_pid_owns_tcp_port", (1234, 43210), "0A"),
            ("_pid_owns_tcp_connection", (1234, 43210, 54321), "01"),
        ],
    )
    @pytest.mark.parametrize(
        ("local_address", "remote_address"),
        [
            ("00000000:NOT_HEX", "00000000:ALSO_BAD"),
            ("MISSING_SEPARATOR", "ALSO_MISSING_SEPARATOR"),
        ],
    )
    def test_malformed_proc_port_fails_closed(
            self, monkeypatch, checker, args, state, local_address, remote_address):
        class ProcPath:
            def __init__(self, path):
                self.path = path

            def __str__(self):
                return self.path

            def is_dir(self):
                return True

            def iterdir(self):
                return [ProcPath(f"{self.path}/1")]

            def read_text(self):
                return (
                    f"header\n0: {local_address} {remote_address} "
                    f"{state} 0 0 0 0 0 12345\n"
                )

        monkeypatch.setattr(engine, "Path", ProcPath)
        monkeypatch.setattr(engine.os, "readlink", lambda _path: "socket:[12345]")
        monkeypatch.setattr(engine.shutil, "which", lambda _name: None)

        assert getattr(engine, checker)(*args) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX socket inspection")
    def test_current_process_owns_its_listening_port(self):
        with socket.socket() as listener:
            listener.bind((engine.HOST, 0))
            listener.listen()
            assert engine._pid_owns_tcp_port(os.getpid(), listener.getsockname()[1]) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX socket inspection")
    def test_current_process_owns_its_established_connection(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.bind((engine.HOST, 0))
            listener.listen()
            server_port = listener.getsockname()[1]
            client.connect((engine.HOST, server_port))
            accepted, _ = listener.accept()
            with accepted:
                assert engine._pid_owns_tcp_connection(
                    os.getpid(), server_port, client.getsockname()[1]) is True

    def test_connection_owner_wait_is_capped(self, monkeypatch):
        times = iter((10.0, 10.0 + engine._OWNER_WAIT_MAX))
        monkeypatch.setattr(engine.time, "monotonic", lambda: next(times))
        monkeypatch.setattr(engine.time, "sleep", lambda _delay: pytest.fail("slept past cap"))
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: True)
        monkeypatch.setattr(engine, "_pid_owns_tcp_connection", lambda *args: False)

        assert engine._wait_for_tcp_connection_owner(1234, 43210, 54321, 120.0) is False

    def test_connection_owner_wait_fails_immediately_without_inspection(
            self, monkeypatch):
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: False)
        monkeypatch.setattr(
            engine, "_pid_owns_tcp_connection",
            lambda *args: pytest.fail("ownership check should not run"))

        assert engine._wait_for_tcp_connection_owner(1234, 43210, 54321, 120.0) is False

    def test_verified_request_names_missing_socket_inspection(self, monkeypatch):
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: False)
        monkeypatch.setattr(
            engine.http.client, "HTTPConnection",
            lambda *args, **kwargs: pytest.fail("connection should not be opened"))

        with pytest.raises(RuntimeError, match="requires /proc or lsof"):
            engine._verified_tcp_request(43210, 1234, "/v1/models")

    def test_alive_propagates_missing_socket_inspection(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: False)

        with pytest.raises(RuntimeError, match="requires /proc or lsof"):
            engine._alive(43210, expected_pid=1234)

    def test_alive_does_not_send_token_to_wrong_pid(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(engine, "_pid_owns_tcp_port", lambda pid, port: False)
        monkeypatch.setattr(
            engine.http.client, "HTTPConnection",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("network request attempted")))
        assert engine._alive(43210, expected_pid=1234) is False

    def test_alive_authenticates_after_connection_owner_check(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        events = []
        statuses = iter((401, 200))

        class Response:
            def __init__(self):
                self.status = next(statuses)

            @staticmethod
            def read(_size=-1):
                return b""

        class Socket:
            @staticmethod
            def getsockname():
                return (engine.HOST, 54321)

        class Connection:
            sock = Socket()

            @staticmethod
            def connect():
                events.append("connect")

            @staticmethod
            def request(method, endpoint, body=None, headers=None):
                events.append(("request", method, endpoint, headers.get("Authorization")))

            @staticmethod
            def getresponse():
                return Response()

            @staticmethod
            def close():
                events.append("close")

        monkeypatch.setattr(engine, "_pid_owns_tcp_port", lambda pid, port: True)
        monkeypatch.setattr(
            engine, "_wait_for_tcp_connection_owner",
            lambda pid, server_port, client_port, timeout: events.append("owner") or True)
        monkeypatch.setattr(engine, "_read_token", lambda: "secret-token")
        monkeypatch.setattr(engine.http.client, "HTTPConnection", lambda *a, **k: Connection())
        assert engine._alive(43210, expected_pid=1234) is True
        assert events[0:2] == [
            "connect",
            "owner",
        ]
        wrong_auth = events[2][3]
        assert events[2][0:3] == ("request", "GET", "/props")
        assert wrong_auth.startswith("Bearer ")
        assert "secret-token" not in wrong_auth
        assert events[3:] == [
            "close",
            "connect",
            "owner",
            ("request", "GET", "/props", "Bearer secret-token"),
            "close",
        ]

    def test_alive_authenticates_unix_socket(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        socket_path = tmp_path / "data" / "run" / "server.sock"
        socket_path.parent.mkdir(parents=True)
        socket_path.touch()
        monkeypatch.setattr(engine, "_read_token", lambda: "secret-token")
        events = []
        statuses = iter((401, 200))

        class Response:
            def __init__(self):
                self.status = next(statuses)

            @staticmethod
            def read(_size=-1):
                return b""

        class Connection:
            @staticmethod
            def request(method, endpoint, body=None, headers=None):
                events.append((method, endpoint, headers.get("Authorization")))

            @staticmethod
            def getresponse():
                return Response()

            @staticmethod
            def close():
                events.append("close")

        monkeypatch.setattr(engine, "_UnixHTTPConnection", lambda *a, **k: Connection())

        assert engine._alive() is True
        wrong_auth = events[0][2]
        assert events[0][0:2] == ("GET", "/props")
        assert wrong_auth.startswith("Bearer ")
        assert "secret-token" not in wrong_auth
        assert events[1:] == [
            "close",
            ("GET", "/props", "Bearer secret-token"),
            "close",
        ]

    def test_alive_rejects_endpoint_that_does_not_enforce_authentication(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: True)
        monkeypatch.setattr(engine, "_pid_owns_tcp_port", lambda pid, port: True)
        monkeypatch.setattr(engine, "_read_token", lambda: "secret-token")
        calls = []

        def fake_probe(endpoint, token, port, expected_pid, timeout):
            calls.append((endpoint, token))
            return 200

        monkeypatch.setattr(engine, "_probe_status", fake_probe)

        assert engine._alive(43210, expected_pid=1234) is False
        assert len(calls) == 1
        assert calls[0][0] == "/props"
        assert "secret-token" not in calls[0][1]

    def test_alive_rejects_invalid_configured_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: True)
        monkeypatch.setattr(engine, "_pid_owns_tcp_port", lambda pid, port: True)
        monkeypatch.setattr(engine, "_read_token", lambda: "secret-token")
        statuses = iter((401, 401))
        monkeypatch.setattr(engine, "_probe_status", lambda *args: next(statuses))

        assert engine._alive(43210, expected_pid=1234) is False

    def test_alive_never_sends_token_when_connection_owner_differs(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))

        class Socket:
            @staticmethod
            def getsockname():
                return (engine.HOST, 54321)

        class Connection:
            sock = Socket()

            @staticmethod
            def connect():
                pass

            @staticmethod
            def request(*args, **kwargs):
                raise AssertionError("token sent to unverified connection")

            @staticmethod
            def close():
                pass

        monkeypatch.setattr(engine, "_pid_owns_tcp_port", lambda pid, port: True)
        monkeypatch.setattr(engine, "_wait_for_tcp_connection_owner", lambda *args: False)
        monkeypatch.setattr(engine, "_read_token", lambda: "secret-token")
        monkeypatch.setattr(engine.http.client, "HTTPConnection", lambda *a, **k: Connection())
        assert engine._alive(43210, expected_pid=1234) is False

    def test_request_verifies_the_prompt_connection_before_sending(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(engine, "_tcp_server_state", lambda: (43210, 1234))
        monkeypatch.setattr(engine, "_read_token", lambda: "secret-token")
        captured = {}

        def fake_request(port, expected_pid, endpoint, data=None, headers=None, timeout=120.0):
            captured.update(port=port, expected_pid=expected_pid, endpoint=endpoint,
                            data=data, headers=headers, timeout=timeout)
            return 200, b'{"choices": []}'

        monkeypatch.setattr(engine, "_verified_tcp_request", fake_request)
        result = engine._request("/v1/chat/completions", {"messages": ["secret prompt"]})

        assert result == {"choices": []}
        assert captured["port"] == 43210
        assert captured["expected_pid"] == 1234
        assert captured["endpoint"] == "/v1/chat/completions"
        assert b"secret prompt" in captured["data"]
        assert captured["headers"]["Authorization"] == "Bearer secret-token"

    def test_start_server_keeps_token_out_of_argv_and_log(self, monkeypatch, tmp_path):
        model = tmp_path / "model.gguf"
        server = tmp_path / "llama-server"
        model.write_bytes(b"model")
        server.write_bytes(b"binary")
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("WHATISIT_FORCE_TCP", "1")
        monkeypatch.setattr(engine, "running_port", lambda: None)
        monkeypatch.setattr(engine, "_free_port", lambda: 43210)
        monkeypatch.setattr(engine, "_alive", lambda port=None, timeout=0.6, expected_pid=None:
                            expected_pid == 2468)
        captured = {}

        class Process:
            pid = 2468
            returncode = None

            @staticmethod
            def poll():
                return None

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs["env"]
            return Process()

        monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
        assert engine.start_server(model, server, threads=2, wait=1, quiet=True) == 43210

        token = captured["env"]["LLAMA_API_KEY"]
        assert token
        assert token not in captured["cmd"]
        assert "--api-key" not in captured["cmd"]
        assert token not in (tmp_path / "data" / "run" / "server.log").read_text()

    def test_start_server_rejects_forced_tcp_without_socket_inspection(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("WHATISIT_FORCE_TCP", "1")
        monkeypatch.setattr(engine, "running_port", lambda: None)
        monkeypatch.setattr(engine, "_can_inspect_sockets", lambda: False)
        monkeypatch.setattr(
            engine.subprocess, "Popen",
            lambda *args, **kwargs: pytest.fail("server should not be launched"))

        with pytest.raises(RuntimeError, match="requires /proc or lsof"):
            engine.start_server(
                tmp_path / "model.gguf", tmp_path / "llama-server",
                threads=2, wait=180, quiet=True)
        assert not (tmp_path / "data" / "run" / "server.token").exists()

    def test_start_server_reuses_authenticated_unix_socket(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        sock = tmp_path / "data" / "run" / "server.sock"
        sock.parent.mkdir(parents=True)
        sock.touch()
        monkeypatch.setattr(engine, "_alive", lambda *a, **k: True)
        monkeypatch.setattr(
            engine.subprocess, "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("server restarted")))
        assert engine.start_server(tmp_path / "model", tmp_path / "server", threads=1) == 0


# ------------------------------------------------------ greedy-first ordering

class TestQueryServerGreedyOrdering:
    def test_greedy_call_is_always_first_and_uses_temperature_zero(self, monkeypatch):
        calls = []

        def fake_post(port, body):
            calls.append(body)
            if body.get("temperature") == 0.0 and "n" not in body:
                return [("greedy answer", "stop")]
            return [("sampled alt 1", "stop"), ("sampled alt 2", "stop")]

        monkeypatch.setattr(engine, "_post", fake_post)
        out = engine._query_server(port=1, prompt="zip up this project",
                                    cfg={"temperature": 0.0, "max_tokens": 64}, n=3)
        assert out[0] == ("greedy answer", "stop")
        # the greedy request (first call made) must be temperature 0
        assert calls[0]["temperature"] == 0.0

    def test_single_candidate_skips_the_sampled_call_entirely(self, monkeypatch):
        calls = []

        def fake_post(port, body):
            calls.append(body)
            return [("greedy answer", "stop")]

        monkeypatch.setattr(engine, "_post", fake_post)
        out = engine._query_server(port=1, prompt="list files",
                                    cfg={"temperature": 0.0}, n=1)
        assert out == [("greedy answer", "stop")]
        assert len(calls) == 1

    def test_sampled_call_failure_does_not_lose_the_greedy_answer(self, monkeypatch):
        def fake_post(port, body):
            if body.get("temperature") == 0.0:
                return [("greedy answer", "stop")]
            raise RuntimeError("server hiccup")

        monkeypatch.setattr(engine, "_post", fake_post)
        out = engine._query_server(port=1, prompt="list files",
                                    cfg={"temperature": 0.0}, n=3)
        assert out == [("greedy answer", "stop")]


# ------------------------------------------------ generate(): finish_reason

class TestGenerateDiscardsTruncatedCandidates:
    """generate() must discard any candidate whose finish_reason == "length":
    a truncated flag-spam loop can carry a destructive flag (e.g. zip -m,
    which deletes the source files) that only appears once the spam runs on
    long enough to hit the token budget.
    """

    def _fake_cfg_and_model(self, monkeypatch, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"fake")
        monkeypatch.setattr(cfg_mod, "find_model", lambda: model)
        monkeypatch.setattr(engine, "hostctx",
                             _FakeHostCtx())
        # Make server_bin resolution succeed without needing a real binary:
        # WHATISIT_LLAMA_SERVER just needs to point at a file that exists.
        srv = tmp_path / "llama-server"
        srv.write_bytes(b"fake")
        monkeypatch.setenv("WHATISIT_LLAMA_SERVER", str(srv))
        monkeypatch.setattr(engine, "start_server", lambda *a, **kw: 12345)

    def test_length_finish_reason_is_dropped(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            return [
                ("zip -r -9 -m -j -0 -1 -1 -1", "length"),   # truncated, has -m: drop
                ("zip -r archive.zip .", "stop"),             # clean: keep
            ]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, elapsed, mode = engine.generate("zip this folder", {}, n=2)
        assert cmds == ["zip -r archive.zip ."]
        assert mode == "server"

    def test_degenerate_candidate_is_also_dropped_even_if_finished(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            return [
                ("zip -r -9 -q -n -j -0 -9 -n -j -0 -9 -n -j -0 -9 -n -j -0 a.zip .", "stop"),
                ("zip -r archive.zip .", "stop"),
            ]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, _ = engine.generate("zip this folder", {}, n=2)
        assert cmds == ["zip -r archive.zip ."]

    def test_duplicate_candidates_are_deduplicated(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            return [("ls -la", "stop"), ("ls -la", "stop")]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, _ = engine.generate("list files", {}, n=2)
        assert cmds == ["ls -la"]

    @pytest.mark.parametrize("kwargs", [{"quiet": True}, {"for_execution": True}])
    def test_execution_paths_suppress_volatile_host_context(
            self, monkeypatch, tmp_path, kwargs):
        self._fake_cfg_and_model(monkeypatch, tmp_path)
        captured = {}

        class CaptureHostCtx:
            @staticmethod
            def build(prompt, enabled=True, cwd=None, include_volatile=True):
                captured["include_volatile"] = include_volatile
                return ("SYSTEM PROMPT", prompt)

            @staticmethod
            def stable_facts():
                return {"pkg": "unknown"}

            @staticmethod
            def postprocess_command(cmd, pkg_mgr):
                return cmd

            @staticmethod
            def grammar_for_pkg(pkg_mgr):
                return None

            @staticmethod
            def is_install_request(prompt):
                return False

        monkeypatch.setattr(engine, "hostctx", CaptureHostCtx())
        monkeypatch.setattr(
            engine, "_query_server",
            lambda port, prompt, cfg, n, system=None, grammar=None: [("ls -la", "stop")])

        engine.generate("list files", {}, **kwargs)
        assert captured["include_volatile"] is False

    def test_no_model_raises_file_not_found(self, monkeypatch):
        monkeypatch.setattr(cfg_mod, "find_model", lambda: None)
        with pytest.raises(FileNotFoundError):
            engine.generate("do something", {})


class TestServerOverridesReachTheBoundary:
    """The --port/--threads/--ctx-size overrides must be observable at the
    engine boundary, not just recorded in the cfg dict handed to generate().

    These sit at the command-construction seam: generate() resolves threads
    and must forward the fixed port + context size to start_server/_query_oneshot.
    """

    def _fake_cfg_and_model(self, monkeypatch, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"fake")
        monkeypatch.setattr(cfg_mod, "find_model", lambda: model)
        monkeypatch.setattr(engine, "hostctx", _FakeHostCtx())
        srv = tmp_path / "llama-server"
        srv.write_bytes(b"fake")
        monkeypatch.setenv("WHATISIT_LLAMA_SERVER", str(srv))

    def test_server_mode_forwards_fixed_port_and_ctx_size(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)
        seen = {}
        monkeypatch.setattr(
            engine, "start_server",
            lambda model, server_bin, threads, wait=180.0, quiet=False,
                   port=None, ctx_size=2048: seen.update(
                       threads=threads, port=port, ctx_size=ctx_size) or 12345)
        monkeypatch.setattr(engine, "_query_server",
                            lambda port, prompt, cfg, n, system=None, grammar=None:
                                [("ls", "stop")])
        cfg = {"server_port": 9100, "ctx_size": 4096, "threads": 2}
        cmds, _, mode = engine.generate("list files", cfg)
        assert mode == "server"
        assert seen["threads"] == 2
        assert seen["port"] == 9100
        assert seen["ctx_size"] == 4096

    def test_oneshot_mode_forwards_ctx_size(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)
        seen = {}
        monkeypatch.setattr(engine, "start_server",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no server")))
        monkeypatch.setattr(cfg_mod, "find_llama_cli", lambda: tmp_path / "llama-cli")
        monkeypatch.setattr(engine, "_query_oneshot",
                            lambda model, cli_bin, prompt, cfg, threads,
                                   system=None, grammar=None, ctx_size=2048: seen.update(
                                       threads=threads, ctx_size=ctx_size) or [("ls", "stop")])
        cfg = {"threads": 1, "ctx_size": 512}
        cmds, _, mode = engine.generate("list files", cfg, force_oneshot=True)
        assert mode == "oneshot"
        assert seen["threads"] == 1
        assert seen["ctx_size"] == 512

    def test_start_server_builds_command_with_fixed_port_and_ctx_size(self, monkeypatch, tmp_path):
        # Command-construction boundary: the fixed port and ctx size must
        # actually reach the llama-server argv, not just the function args.
        model = tmp_path / "model.gguf"
        server_bin = tmp_path / "llama-server"
        model.write_bytes(b"x")
        server_bin.write_bytes(b"x")
        state_dir = tmp_path / "run"
        state_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(cfg_mod, "env", lambda suffix, default=None: None)
        monkeypatch.setattr(cfg_mod, "load_config", lambda: {})
        monkeypatch.setattr(engine, "running_port", lambda: None)
        monkeypatch.setattr(engine, "_state_dir", lambda: state_dir)
        monkeypatch.setattr(engine, "_write_private", lambda *a, **k: None)
        monkeypatch.setattr(engine, "_alive",
                            lambda port=None, timeout=0.6, expected_pid=None: True)
        captured = {}
        class _FakeProc:
            pid = 2468

            def poll(self):
                return None
        def fake_popen(cmd, *a, **k):
            captured["cmd"] = cmd
            return _FakeProc()
        monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
        port = engine.start_server(model, server_bin, 2, wait=0.05, quiet=True,
                                   port=9100, ctx_size=4096)
        assert port == 9100
        cmd = captured["cmd"]
        assert "--port" in cmd
        assert cmd[cmd.index("--port") + 1] == "9100"
        assert "-c" in cmd
        assert cmd[cmd.index("-c") + 1] == "4096"
        assert "-t" in cmd
        assert cmd[cmd.index("-t") + 1] == "2"


class _FakeHostCtx:
    @staticmethod
    def build(prompt, enabled=True, cwd=None, include_volatile=True):
        return ("SYSTEM PROMPT", prompt)
    @staticmethod
    def stable_facts():
        return {"pkg": "unknown"}
    @staticmethod
    def postprocess_command(cmd, pkg_mgr):
        return cmd
    @staticmethod
    def grammar_for_pkg(pkg_mgr):
        return None
    @staticmethod
    def is_install_request(prompt):
        return "install" in prompt


class _FakeResp:
    """Stand-in for urllib's HTTPResponse: just a callable read()."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._exhausted = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def close(self):
        self._exhausted = True

    def read(self, size=-1, *a, **k):
        # One-shot stream, like a real HTTP response body: emit the payload
        # once, then EOF, so _read_capped()'s read loop terminates.
        if self._exhausted:
            return b""
        self._exhausted = True
        return self._payload


class TestNormalizeEndpoint:
    def test_trailing_slash_removed(self):
        assert engine.normalize_endpoint_url("http://h:1/v1/") == "http://h:1/v1"

    def test_full_route_stripped(self):
        assert (engine.normalize_endpoint_url("http://h:1/v1/chat/completions/") ==
                "http://h:1/v1")

    def test_preserves_path_prefix(self):
        assert (engine.normalize_endpoint_url("https://ex.com/llm/v1/") ==
                "https://ex.com/llm/v1")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            engine.normalize_endpoint_url("")

    def test_non_http_rejected(self):
        with pytest.raises(ValueError, match="http"):
            engine.normalize_endpoint_url("ftp://h/")

    def test_embedded_credentials_rejected(self):
        with pytest.raises(ValueError, match="credentials"):
            engine.normalize_endpoint_url("http://u:p@h/v1")

    def test_fragment_rejected(self):
        with pytest.raises(ValueError, match="fragment"):
            engine.normalize_endpoint_url("http://h/v1#frag")

    def test_no_host_rejected(self):
        with pytest.raises(ValueError, match="no host"):
            engine.normalize_endpoint_url("http:///v1")


class TestParseChoices:
    def test_ok(self):
        assert engine._parse_choices(
            {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        ) == [("x", "stop")]

    def test_missing_choices(self):
        with pytest.raises(RuntimeError, match="no choices"):
            engine._parse_choices({})

    def test_non_string_content_skipped(self):
        r = {"choices": [{"message": {"content": None}},
                         {"message": {"content": "ok"}, "finish_reason": "stop"}]}
        assert engine._parse_choices(r) == [("ok", "stop")]

    def test_no_usable_choices(self):
        with pytest.raises(RuntimeError, match="no usable"):
            engine._parse_choices({"choices": [{"message": {"content": None}}]})


class TestRemotePost:
    """_remote_post(): the HTTP layer, with urllib fully mocked."""

    def _fake_opener(self, resp):
        class _Opener:
            def __init__(self, r):
                self.r = r

            def open(self, req, **kw):
                self.req, self.kw = req, kw
                return self.r

        o = _Opener(resp)
        return o

    def test_sends_bearer_and_parses(self, monkeypatch):
        import json
        payload = json.dumps({"choices": [{"message": {"content": "ls -la"},
                                            "finish_reason": "stop"}]}).encode()
        opener = self._fake_opener(_FakeResp(payload))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        out = engine._remote_post("http://h:1/v1/", "m", "secret", {"model": "m"})
        assert out == [("ls -la", "stop")]
        req = opener.req
        assert req.full_url == "http://h:1/v1/chat/completions"
        assert req.get_method() == "POST"
        assert req.get_header("Authorization") == "Bearer secret"
        assert json.loads(req.data)["model"] == "m"

    def test_no_authorization_header_when_keyless(self, monkeypatch):
        opener = self._fake_opener(_FakeResp(b'{"choices":[]}'))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        with pytest.raises(RuntimeError, match="no usable"):
            engine._remote_post("http://h:1/v1", "m", "", {})
        assert opener.req.get_header("Authorization") is None

    def test_http_error_message_no_secret(self, monkeypatch):
        import urllib.error
        err = urllib.error.HTTPError("http://h:1/v1/chat/completions", 401,
                                     "Unauthorized", {}, _FakeResp(b"invalid key"))

        def boom(*a, **k):
            raise err

        class _Bad:
            open = staticmethod(boom)

        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: _Bad())
        with pytest.raises(RuntimeError) as ei:
            engine._remote_post("http://h:1/v1", "m", "sekrit-key", {})
        assert "401" in str(ei.value)
        assert "sekrit-key" not in str(ei.value)

    def test_non_json_response(self, monkeypatch):
        opener = self._fake_opener(_FakeResp(b"<html>proxy error</html>"))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        with pytest.raises(RuntimeError, match="non-JSON"):
            engine._remote_post("http://h:1/v1", "m", "", {})

    def test_no_cross_origin_redirect_handler(self):
        h = engine._NoCrossOriginRedirect()
        req = urllib.request.Request("http://h:1/v1/chat/completions",
                                     data=b"{}", method="POST")
        # Refuses a redirect to a different authority.
        assert h.redirect_request(req, None, 302, "Found", {},
                                  "http://evil:9/steal") is None

    def test_list_remote_models(self, monkeypatch):
        opener = self._fake_opener(_FakeResp(b'{"data":[{"id":"a"},{"id":"b"}]}'))
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
        names = engine.list_remote_models({"base_url": "http://h:1/v1", "api_key": ""})
        assert names == ["a", "b"]
        assert opener.req.full_url == "http://h:1/v1/models"
        assert opener.req.get_method() == "GET"


class TestQueryRemote:
    REMOTE = {"base_url": "http://h/v1", "model": "m", "api_key": "",
              "timeout": 120.0, "max_tokens": 512}

    def test_greedy_then_separate_sampled_calls(self, monkeypatch):
        calls = []

        def fake_post(base, model, key, body, timeout=120.0):
            calls.append((body.get("temperature"), body.get("n")))
            return [(f"cmd{len(calls)}", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        out = engine._query_remote(self.REMOTE, "prompt",
                                   {"temperature": 0.2}, n=3, system="S")
        assert [c for c, _ in out] == ["cmd1", "cmd2", "cmd3"]
        assert calls[0] == (0.2, 1)
        assert calls[1] == calls[2] == (max(0.6, 0.2), 1)  # never uses n>1

    def test_single_candidate_skips_sampling(self, monkeypatch):
        calls = []

        def fake_post(*a, **k):
            calls.append(1)
            return [("cmd", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        engine._query_remote(self.REMOTE, "p", {}, n=1, system="S")
        assert calls == [1]

    def test_sampled_failure_keeps_greedy(self, monkeypatch):
        state = {"i": 0}

        def fake_post(*a, **k):
            state["i"] += 1
            if state["i"] > 1:
                raise RuntimeError("boom")
            return [("cmd", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        out = engine._query_remote(self.REMOTE, "p", {}, n=3, system="S")
        assert out == [("cmd", "stop")]

    def test_passes_model_max_tokens_and_key(self, monkeypatch):
        seen = {}

        def fake_post(base, model, key, body, timeout=120.0):
            seen.update(base=base, model=model, key=key, body=body, to=timeout)
            return [("c", "stop")]

        monkeypatch.setattr(engine, "_remote_post", fake_post)
        engine._query_remote({**self.REMOTE, "api_key": "K", "timeout": 30.0,
                              "max_tokens": 1024}, "p", {}, n=1, system="S")
        assert seen["model"] == "m" and seen["key"] == "K"
        assert seen["body"]["model"] == "m"
        assert seen["body"]["max_tokens"] == 1024
        assert seen["to"] == 30.0


class TestGenerateRemote:
    def _remote(self, **over):
        return {"base_url": "http://h/v1", "model": "m", "api_key": "",
                "timeout": 120.0, "max_tokens": 512, **over}

    def _enable_remote(self, monkeypatch, remote):
        monkeypatch.setattr(cfg_mod, "remote_config", lambda cfg: remote)
        monkeypatch.setattr(engine.hostctx, "build",
                            lambda p, enabled=True, cwd=None, include_volatile=True: ("SYS", p))

    @pytest.mark.parametrize("kwargs", [{"quiet": True}, {"for_execution": True}])
    def test_remote_execution_paths_suppress_volatile_host_context(
            self, monkeypatch, kwargs):
        remote = self._remote()
        monkeypatch.setattr(cfg_mod, "remote_config", lambda cfg: remote)
        captured = {}

        def capture_context(prompt, enabled=True, cwd=None, include_volatile=True):
            captured["include_volatile"] = include_volatile
            return ("SYS", prompt)

        monkeypatch.setattr(engine.hostctx, "build", capture_context)
        monkeypatch.setattr(engine, "_query_remote",
                            lambda *a, **k: [("ls -la", "stop")])

        engine.generate("list files", {}, **kwargs)
        assert captured["include_volatile"] is False

    def test_remote_mode_needs_no_local_model(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        monkeypatch.setattr(cfg_mod, "find_model", lambda: None)
        monkeypatch.setattr(engine, "_query_remote",
                            lambda *a, **k: [("ls -la", "stop")])
        cmds, elapsed, mode = engine.generate("list files", {})
        assert cmds == ["ls -la"]
        assert mode == "remote"

    def test_remote_missing_model_raises(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote(model=None))
        with pytest.raises(RuntimeError, match="no model selected"):
            engine.generate("list files", {})

    def test_remote_oneshot_incompatible(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        with pytest.raises(RuntimeError, match="oneshot"):
            engine.generate("list files", {}, force_oneshot=True)

    def test_remote_drops_length_finish(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        monkeypatch.setattr(engine, "_query_remote",
                            lambda *a, **k: [("ls -la", "length"), ("pwd", "stop")])
        cmds, _, _ = engine.generate("list files", {})
        assert cmds == ["pwd"]

    def test_remote_dedups_candidates(self, monkeypatch):
        self._enable_remote(monkeypatch, self._remote())
        monkeypatch.setattr(engine, "_query_remote",
                            lambda *a, **k: [("ls", "stop"), ("ls", "stop")])
        cmds, _, mode = engine.generate("list files", {}, n=2)
        assert cmds == ["ls"]
        assert mode == "remote"


class TestRemoteWarnings:
    def test_warns_request_leaves_machine(self):
        w = engine.remote_warnings({"base_url": "https://api.example.com/v1",
                                    "api_key": ""})
        assert any("leaves this machine" in x for x in w)

    def test_http_with_key_warns_cleartext(self):
        w = engine.remote_warnings({"base_url": "http://remote.example.com/v1",
                                    "api_key": "k"})
        assert any("unencrypted" in x for x in w)

    def test_no_warning_for_local_loopback(self):
        w = engine.remote_warnings({"base_url": "http://127.0.0.1:8080/v1",
                                    "api_key": ""})
        assert w == []

    def test_https_loopback_does_not_claim_data_leaves(self):
        w = engine.remote_warnings({"base_url": "https://localhost:8443/v1",
                                    "api_key": ""})
        assert w == []

    def test_invalid_url_reported(self):
        w = engine.remote_warnings({"base_url": "not-a-url", "api_key": ""})
        assert any("invalid" in x for x in w)


class TestGenerateGrammarAndPostprocess:
    """generate() must pass the GBNF grammar to the local backends and run each
    extracted command through hostctx.postprocess_command using the host pkg.
    """

    def _fake_cfg_and_model(self, monkeypatch, tmp_path):
        model = tmp_path / "model.gguf"
        model.write_bytes(b"fake")
        monkeypatch.setattr(cfg_mod, "find_model", lambda: model)
        srv = tmp_path / "llama-server"
        srv.write_bytes(b"fake")
        monkeypatch.setenv("WHATISIT_LLAMA_SERVER", str(srv))
        monkeypatch.setattr(engine, "start_server", lambda *a, **kw: 12345)

    def test_passes_grammar_to_query_server(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)

        class FakeHost:
            @staticmethod
            def build(prompt, enabled=True, cwd=None, include_volatile=True):
                return ("SYS", prompt)
            @staticmethod
            def stable_facts():
                return {"pkg": "pacman"}
            @staticmethod
            def grammar_for_pkg(pkg_mgr):
                return "fake-grammar"
            @staticmethod
            def is_install_request(prompt):
                return True
            @staticmethod
            def postprocess_command(cmd, pkg_mgr):
                assert pkg_mgr == "pacman"
                return cmd

        monkeypatch.setattr(engine, "hostctx", FakeHost())
        captured = {}
        def fake_query_server(port, prompt, cfg, n, system=None, grammar=None):
            captured["grammar"] = grammar
            return [("pacman -S htop", "stop")]
        monkeypatch.setattr(engine, "_query_server", fake_query_server)

        cmds, _, mode = engine.generate("install htop", {}, n=1)
        assert captured["grammar"] == "fake-grammar"
        assert mode == "server"

    def test_postprocesses_wrong_distro_syntax(self, monkeypatch, tmp_path):
        self._fake_cfg_and_model(monkeypatch, tmp_path)
        # Drive postprocessing with the real implementation, pinned to pacman.
        monkeypatch.setattr(engine.hostctx, "build",
                            lambda p, enabled=True, cwd=None, include_volatile=True: ("SYS", p))
        monkeypatch.setattr(engine.hostctx, "stable_facts",
                            lambda *a, **k: {"pkg": "pacman"})
        monkeypatch.setattr(engine.hostctx, "grammar_for_pkg",
                            lambda pkg_mgr: None)
        monkeypatch.setattr(engine, "_query_server",
                            lambda *a, **k: [("apt-get install -y htop", "stop")])
        cmds, _, _ = engine.generate("install htop", {}, n=1)
        assert cmds == ["pacman -S htop"]

    def test_oneshot_passes_prompts_and_grammar_via_files(self, monkeypatch, tmp_path):
        """_query_oneshot must not put the prompt/system/grammar on the argv:
        llama-cli's command line is visible to any co-tenant via `ps`."""
        model = tmp_path / "model.gguf"
        model.write_bytes(b"fake")
        cli = tmp_path / "llama-cli"
        cli.write_bytes(b"fake")

        captured = {}
        class _Proc:
            returncode = 0
            stdout = "> install htop\npacman -S htop\n[ Prompt: 12 tokens ]\n"
            stderr = ""
        def fake_run(cmd, **kw):
            captured["cmd"] = list(cmd)
            return _Proc()

        monkeypatch.setattr(engine.subprocess, "run", fake_run)
        engine._query_oneshot(model, cli, "install htop", {}, threads=1,
                              system="SYS", grammar="GBNF")

        cmd = captured["cmd"]
        assert "--file" in cmd and "--system-prompt-file" in cmd and "--grammar-file" in cmd
        # The raw prompt, system prompt and grammar must never appear as argv
        # elements, only as paths to private files.
        assert "install htop" not in cmd
        assert "SYS" not in cmd
        assert "GBNF" not in cmd
        # The files themselves must be 0600 and cleaned up afterwards.
        for flag in ("--file", "--system-prompt-file", "--grammar-file"):
            path = cmd[cmd.index(flag) + 1]
            assert not os.path.exists(path)


# ------------------------------------------------------- stopping the server

class TestStopServerOwnership:
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        sd = tmp_path / "data" / "run"
        sd.mkdir(parents=True)
        return sd

    @pytest.mark.parametrize("gone,record_kept,verdict",
                             [(True, False, False), (False, True, None)])
    def test_stop_server_spares_a_process_that_is_not_ours(
            self, monkeypatch, tmp_path, gone, record_kept, verdict):
        # Never ours, so never signalled. But a pid that is still alive may be
        # ours and merely unattributable, and wiping the record then strands a
        # resident server nothing can find. A dead pid is just a stale record.
        sd = self._isolate(monkeypatch, tmp_path)
        for name in ("server.pid", "server.port", "server.params"):
            (sd / name).write_text("99999\n")
        killed = []
        monkeypatch.setattr(engine, "_is_our_server", lambda pid: False)
        monkeypatch.setattr(engine, "_pid_gone", lambda pid: gone)
        monkeypatch.setattr(engine.os, "kill",
                            lambda pid, sig: killed.append(sig))
        # None, not False: a live pid we could not attribute is not the same
        # answer as nothing to stop, and cmd_stop has to say so.
        assert engine.stop_server() is verdict
        assert killed == []
        assert (sd / "server.pid").exists() is record_kept
        assert (sd / "server.params").exists() is record_kept

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="no SIGKILL on Windows; _terminate returns before it")
    def test_stop_server_escalates_when_sigterm_is_ignored(
            self, monkeypatch, tmp_path):
        # The pid record is unlinked either way, so a survivor would keep the
        # model resident with nothing left able to find it.
        sd = self._isolate(monkeypatch, tmp_path)
        (sd / "server.pid").write_text("99999\n")
        sent = []
        monkeypatch.setattr(engine, "_is_windows", lambda: False)
        monkeypatch.setattr(engine, "_is_our_server", lambda pid: True)
        monkeypatch.setattr(engine, "_pid_gone", lambda pid: False)
        monkeypatch.setattr(engine, "_STOP_WAIT", 0.0)
        monkeypatch.setattr(engine.os, "kill",
                            lambda pid, sig: sent.append(sig))
        assert engine.stop_server() is False
        assert sent == [signal.SIGTERM, signal.SIGKILL]

    def test_terminate_does_not_escalate_on_windows(self, monkeypatch):
        # TerminateProcess is already unconditional, and there is no SIGKILL
        # to fall back to. Faked, so the branch is covered off Windows too.
        sent = []
        monkeypatch.setattr(engine, "_is_windows", lambda: True)
        monkeypatch.setattr(engine.os, "kill", lambda pid, sig: sent.append(sig))
        assert engine._terminate(4242) is True
        assert sent == [signal.SIGTERM]

    @pytest.mark.parametrize("cmdline,expected", [
        ("/opt/l/bin/llama-server --port 8080", True),
        ("tail -f /opt/l/bin/llama-server.log", False),
        ("vim /opt/l/bin/llama-server.cpp", False),
        ("", False),
    ])
    def test_is_our_server_needs_the_binary_as_an_argument(
            self, monkeypatch, tmp_path, cmdline, expected):
        # The bystander that got killed was a tail of the server's own log.
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(engine, "_is_windows", lambda: False)
        monkeypatch.setattr(engine, "_pid_cmdline", lambda pid: cmdline)
        monkeypatch.setattr(engine, "_read_params",
                            lambda: {"server_bin": "/opt/l/bin/llama-server"})
        assert engine._is_our_server(4242) is expected

    def test_params_are_recorded_when_the_server_never_becomes_ready(
            self, monkeypatch, tmp_path):
        # A start that times out still leaves a pid. Without the params beside
        # it, stop_server cannot attribute that pid and falls back to a looser
        # test, which is the state a user is in when they reach for `stop`.
        sd = self._isolate(monkeypatch, tmp_path)
        model, srv = tmp_path / "m.gguf", tmp_path / "llama-server"
        model.write_bytes(b"x")
        srv.write_bytes(b"x")
        monkeypatch.setattr(engine.subprocess, "Popen",
                            lambda *a, **k: SimpleNamespace(pid=4242,
                                                            poll=lambda: None))
        monkeypatch.setattr(engine, "_alive", lambda *a, **k: False)
        with pytest.raises(RuntimeError):
            engine.start_server(model, srv, threads=1, wait=0.0, quiet=True)
        assert (sd / "server.pid").read_text().strip() == "4242"
        assert json.loads((sd / "server.params").read_text())["server_bin"] == str(srv)

    def test_legacy_state_without_params_still_rejects_a_log_path(
            self, monkeypatch, tmp_path):
        # Pre-0.3.1 state dirs have no server_bin. The name still has to be a
        # whole argument, or the original bystander kill comes straight back.
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(engine, "_is_windows", lambda: False)
        monkeypatch.setattr(engine, "_read_params", lambda: None)
        monkeypatch.setattr(engine, "_pid_cmdline",
                            lambda pid: "tail -f /opt/l/llama-server.log")
        assert engine._is_our_server(4242) is False
        monkeypatch.setattr(engine, "_pid_cmdline",
                            lambda pid: "/opt/l/bin/llama-server --port 8080")
        assert engine._is_our_server(4242) is True

    def test_legacy_state_matches_a_windows_image_name(
            self, monkeypatch, tmp_path):
        # tasklist reports `llama-server.exe`, and the loose substring test this
        # replaces accepted it. Without the suffix we would refuse to stop a
        # pre-upgrade Windows server, which is the only way to reach here.
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(engine, "_is_windows", lambda: True)
        monkeypatch.setattr(engine, "_read_params", lambda: None)
        monkeypatch.setattr(engine, "_pid_cmdline",
                            lambda pid: " llama-server.exe   4242 Console  1  1 612 344 K \n")
        assert engine._is_our_server(4242) is True
        monkeypatch.setattr(engine, "_pid_cmdline",
                            lambda pid: " notepad.exe   4242 Console  1  9 000 K \n")
        assert engine._is_our_server(4242) is False

    def test_ps_output_is_not_rewritten(self, monkeypatch, tmp_path):
        # The tasklist CSV normalisation must not reach ps: a comma there is
        # part of an argument, and blanking it can forge a matching token.
        self._isolate(monkeypatch, tmp_path)
        out = "/usr/bin/python3 -c x --data=a,/opt/l/bin/llama-server,b"
        monkeypatch.setattr(engine, "_is_windows", lambda: False)
        monkeypatch.setattr(engine.subprocess, "run",
                            lambda cmd, **k: SimpleNamespace(returncode=0, stdout=out))
        assert engine._pid_cmdline(4242) == out
        monkeypatch.setattr(engine, "_read_params",
                            lambda: {"server_bin": "/opt/l/bin/llama-server"})
        assert engine._is_our_server(4242) is False

    def test_no_sigkill_once_the_pid_stopped_being_ours(
            self, monkeypatch, tmp_path):
        # SIGKILL cannot be caught, and the attribution is up to _STOP_WAIT old
        # by the time it is sent.
        self._isolate(monkeypatch, tmp_path)
        sent = []
        monkeypatch.setattr(engine, "_is_windows", lambda: False)
        monkeypatch.setattr(engine, "_STOP_WAIT", 0.0)
        monkeypatch.setattr(engine, "_pid_gone", lambda pid: False)
        # stop_server already attributed this pid; by the re-check it is gone.
        monkeypatch.setattr(engine, "_is_our_server", lambda pid: False)
        monkeypatch.setattr(engine.os, "kill",
                            lambda pid, sig: sent.append(sig))
        assert engine._terminate(4242) is False
        assert sent == [signal.SIGTERM]

    def test_is_our_server_is_false_when_the_pid_is_gone(
            self, monkeypatch, tmp_path):
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setattr(engine, "_pid_cmdline", lambda pid: None)
        assert engine._is_our_server(4242) is False

    def _child(self, request):
        """A real child process, reaped whatever the test does.

        It reports readiness on stdout: until exec has happened the child's
        command line is still a copy of this process's own.
        """
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import sys, time; sys.stdout.write('x'); sys.stdout.flush();"
             " time.sleep(60)"],
            stdout=subprocess.PIPE)

        def cleanup():
            proc.kill()
            proc.wait()
            proc.stdout.close()

        request.addfinalizer(cleanup)
        assert proc.stdout.read(1) == b"x"
        return proc

    def _recorded_bin(self, proc):
        """The argv[0] the OS actually reports for this child.

        Not sys.executable: a macOS framework python re-execs itself, so ps
        names the framework binary, and tasklist reports an image name rather
        than a path. start_server records the binary it launched, which is the
        same string this reads back.
        """
        cmdline = engine._pid_cmdline(proc.pid)
        assert cmdline, "the child should still be running"
        return cmdline.split()[0]

    # --- real processes: the only way the Windows branch gets exercised, and
    # CI runs windows-latest on 3.9 and 3.13.

    def test_is_our_server_recognises_a_real_child(
            self, monkeypatch, tmp_path, request):
        self._isolate(monkeypatch, tmp_path)
        proc = self._child(request)
        binary = self._recorded_bin(proc)
        monkeypatch.setattr(engine, "_read_params",
                            lambda: {"server_bin": binary})
        assert engine._is_our_server(proc.pid) is True

    def test_is_our_server_rejects_a_real_unrelated_process(
            self, monkeypatch, tmp_path, request):
        self._isolate(monkeypatch, tmp_path)
        proc = self._child(request)
        monkeypatch.setattr(engine, "_read_params",
                            lambda: {"server_bin": "/nowhere/llama-server"})
        assert engine._is_our_server(proc.pid) is False

    def test_is_our_server_is_false_for_a_pid_that_is_gone(
            self, monkeypatch, tmp_path, request):
        self._isolate(monkeypatch, tmp_path)
        proc = self._child(request)
        pid = proc.pid
        proc.kill()
        proc.wait()
        monkeypatch.setattr(engine, "_read_params",
                            lambda: {"server_bin": sys.executable})
        assert engine._is_our_server(pid) is False

    def test_terminate_actually_kills_a_real_process(
            self, monkeypatch, tmp_path, request):
        self._isolate(monkeypatch, tmp_path)
        proc = self._child(request)
        began = time.time()
        assert engine._terminate(proc.pid) is True
        # A process that takes SIGTERM must not cost the escalation window.
        assert time.time() - began < 0.5
        assert proc.wait(timeout=10) is not None

    def test_stop_server_kills_a_real_child_end_to_end(
            self, monkeypatch, tmp_path, request):
        sd = self._isolate(monkeypatch, tmp_path)
        proc = self._child(request)
        (sd / "server.pid").write_text(f"{proc.pid}\n")
        (sd / "server.params").write_text(
            json.dumps({"server_bin": self._recorded_bin(proc)}))
        assert engine.stop_server() is True
        assert proc.wait(timeout=10) is not None
        assert not (sd / "server.pid").exists()

    def test_windows_ownership_reads_tasklist_output(self, monkeypatch, tmp_path):
        # Pinned against a recorded sample so the CSV shape is checked on every
        # platform, not only when CI happens to be on Windows.
        self._isolate(monkeypatch, tmp_path)
        sample = '"llama-server.exe","4242","Console","1","1,612,344 K"\n'
        argv = []
        monkeypatch.setattr(engine, "_is_windows", lambda: True)
        monkeypatch.setattr(engine.subprocess, "run",
                            lambda cmd, **k: (argv.append(cmd),
                                              SimpleNamespace(returncode=0,
                                                              stdout=sample))[1])
        monkeypatch.setattr(engine, "_read_params",
                            lambda: {"server_bin": "C:\\llama\\llama-server.exe"})
        assert engine._is_our_server(4242) is True
        assert argv[0][0] == "tasklist" and "PID eq 4242" in argv[0]
        monkeypatch.setattr(engine, "_read_params",
                            lambda: {"server_bin": "C:\\llama\\other.exe"})
        assert engine._is_our_server(4242) is False

    def test_pid_gone_never_signals_on_windows(self, monkeypatch):
        # os.kill(pid, 0) calls TerminateProcess there, so it must not run.
        monkeypatch.setattr(engine, "_is_windows", lambda: True)
        monkeypatch.setattr(engine.os, "kill",
                            lambda *a: pytest.fail("os.kill reached on Windows"))
        assert engine._pid_gone(4242) is False
