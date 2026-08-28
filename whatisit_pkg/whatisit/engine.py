"""Model execution, in three modes.

remote mode (opt-in)
    An OpenAI-compatible endpoint (any provider, a LAN llama-server, or Ollama)
    answers. Configured via openai_base_url / openai_model; needs no local
    model file. The request text leaves the machine, so it is only active when
    explicitly configured and the CLI warns about it.

server mode (preferred, local)
    A `llama-server` process holds the model in RAM and answers over localhost
    HTTP. Started lazily on first query and left running.

oneshot mode (fallback, local)
    One `llama-cli` invocation per query.

Why the server exists at all: measured in the target udocker environment, a
one-shot query took 5.5 s wall of which only ~1.2 s was generation -- the rest
was re-reading the 941 MB model. Keeping the model resident is the difference
between "feels instant" and "why is this slower than typing it myself".
"""
from __future__ import annotations

import http.client
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config as cfg_mod
from . import hostctx
from .extract import extract

HOST = "127.0.0.1"
STOP = ["\n", "<|im_end|>", "```"]
_OWNER_WAIT_MAX = 2.0
_TCP_INSPECTION_ERROR = (
    "TCP transport requires /proc or lsof to attribute the server connection; "
    "use the UNIX socket transport instead"
)


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over a UNIX domain socket.

    Why the server is not on a TCP port by default. On a normal multi-user Linux
    box -- which every HPC login and compute node is -- loopback is shared across
    UIDs, so a TCP llama-server is reachable by any co-tenant, and the port
    number sits in a file they can read. Worse, choosing a port with bind(0),
    closing it, then starting the server on that number leaves a window in which
    another local process can claim it first and answer in our place, returning
    an arbitrary "generated command".

    A socket file inside a 0700 directory removes the whole class: there is no
    port to squat and access is governed by filesystem permissions.
    """

    def __init__(self, path: str, timeout: float = 120.0):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def _sock_path() -> Path:
    return _state_dir() / "server.sock"


def _token_path() -> Path:
    return _state_dir() / "server.token"


def _read_token() -> str | None:
    try:
        return _token_path().read_text().strip() or None
    except OSError:
        return None


# A cap on how much of the server's response we will ever buffer. This is our
# own llama-server, started by us with a small `-c 2048` context, so a normal
# reply is at most a few KB -- but nothing upstream bounds `resp.read()`, and
# a compromised, hung, or simply buggy server (or, with WHATISIT_FORCE_TCP=1, a
# co-tenant that won the documented port-squat race) can otherwise stream an
# unbounded response and OOM the client before a single byte is validated.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _read_capped(readable, limit: int = _MAX_RESPONSE_BYTES) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = readable.read(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise RuntimeError(f"server response exceeded {limit} bytes -- refusing to buffer more")
    return b"".join(chunks)


def _request(endpoint: str, body: dict | None = None, timeout: float = 120.0):
    """POST/GET to the running server over its UNIX socket or TCP port."""
    token = _read_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None

    sp = _sock_path()
    if sp.exists():
        conn = _UnixHTTPConnection(str(sp), timeout=timeout)
        try:
            conn.request("POST" if data else "GET", endpoint, body=data, headers=headers)
            resp = conn.getresponse()
            payload = _read_capped(resp)
            if resp.status != 200:
                raise RuntimeError(f"server returned {resp.status}: {payload[:200]!r}")
            return json.loads(payload) if payload else {}
        finally:
            conn.close()

    state = _tcp_server_state()
    if state is None:
        raise RuntimeError("no running whatisit server")
    port, pid = state
    status, payload = _verified_tcp_request(
        port, pid, endpoint, data=data, headers=headers, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"server returned {status}: {payload[:200]!r}")
    return json.loads(payload) if payload else {}


def _runtime_env() -> dict:
    """Prepend any bundled shared libs, the way a manylinux wheel would.

    The prebuilt binaries here are compiled against a newer libstdc++ than
    Ubuntu 22.04 ships (GLIBCXX_3.4.32 vs 3.4.30), so the wheel carries its own.
    """
    env = dict(os.environ)
    bundled = Path(__file__).resolve().parent.parent.parent / "runtime" / "lib"
    if extra := cfg_mod.env("RUNTIME_LIB"):
        bundled = Path(extra)
    if bundled.is_dir():
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bundled}{':' + prev if prev else ''}"
    return env


def _state_dir() -> Path:
    d = cfg_mod.data_dir() / "run"
    d.mkdir(parents=True, exist_ok=True)
    # Owner-only: this dir holds the port the local model server listens on.
    # On a multi-user box loopback is shared across UIDs, so a world-readable
    # port file tells any co-tenant exactly where to find an unauthenticated
    # endpoint.
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _write_private(path: Path, text: str) -> None:
    """Write owner-only, creating with 0600 rather than chmod-ing afterwards.

    O_NOFOLLOW: this writes the token/pid/port files inside a 0700 state dir,
    so a symlink planted there normally requires having already broken into
    that directory -- but WHATISIT_DATA_DIR is user-controlled, and if it is ever
    pointed at a shared or otherwise attacker-writable location, a pre-planted
    symlink here would silently redirect this write (with O_TRUNC!) onto
    whatever it points to. O_NOFOLLOW turns that into a hard ELOOP failure
    instead of a silent overwrite of an arbitrary file.
    """
    # O_NOFOLLOW is POSIX-only; Windows has no equivalent flag on os.open.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if os.name != "nt":
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o600)
    try:
        # The 0o600 above applies ONLY when open() creates the file. Without
        # this, a stale group-readable server.token would stay group-readable
        # across every restart, on exactly the shared-NFS-home clusters this
        # targets.
        os.fchmod(fd, 0o600)
    except (OSError, AttributeError):
        pass
    with os.fdopen(fd, "w") as f:
        f.write(text)


def _port_file() -> Path:
    return _state_dir() / "server.port"


def _params_file() -> Path:
    return _state_dir() / "server.params"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _pid_owns_tcp_port(pid: int, port: int) -> bool:
    """Confirm that ``pid`` owns the listening socket before sending the token.

    TCP fallback cannot hold the port open while llama-server binds it. A local
    process can therefore win that race, but it must never receive our API key
    or be accepted as the model server. Linux exposes socket ownership through
    /proc; macOS and other POSIX hosts use lsof when it is available.
    """
    proc_fd = Path(f"/proc/{pid}/fd")
    if proc_fd.is_dir():
        try:
            inodes = set()
            for fd in proc_fd.iterdir():
                try:
                    target = os.readlink(str(fd))
                except OSError:  # an fd can close while /proc is being walked
                    continue
                if target.startswith("socket:[") and target.endswith("]"):
                    inodes.add(target[8:-1])
            for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
                try:
                    lines = table.read_text().splitlines()[1:]
                except OSError:
                    continue
                for line in lines:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                        continue
                    if int(fields[1].rsplit(":", 1)[1], 16) == port:
                        return True
            return False
        except (OSError, ValueError, IndexError):
            pass

    lsof = shutil.which("lsof")
    if not lsof:
        return False
    try:
        p = subprocess.run(
            [lsof, "-nP", "-a", "-p", str(pid), f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True, text=True, timeout=1.0)
        return p.returncode == 0 and f"p{pid}" in p.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _pid_owns_tcp_connection(pid: int, server_port: int, client_port: int) -> bool:
    """Confirm that ``pid`` owns this exact established loopback connection."""
    proc_fd = Path(f"/proc/{pid}/fd")
    if proc_fd.is_dir():
        try:
            inodes = set()
            for fd in proc_fd.iterdir():
                try:
                    target = os.readlink(str(fd))
                except OSError:
                    continue
                if target.startswith("socket:[") and target.endswith("]"):
                    inodes.add(target[8:-1])
            for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
                try:
                    lines = table.read_text().splitlines()[1:]
                except OSError:
                    continue
                for line in lines:
                    fields = line.split()
                    if len(fields) < 10 or fields[3] != "01" or fields[9] not in inodes:
                        continue
                    local_port = int(fields[1].rsplit(":", 1)[1], 16)
                    remote_port = int(fields[2].rsplit(":", 1)[1], 16)
                    if local_port == server_port and remote_port == client_port:
                        return True
            return False
        except (OSError, ValueError, IndexError):
            pass

    lsof = shutil.which("lsof")
    if not lsof:
        return False
    try:
        p = subprocess.run(
            [lsof, "-nP", "-a", "-p", str(pid), f"-iTCP:{server_port}",
             "-sTCP:ESTABLISHED", "-Fn"],
            capture_output=True, text=True, timeout=1.0)
        if p.returncode != 0:
            return False
        for line in p.stdout.splitlines():
            if not line.startswith("n") or "->" not in line:
                continue
            local, remote = line[1:].split("->", 1)
            remote = remote.split(" ", 1)[0]
            if (local.endswith(f":{server_port}")
                    and remote.endswith(f":{client_port}")):
                return True
        return False
    except (OSError, subprocess.TimeoutExpired):
        return False


def _can_inspect_sockets() -> bool:
    """Return whether this host exposes a supported socket ownership API."""
    return Path("/proc/net/tcp").exists() or shutil.which("lsof") is not None


def _wait_for_tcp_connection_owner(pid: int, server_port: int, client_port: int,
                                   timeout: float) -> bool:
    """Wait briefly for the server to accept a just-established connection."""
    if not _can_inspect_sockets():
        return False
    deadline = time.monotonic() + min(max(timeout, 0), _OWNER_WAIT_MAX)
    while True:
        if _pid_owns_tcp_connection(pid, server_port, client_port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.01, max(0, deadline - time.monotonic())))


def _verified_tcp_request(port: int, expected_pid: int, endpoint: str,
                          data: bytes | None = None, headers: dict | None = None,
                          timeout: float = 120.0) -> tuple[int, bytes]:
    """Send one request only after attributing its connection to our server."""
    if not _can_inspect_sockets():
        raise RuntimeError(_TCP_INSPECTION_ERROR)
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        # Connect first without transmitting HTTP headers or a body.  The
        # server-side socket for this exact connection must belong to the saved
        # llama-server PID before credentials or prompt data can leave us.
        conn.connect()
        if conn.sock is None:
            raise RuntimeError("TCP connection has no socket")
        client_port = conn.sock.getsockname()[1]
        if not _wait_for_tcp_connection_owner(expected_pid, port, client_port, timeout):
            raise RuntimeError("TCP connection is not owned by the expected llama-server")
        conn.request("POST" if data else "GET", endpoint, body=data, headers=headers or {})
        response = conn.getresponse()
        return response.status, _read_capped(response)
    finally:
        conn.close()


def _probe_status(endpoint: str, token: str, port: int | None,
                  expected_pid: int | None, timeout: float) -> int | None:
    """Return an authenticated GET's status over the configured transport."""
    headers = {"Authorization": f"Bearer {token}"}
    sp = _sock_path()
    if sp.exists():
        conn = _UnixHTTPConnection(str(sp), timeout=timeout)
        try:
            conn.request("GET", endpoint, headers=headers)
            response = conn.getresponse()
            _read_capped(response)
            return response.status
        finally:
            conn.close()
    if port is None or expected_pid is None:
        return None
    status, _ = _verified_tcp_request(
        port, expected_pid, endpoint, headers=headers, timeout=timeout)
    return status


def _alive(port: int | None = None, timeout: float = 0.6,
           expected_pid: int | None = None) -> bool:
    sp = _state_dir() / "server.sock"
    if not sp.exists():
        if port is None:
            return False
        if not _can_inspect_sockets():
            raise RuntimeError(_TCP_INSPECTION_ERROR)
        if expected_pid is None or not _pid_owns_tcp_port(expected_pid, port):
            return False
    token = _read_token()
    if not token:
        return False
    try:
        # /health and /v1/models are public in llama-server. /props is
        # side-effect-free and protected, so require proof that authentication
        # is both enforced and accepts the configured token.
        wrong_token = secrets.token_urlsafe(24)
        while secrets.compare_digest(wrong_token, token):
            wrong_token = secrets.token_urlsafe(24)
        return (_probe_status("/props", wrong_token, port, expected_pid, timeout) == 401
                and _probe_status("/props", token, port, expected_pid, timeout) == 200)
    except Exception:
        return False


def _tcp_server_state() -> tuple[int, int] | None:
    """Return the recorded TCP port and PID after validating process identity."""
    p = _port_file()
    if not p.exists():
        return None
    try:
        port = int(p.read_text().strip())
        pid = int((_state_dir() / "server.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    return (port, pid) if _is_our_server(pid) else None


def running_port() -> int | None:
    state = _tcp_server_state()
    if state is None:
        return None
    port, pid = state
    try:
        return port if _alive(port, expected_pid=pid) else None
    except RuntimeError:
        # A recorded port we cannot attribute to a pid is a port we will not
        # talk to, which is the same answer as "nothing is running". This is a
        # status query -- `doctor` calls it -- so it must not raise.
        return None


def _is_our_server(pid: int) -> bool:
    """Confirm a pid really is our llama-server before signalling it.

    Without this, `whatisit stop` blindly SIGTERMs whatever now owns a recycled
    pid. On a long-lived shared node that is somebody's -- possibly your own --
    unrelated process.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmdline = raw.replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        # macOS has no /proc. ps is used only to inspect one already-known pid;
        # argument vectors are still passed without a shell.
        try:
            p = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                               capture_output=True, text=True, timeout=1.0)
            if p.returncode != 0:
                return False
            cmdline = p.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
    return "llama-server" in cmdline


def stop_server() -> bool:
    pid_f = _state_dir() / "server.pid"
    stopped = False
    if pid_f.exists():
        try:
            pid = int(pid_f.read_text().strip())
            if _is_our_server(pid):
                os.kill(pid, signal.SIGTERM)
                stopped = True
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pid_f.unlink(missing_ok=True)
    _port_file().unlink(missing_ok=True)
    _sock_path().unlink(missing_ok=True)
    _token_path().unlink(missing_ok=True)
    _params_file().unlink(missing_ok=True)
    return stopped


def _read_params() -> dict | None:
    """The launch parameters of the resident server, if any are recorded."""
    pf = _params_file()
    if not pf.exists():
        return None
    try:
        return json.loads(pf.read_text())
    except (OSError, ValueError):
        return None


def start_server(model: Path, server_bin: Path, threads: int,
                 wait: float = 180.0, quiet: bool = False,
                 port: int | None = None, ctx_size: int = 2048) -> int:
    # The UNIX socket is preferred (see _UnixHTTPConnection), but older
    # llama-server builds read --host as a hostname and fail to bind, so the
    # transport has to be settable. TCP is safe only when its socket owner can
    # be verified before any credential or prompt data is sent.
    use_socket = (cfg_mod.env("FORCE_TCP") != "1"
                  and not cfg_mod.load_config().get("force_tcp"))
    if not use_socket and not _can_inspect_sockets():
        raise RuntimeError(_TCP_INSPECTION_ERROR)

    requested = {
        "model": str(model),
        "server_bin": str(server_bin),
        "threads": threads,
        "ctx_size": ctx_size,
        "port": port,
        "use_socket": use_socket,
    }
    recorded = _read_params()
    # A resident server may only be reused when it was launched with the exact
    # parameters being requested. --port/--threads/--model/--ctx-size would
    # otherwise be silently ignored on every query after the first one. A
    # server with no recorded params is a legacy/unknown launch, which we keep
    # reusing as before.
    if recorded is None or recorded == requested:
        if use_socket and _sock_path().exists() and _alive():
            return 0
        if (existing := running_port()) is not None:
            return existing
    if recorded is not None:
        stop_server()

    sd = _state_dir()
    log = sd / "server.log"

    # A token is generated even for the socket transport: it costs nothing and
    # means a stale/hijacked endpoint cannot silently serve us.
    token = secrets.token_urlsafe(24)
    _write_private(_token_path(), token)

    if use_socket and port is None:
        sp = _sock_path()
        sp.unlink(missing_ok=True)
        port = 0
        cmd = [str(server_bin), "-m", str(model), "--host", str(sp),
               "-t", str(threads), "-c", str(ctx_size), "--no-webui"]
    else:
        if port is None:
            port = _free_port()
        cmd = [str(server_bin), "-m", str(model), "--host", HOST,
               "--port", str(port), "-t", str(threads), "-c", str(ctx_size),
               "--no-webui"]
    # The server log records the launch command line and can contain prompt
    # text; it gets the same owner-only treatment as the pid and token.
    log_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if os.name != "nt":
        log_flags |= os.O_NOFOLLOW
    log_fd = os.open(str(log), log_flags, 0o600)
    try:
        os.fchmod(log_fd, 0o600)
    except (OSError, AttributeError):
        pass
    launch_env = _runtime_env()
    # llama-server documents LLAMA_API_KEY as the environment equivalent of
    # --api-key. Keeping the secret out of argv also keeps it out of ps output
    # and out of the launch command recorded below.
    launch_env["LLAMA_API_KEY"] = token
    with os.fdopen(log_fd, "ab") as lf:
        lf.write(f"\n=== start {time.strftime('%F %T')}: {' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                                env=launch_env, start_new_session=True)
    _write_private(sd / "server.pid", str(proc.pid))
    if port:
        _write_private(_port_file(), str(port))

    if not quiet:
        print("whatisit: loading model into memory (first run only)...",
              file=sys.stderr, end="", flush=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={proc.returncode}; see {log}")
        if _alive(port or None, expected_pid=proc.pid if port else None):
            if not quiet:
                print(" ready.", file=sys.stderr, flush=True)
            _write_private(_params_file(), json.dumps(requested))
            return port
        time.sleep(0.3)
    raise RuntimeError(f"llama-server did not become ready in {wait:.0f}s; see {log}")


def _post(port: int, body: dict) -> list[str]:
    data = _request("/v1/chat/completions", body)
    return [(c["message"]["content"], c.get("finish_reason")) for c in data.get("choices", [])]


def _std_body(system: str, user_msg: str, cfg: dict) -> dict:
    """The pieces any chat-completions body must have.

    Llama-server on our localhost and a remote OpenAI-compatible endpoint
    accept the same core fields; the caller adds the transport/model-specific
    ones (model name, repeat penalties, temperature).
    """
    return {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_msg}],
        "max_tokens": cfg.get("max_tokens", 64),
        "stop": STOP,
    }


def _greedy_then_sample(base: dict, n: int, temperature: float,
                        post) -> list:
    """Greedy answer first, then sampled alternatives. post(body)->list.

    This is the heart of `-n N`: slot 1 is always the greedy answer (callers
    treat it as the pick), then `n-1` sampled, distinct-ish alternatives. A
    failure during sampling must never cost the greedy answer we already have.
    """
    out = post({**base, "temperature": temperature})
    if n <= 1:
        return out
    try:
        out += post({**base, "n": n - 1,
                     "temperature": max(0.6, float(temperature)),
                     "top_p": 0.95})
    except Exception:
        pass  # alternatives are a bonus; never lose the greedy answer over them
    return out


def _parse_choices(payload: object) -> list[tuple[str, object]]:
    """Turn a chat-completions response into [(content, finish_reason)].

    The content is untrusted model output (handled downstream by extract), but
    the *shape* must still be validated: a proxy returning HTML or a partial
    error instead of JSON should be an explicit failure, not a silent TypeError
    deep in generate(). Non-string content (some reasoning models emit a dict)
    is skipped rather than crashing on it.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        raise RuntimeError("endpoint response had no choices")
    out = []
    for c in payload["choices"]:
        if not isinstance(c, dict):
            continue
        msg = c.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            continue
        out.append((content, c.get("finish_reason")))
    if not out:
        raise RuntimeError("endpoint response contained no usable choices")
    return out


class _NoCrossOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect that changes scheme or host.

    A bearer token is attached to the request; following a redirect elsewhere
    (scheme or authority) would send that token to a host that never asked for
    it. Returning None from redirect_request makes urllib surface the redirect
    as an HTTPError, which we turn into a normal error message."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if (old.scheme, old.hostname, old.port) != (new.scheme, new.hostname, new.port):
            return None
        try:
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        except http.client.HTTPException:
            return None


def normalize_endpoint_url(base: str) -> str:
    """Canonicalize a base URL: no trailing slash, no fragments/creds, http(s).

    Accepts either `http://host:port/v1` or the full route
    `http://host:port/v1/chat/completions`; either is reduced to the base to
    which the engine appends `/chat/completions`. Raises ValueError on input a
    remote request must never be built from.
    """
    base = (base or "").strip()
    if not base:
        raise ValueError("endpoint URL is empty")
    if base.startswith(("http://", "https://")) is False:
        raise ValueError("endpoint URL must start with http:// or https://")
    p = urllib.parse.urlsplit(base)
    if not p.hostname:
        raise ValueError("endpoint URL has no host")
    if p.username or p.password:
        raise ValueError("endpoint URL must not embed credentials")
    if p.fragment:
        raise ValueError("endpoint URL must not contain a fragment")
    path = p.path.rstrip("/")
    # Pasted the full route? Drop it so we append once below.
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    return urllib.parse.urlunsplit((p.scheme, p.netloc, path, "", ""))


def _is_loopback(hostname: str | None) -> bool:
    return (hostname or "").lower() in ("127.0.0.1", "::1", "localhost")


def remote_warnings(remote: dict) -> list[str]:
    """Human-facing safety notes for a configured remote endpoint.

    Remote mode deliberately breaks the tool's core promise that nothing leaves
    the machine, so the caller should surface these before the first request.
    """
    try:
        url = normalize_endpoint_url(remote["base_url"])
    except ValueError as e:
        return [f"invalid remote endpoint: {e}"]
    sp = urllib.parse.urlsplit(url)
    loopback = _is_loopback(sp.hostname)
    warns = []
    if not loopback:
        warns.append("remote endpoint: your request text leaves this machine")
    if sp.scheme == "http" and not loopback:
        if remote.get("api_key"):
            warns.append("endpoint is plain HTTP; the API key and request travel unencrypted")
        else:
            warns.append("endpoint is plain HTTP; the request travels unencrypted")
    return warns


def _remote_post(base: str, model: str, api_key: str, body: dict,
                 timeout: float = 120.0) -> list[tuple[str, object]]:
    """POST a chat-completion to a remote OpenAI-compatible endpoint.

    Bearer auth only when a key is given (a LAN llama-server needs none). The
    response is buffered under the same _MAX_RESPONSE_BYTES cap as the local
    server, and errors never include the API key.
    """
    url = normalize_endpoint_url(base) + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoCrossOriginRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            payload = _read_capped(r)
    except urllib.error.HTTPError as e:
        detail = _read_capped(e).decode("utf-8", "replace").strip()
        # Error bodies are untrusted and may echo our request; keep it tiny.
        if len(detail) > 200:
            detail = detail[:200] + "..."
        raise RuntimeError(f"endpoint returned HTTP {e.code}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"endpoint unreachable: {e.reason}") from None
    try:
        return _parse_choices(json.loads(payload))
    except json.JSONDecodeError:
        raise RuntimeError("endpoint returned non-JSON response") from None


def list_remote_models(remote: dict, timeout: float = 5.0) -> list[str]:
    """GET {base}/models. Used by doctor only; generation never guesses a model."""
    url = normalize_endpoint_url(remote["base_url"]) + "/models"
    headers = {}
    if remote.get("api_key"):
        headers["Authorization"] = f"Bearer {remote['api_key']}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_NoCrossOriginRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            payload = _read_capped(r)
    except urllib.error.HTTPError as e:
        detail = _read_capped(e).decode("utf-8", "replace").strip()
        if len(detail) > 200:
            detail = detail[:200] + "..."
        raise RuntimeError(f"endpoint returned HTTP {e.code}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"endpoint unreachable: {e.reason}") from None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise RuntimeError("endpoint /models returned non-JSON") from None
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("endpoint /models had no data list")
    out = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out.append(item["id"])
    return out


def _query_server(port: int, prompt: str, cfg: dict, n: int,
                  system: str | None = None,
                  grammar: str | None = None) -> list[str]:
    """Greedy answer first, then sampled alternatives (see _greedy_then_sample).

    The llama-specific penalties stay on the server body only -- they are what
    the local model needs to avoid flag-spam loops, and remote endpoints do not
    all accept them. An optional GBNF ``grammar`` constrains the model to the
    host's package-manager syntax when the request is an install.
    """
    base = _std_body(system or cfg_mod.SYSTEM_PROMPT, prompt, cfg)
    base["repeat_penalty"] = cfg.get("repeat_penalty", 1.08)
    base["repeat_last_n"] = 64
    if grammar:
        base["grammar"] = grammar
    return _greedy_then_sample(base, n, float(cfg.get("temperature", 0.0)),
                               lambda b: _post(port, b))


def _query_remote(remote: dict, prompt: str, cfg: dict, n: int,
                  system: str | None = None) -> list[tuple[str, object]]:
    """Query an OpenAI-compatible endpoint, greedy answer then alternatives.

    Only broadly-supported fields go on the wire: model, messages, max_tokens,
    stop, temperature, n, top_p. Llama-specific knobs (repeat penalties) are
    local-only.

    Alternatives are one temperature-sampled call each with n=1 rather than a
    single call with n=N-1: llama-server commonly runs `--parallel 1`, which
    rejects n > 1 outright (HTTP 400), and per-call sampling still yields
    distinct candidates. A failed alternative is non-fatal -- the greedy answer
    we already have is never lost.
    """
    base = _std_body(system or cfg_mod.SYSTEM_PROMPT, prompt, cfg)
    base["model"] = remote["model"]
    base["max_tokens"] = remote.get("max_tokens", base.get("max_tokens", 512))
    greedy_temp = float(cfg.get("temperature", 0.0))
    post = lambda b: _remote_post(remote["base_url"], remote["model"],
                                  remote["api_key"], b,
                                  timeout=remote.get("timeout", 120.0))
    out = post({**base, "temperature": greedy_temp, "n": 1})
    if n <= 1:
        return out[:1]
    sample_temp = max(0.6, greedy_temp)
    for _ in range(n - 1):
        if len(out) >= n:
            break
        try:
            out += post({**base, "temperature": sample_temp,
                         "n": 1, "top_p": 0.95})
        except Exception:
            pass  # alternatives are a bonus; never lose the greedy answer over them
    return out[:n]


def _query_oneshot(model: Path, cli_bin: Path, prompt: str, cfg: dict, threads: int,
                   system: str | None = None,
                   grammar: str | None = None,
                   ctx_size: int = 2048) -> list[str]:
    """One-shot generation via a single llama-cli invocation.

    The prompt, system prompt and grammar all go through 0o600 temp files, not
    argv: llama-cli's command line is visible to any co-tenant via `ps`, and
    the prompt carries the user's request text plus the host-facts block.
    """
    sys_file = gram_file = prm_file = None
    try:
        prm_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.txt')
        prm_file.write(prompt)
        prm_file.close()

        sys_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.txt')
        sys_file.write(system or cfg_mod.SYSTEM_PROMPT)
        sys_file.close()

        cmd = [str(cli_bin), "-m", str(model), "--file", prm_file.name,
               "-st", "--no-display-prompt", "--no-warmup",
               "--temp", str(cfg.get("temperature", 0.0)),
               "-n", str(cfg.get("max_tokens", 64)), "-t", str(threads),
               "-c", str(ctx_size)]

        cmd.extend(["--system-prompt-file", sys_file.name])

        if grammar:
            gram_file = tempfile.NamedTemporaryFile(
                mode='w', delete=False, suffix='.gbnf')
            gram_file.write(grammar)
            gram_file.close()
            cmd.extend(["--grammar-file", gram_file.name])

        p = subprocess.run(cmd, capture_output=True, text=True,
                           env=_runtime_env(), timeout=300)
    finally:
        if prm_file:
            os.unlink(prm_file.name)
        if sys_file:
            os.unlink(sys_file.name)
        if gram_file:
            os.unlink(gram_file.name)

    if p.returncode != 0:
        raise RuntimeError(f"llama-cli rc={p.returncode}: {p.stderr[-400:]}")
    # one-shot mode gives no finish_reason; treat as unknown
    return [(_strip_cli_chrome(p.stdout), None)]


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_cli_chrome(out: str) -> str:
    """Pull the answer out of llama-cli's interactive stdout.

    llama-cli in single-turn mode still runs its conversation UI, so stdout is
    a banner, a load spinner, an echo of the prompt as `> <prompt>`, the answer,
    then a throughput footer. `--no-display-prompt` does NOT suppress the echo.
    The answer is what lies between the last `> ` line and the footer.
    """
    text = _ANSI_RE.sub("", out.replace("\r", "\n"))
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("> "):
            start = i + 1
    body = []
    for line in lines[start:]:
        if line.startswith("[ Prompt:") or line.startswith("Exiting..."):
            break
        body.append(line)
    return "\n".join(body).strip()


def looks_degenerate(cmd: str, min_repeats: int = 4) -> bool:
    """Detect flag-spam loops that a length stop alone would miss.

    Catches the reproducible `zip` failure even when it happens to terminate:
    `zip -r -9 -q -n -j -0 -9 -n -j -0 -9 -n -j -0 ...`. Keyed on a token
    repeating far more often than any real command repeats an argument.
    """
    toks = cmd.split()
    if len(toks) < 8:
        return False
    from collections import Counter
    counts = Counter(t for t in toks if t.startswith("-"))
    return bool(counts) and counts.most_common(1)[0][1] >= min_repeats


def _collect_commands(raws, host_pkg: str = "unknown") -> list[str]:
    """Extract, drop bad/finished/duplicate candidates (shared by all modes).

    If host_pkg is known, apply distro-aware postprocessing as a backstop.
    """
    cmds, seen = [], set()
    for raw, finish in raws:
        c = extract(raw)
        if not c or c in seen:
            continue
        if host_pkg != "unknown":
            c = hostctx.postprocess_command(c, host_pkg)
        if c in seen:
            continue
        # A generation that stopped because it ran out of budget is not a
        # finished command, and must not be presented as one. The observed case
        # was `zip -r -9 -q -m -j -0 -1 -1 ...`: truncated mid-flag-spam, yet it
        # still carried `-m`, which DELETES the source files. Showing that as an
        # answer is worse than showing nothing.
        if finish == "length" or looks_degenerate(c):
            continue
        seen.add(c)
        cmds.append(c)
    return cmds


def generate(prompt: str, cfg: dict, n: int = 1, force_oneshot: bool = False,
             quiet: bool = False, for_execution: bool = False) -> tuple[list[str], float, str]:
    """Return (commands, elapsed_seconds, mode). Commands are already extracted.

    mode is one of "remote", "server", or "oneshot". Remote mode is selected by
    a configured OpenAI-compatible base URL and needs no local model at all.
    """
    # Distro guidance (opt-in) and host context both need the host package
    # manager: guidance uses it for the GBNF grammar and the rewrite backstop,
    # host context for the full facts block. The grammar is gated on BOTH
    # install-intent and use_grammar (--no-grammar turns only the constraint
    # off), so a GBNF that rejects a valid command form must not disable the
    # rewrite path, and a non-install query is never grammar-constrained at
    # all. Crucially, the grammar and the rewrite do NOT require host_context:
    # the facts block is measured harmful on the shipped model, but grammar +
    # regex cost no tokens and fix a real failure class on their own.
    guidance_on = bool(cfg.get("distro_guidance", False))
    context_on = bool(cfg.get("host_context", True))
    host_pkg = "unknown"
    if guidance_on or context_on:
        try:
            host_pkg = hostctx.stable_facts().get("pkg", "unknown")
        except (OSError, UnicodeDecodeError):
            pass
    grammar = None
    if (guidance_on and cfg.get("use_grammar", True) and host_pkg != "unknown"
            and hostctx.is_install_request(prompt)):
        grammar = hostctx.grammar_for_pkg(host_pkg)

    remote = cfg_mod.remote_config(cfg)
    if remote is not None:
        if not remote["model"]:
            raise RuntimeError(
                "remote endpoint configured but no model selected -- "
                "set openai_model or WHATISIT_OPENAI_MODEL")
        if force_oneshot:
            raise RuntimeError(
                "--oneshot applies to the local backend and cannot be used with a remote endpoint")
        # -e and -q feed a shell. The volatile block carries filenames from the
        # cwd, which the user did not type, so it is withheld from the two
        # flows whose output can run. Ordinary suggestions still get it.
        system, user_msg = hostctx.build(
            prompt, enabled=context_on,
            include_volatile=not (for_execution or quiet), pkg_line=guidance_on)
        t0 = time.time()
        # No `grammar` wire field here: llama-server accepts it, but generic
        # OpenAI-compatible endpoints may reject unknown fields outright. The
        # postprocess rewrite below remains the backstop on this path.
        raws = _query_remote(remote, user_msg, cfg, n, system=system)
        return _collect_commands(raws, host_pkg), time.time() - t0, "remote"

    model = cfg_mod.find_model()
    if model is None:
        raise FileNotFoundError("no model found -- run `whatisit setup`")
    threads = cfg_mod.resolve_threads(cfg)

    server_bin = None
    if not force_oneshot:
        sb = cfg_mod.env("LLAMA_SERVER")
        if sb and Path(sb).exists():
            server_bin = Path(sb)
        elif (c := cfg_mod.data_dir() / "bin" / "llama-server").exists():
            server_bin = c
        elif (c := Path(cfg.get("llama_server", "/nonexistent"))).exists():
            server_bin = c

    # Host context defaults ON when the key is absent (DEFAULTS sets it False);
    # it is prefix-cached, so it costs one-time prefill rather than per-query
    # latency. cfg["host_context"]=false disables it. distro_guidance folds in
    # only the one-line package-manager sentence instead of the full block.
    system, user_msg = hostctx.build(
        prompt, enabled=context_on,
        include_volatile=not (for_execution or quiet), pkg_line=guidance_on)

    t0 = time.time()
    if server_bin is not None:
        port = start_server(model, server_bin, threads, quiet=quiet,
                            port=cfg.get("server_port"),
                            ctx_size=cfg.get("ctx_size", 2048))
        raws = _query_server(port, user_msg, cfg, n, system=system, grammar=grammar)
        mode = "server"
    else:
        cli = cfg_mod.find_llama_cli()
        if cli is None:
            raise FileNotFoundError(
                "neither llama-server nor llama-cli found -- run `whatisit doctor`")
        raws = _query_oneshot(model, cli, user_msg, cfg, threads, system=system,
                              grammar=grammar,
                              ctx_size=cfg.get("ctx_size", 2048))
        mode = "oneshot"

    cmds = _collect_commands(raws, host_pkg)
    return cmds, time.time() - t0, mode
