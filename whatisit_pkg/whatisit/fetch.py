"""Downloading the llama.cpp runtime and the model, using only the stdlib.

The package declares zero runtime dependencies and keeps it that way, so this
talks HTTP with urllib and unpacks with tarfile rather than pulling in
`requests` or `huggingface_hub`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

# Pinned, not tracked as "latest": a new upstream build can change behaviour
# without warning, and this one is what the tests and the README describe.
LLAMA_BUILD = "b10333"
LLAMA_REPO = "ggml-org/llama.cpp"

# The prebuilt Linux binaries are built against glibc 2.34. Below that they
# download fine and then die with "GLIBC_2.34 not found" on first query, which
# tells the user nothing. Detect it up front instead.
MIN_GLIBC = (2, 34)

# Platform -> the upstream asset suffix. Anything absent here has no prebuilt
# archive we can use.
ASSET_SUFFIX = {
    ("Linux", "x86_64"): "ubuntu-x64",
    ("Linux", "aarch64"): "ubuntu-arm64",
    ("Darwin", "arm64"): "macos-arm64",
    ("Darwin", "x86_64"): "macos-x64",
}

# sha256 and size come from Hugging Face's x-linked-etag / x-linked-size, which
# are the LFS object's own digest, so they can be pinned without downloading.
MODELS = {
    "1.5b": {
        "repo": "ThorOdinson246/nl2sh-1.5b-Q4_K_M",
        "file": "nl2sh-1.5b-Q4_K_M.gguf",
        "sha256": "6f8a17a11129a31074c944f4c2602453fafd9de43bdaeb1630a8f511ec820f71",
        "size": 986_048_000,
    },
    "3b": {
        "repo": "ThorOdinson246/nl2sh-3b-Q4_K_M",
        "file": "nl2sh-3b-Q4_K_M.gguf",
        "sha256": "e36f4e610276e2197f02be7b47bd871f47e29fde8d74b0387c90a5939130d884",
        "size": 1_929_902_464,
    },
}

# Rough extracted size of the runtime archive, for the disk-space check.
RUNTIME_BYTES = 120_000_000

# Download size of each upstream archive, for the "download it?" prompt. These
# are the compressed tarballs, which is what the progress bar counts, so the
# figure offered and the figure shown agree. Tied to LLAMA_BUILD: update both
# together. The compat runtime carries its own size in COMPAT_RUNTIME.
ASSET_BYTES = {
    ("Linux", "x86_64"): 16_507_165,
    ("Linux", "aarch64"): 13_377_770,
    ("Darwin", "arm64"): 11_015_270,
    ("Darwin", "x86_64"): 11_290_712,
}
USER_AGENT = "whatisit-setup"

# Fallback for Linux boxes below MIN_GLIBC, where the upstream archive downloads
# fine and then cannot start. Built from llama.cpp with LLAMA_SUBPROCESS=OFF and
# GGML_NATIVE=OFF plus explicit AVX2/FMA/F16C, so it links no symbol newer than
# GLIBC_2.2.5 -- but it has no runtime CPU dispatch, hence REQUIRES_AVX2.
COMPAT_RUNTIME = {
    ("Linux", "x86_64"): {
        "url": ("https://github.com/ThorOdinson246/whatisit-nl2sh/releases/download/"
                "runtime-el7-v1/whatisit-llama-el7-x64.tar.gz"),
        "sha256": "9006f1326be8f39fa010cc83cecf01e399f6ea7044805255dad6c12341032b70",
        "size": 13_754_582,
    },
}


class FetchError(Exception):
    """Anything that should stop setup with a message rather than a traceback."""


# --------------------------------------------------------------------------
# platform and libc
# --------------------------------------------------------------------------

def platform_key(system: str | None = None, machine: str | None = None) -> tuple:
    system = system or platform.system()
    m = (machine or platform.machine()).lower()
    if m in ("amd64", "x86_64", "x64"):
        m = "x86_64"
    elif m in ("arm64", "aarch64"):
        m = "arm64" if system == "Darwin" else "aarch64"
    return (system, m)


def asset_name(build: str = LLAMA_BUILD, key: tuple | None = None) -> str | None:
    suffix = ASSET_SUFFIX.get(key or platform_key())
    return f"llama-{build}-bin-{suffix}.tar.gz" if suffix else None


def asset_url(build: str = LLAMA_BUILD, key: tuple | None = None) -> str | None:
    name = asset_name(build, key)
    if not name:
        return None
    return f"https://github.com/{LLAMA_REPO}/releases/download/{build}/{name}"


def glibc_version() -> tuple | None:
    """(major, minor), or None when it cannot be determined."""
    try:
        v = os.confstr("CS_GNU_LIBC_VERSION")  # e.g. "glibc 2.34"
        if v:
            m = re.search(r"(\d+)\.(\d+)", v)
            if m:
                return (int(m.group(1)), int(m.group(2)))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        _, ver = platform.libc_ver()
        m = re.match(r"(\d+)\.(\d+)", ver or "")
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except OSError:
        pass
    return None


def is_musl() -> bool:
    """True on Alpine and other musl systems, where these builds will not run."""
    try:
        if platform.libc_ver()[0] == "musl":
            return True
    except OSError:
        pass
    # libc_ver() commonly reports ("", "") on musl, so also look for the loader.
    return any(Path(p).exists() for p in
               ("/lib/ld-musl-x86_64.so.1", "/lib/ld-musl-aarch64.so.1"))


def runtime_plan(key: tuple | None = None) -> dict:
    """Decide which runtime, if any, this machine can actually run.

    Returns {"kind": "upstream"|"compat"|"none", "url", "sha256", "size",
             "reason", "warn"}. Deciding this before downloading is the whole
    point: the failure it avoids is a successful download of a binary that
    cannot start.
    """
    key = key or platform_key()
    out = {"kind": "none", "url": None, "sha256": None, "size": None,
           "reason": "", "warn": ""}

    if key not in ASSET_SUFFIX:
        out["reason"] = f"no prebuilt llama.cpp archive for {key[0]}/{key[1]}"
        return out

    def upstream():
        # size is the DOWNLOAD size, matching what the progress bar counts and
        # what the compat branch reports. RUNTIME_BYTES is the extracted size
        # and is used only for the disk-space check.
        out.update(kind="upstream", url=asset_url(key=key),
                   size=ASSET_BYTES.get(key, RUNTIME_BYTES))
        return out

    if key[0] != "Linux":
        return upstream()

    if is_musl():
        out["reason"] = "musl libc (Alpine): these binaries are glibc-only"
        return out

    v = glibc_version()
    if v is None:
        out["warn"] = "could not determine the glibc version; assuming it is recent enough"
        return upstream()
    if v >= MIN_GLIBC:
        return upstream()

    spec = compat_runtime(key)
    reason = (f"glibc {v[0]}.{v[1]} is older than the "
              f"{MIN_GLIBC[0]}.{MIN_GLIBC[1]} the upstream binaries need")
    if not spec:
        out["reason"] = reason
        return out
    if not has_avx2():
        out["reason"] = reason + ", and the compatibility build requires AVX2"
        return out
    out.update(kind="compat", url=spec["url"], sha256=spec["sha256"],
               size=spec["size"], reason=reason)
    return out


def has_avx2() -> bool:
    """Only consulted for COMPAT_RUNTIME, which bakes in AVX2 with no dispatch.

    The upstream archive ships per-microarchitecture libggml-cpu-*.so and picks
    one at load time, so it needs no check of any kind.
    """
    try:
        with open("/proc/cpuinfo") as f:
            return " avx2 " in f.read().replace("\n", " ")
    except OSError:
        return True  # unknown: do not block on a guess


def compat_runtime(key: tuple | None = None) -> dict | None:
    """A runtime that works below MIN_GLIBC, if one exists for this machine."""
    return COMPAT_RUNTIME.get(key or platform_key())


def existing_llama_server() -> Path | None:
    w = shutil.which("llama-server")
    return Path(w) if w else None


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(str(p)).free


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

def _open(url: str, timeout: float = 60.0, method: str = "GET"):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def release_digests(build: str = LLAMA_BUILD, timeout: float = 30.0) -> dict:
    """Asset name -> sha256, from the GitHub release API.

    GitHub publishes a digest per asset, so the runtime is verified against
    upstream's own hash rather than one we would have to maintain here.
    """
    url = f"https://api.github.com/repos/{LLAMA_REPO}/releases/tags/{build}"
    try:
        with _open(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise FetchError(f"could not reach the GitHub release API: {e}") from e
    out = {}
    for a in data.get("assets", []):
        d = a.get("digest") or ""
        if d.startswith("sha256:"):
            out[a["name"]] = d.split(":", 1)[1]
    return out


def model_url(size: str) -> str:
    spec = MODELS[size]
    return f"https://huggingface.co/{spec['repo']}/resolve/main/{spec['file']}"


def fmt_size(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str | None = None,
             expected_size: int | None = None, progress=None) -> Path:
    """Download to `dest`, verifying before it appears at that path.

    Writes to `dest.part` and renames only after the hash matches, so an
    interrupted or corrupted run can never leave something at `dest` that
    later fails with a confusing error from llama.cpp.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    got = 0
    try:
        with _open(url) as r:
            total = expected_size or int(r.headers.get("content-length") or 0)
            with open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    got += len(chunk)
                    if progress:
                        progress(got, total)
    except (urllib.error.URLError, OSError) as e:
        part.unlink(missing_ok=True)
        raise FetchError(f"download failed: {e}") from e
    except BaseException:
        # Ctrl-C included: never leave a partial file behind.
        part.unlink(missing_ok=True)
        raise

    if sha256 and h.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"checksum mismatch for {dest.name}\n"
            f"  expected {sha256}\n  got      {h.hexdigest()}\n"
            "  the file was deleted; nothing was installed")
    part.replace(dest)
    return dest


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def _safe_members(tf: tarfile.TarFile, root: Path):
    """Reject anything that would write outside `root`.

    Python 3.12 has `filter="data"` for this, but the package supports 3.9.

    Symlinks are kept, not skipped: the shared libraries ship as a chain of
    them (libllama-common.so -> .so.0 -> .so.0.17.0) and the loader resolves
    the SONAME through that chain. Dropping them extracts files that look
    complete and then fail with "cannot open shared object file".
    """
    root = root.resolve()
    prefix = str(root) + os.sep
    for m in tf.getmembers():
        where = os.path.normpath(os.path.join(str(root), m.name))
        if not (where + os.sep).startswith(prefix):
            raise FetchError(f"refusing to extract {m.name!r}: escapes the target directory")
        if m.issym() or m.islnk():
            # POSIX archives use leading '/' for absolute targets; on Windows
            # os.path.isabs('/etc/shadow') is False, so also catch / and \.
            ln = m.linkname
            if os.path.isabs(ln) or ln.startswith(("/", "\\")):
                raise FetchError(f"refusing to extract {m.name!r}: absolute link target")
            base = os.path.dirname(where) if m.issym() else str(root)
            dest = os.path.normpath(os.path.join(base, ln))
            if not (dest + os.sep).startswith(prefix):
                raise FetchError(f"refusing to extract {m.name!r}: link escapes the target")
        yield m


def extract_runtime(archive: Path, dest: Path) -> Path:
    """Unpack the archive and return the directory holding llama-server.

    The upstream tarball puts everything in one versioned directory, e.g.
    `llama-b10333/`, and the binaries are dynamically linked against the .so
    files sitting beside them. The whole directory has to stay together, so
    this returns that directory rather than copying binaries out of it.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = _safe_members(tf, dest)
        try:
            # Available from 3.12 (and backported); the default in 3.14.
            tf.extractall(dest, members=members, filter="data")
        except TypeError:
            tf.extractall(dest, members=_safe_members(tf, dest))
    for cand in sorted(dest.rglob("llama-server")):
        if cand.is_file():
            return cand.parent
    raise FetchError(f"llama-server not found in {archive.name}")


def manual_instructions(key: tuple | None = None, build: str = LLAMA_BUILD) -> str:
    """What to do when we cannot fetch a runtime for this machine."""
    key = key or platform_key()
    name = asset_name(build, key)
    lines = ["  Set it up by hand:"]
    if name:
        lines.append(f"    1. Download {name}")
        lines.append(f"       from https://github.com/{LLAMA_REPO}/releases/tag/{build}")
    else:
        lines.append(f"    1. Build llama.cpp from source for {key[0]}/{key[1]}:")
        lines.append(f"       https://github.com/{LLAMA_REPO}")
    lines.append("    2. Download the model from")
    lines.append(f"       https://huggingface.co/{MODELS['1.5b']['repo']}")
    lines.append("    3. whatisit setup --model ./MODEL.gguf --bin-dir /path/to/llama/bin")
    return "\n".join(lines)


def source_build_instructions() -> str:
    """For glibc older than the prebuilt binaries need.

    These flags are what produced a working build on a glibc 2.17 machine:
    LLAMA_SUBPROCESS=OFF avoids posix_spawn_file_actions_addchdir_np, and
    GGML_NATIVE=OFF with explicit ISA flags keeps the result portable.
    """
    return (
        "  Build llama.cpp from source (works on older glibc):\n"
        f"    git clone https://github.com/{LLAMA_REPO}\n"
        "    cd llama.cpp && cmake -B build \\\n"
        "      -DLLAMA_SUBPROCESS=OFF -DGGML_NATIVE=OFF \\\n"
        "      -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON\n"
        "    cmake --build build --config Release -j\n"
        "    whatisit setup --bin-dir ./build/bin"
    )
