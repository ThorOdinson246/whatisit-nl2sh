"""Tests for the auto-setup path. Nothing here touches the network.

The failure this feature has to avoid is faulting on someone else's machine, so
the platform, libc and disk-space branches are tested directly rather than
through whatever this test runner happens to be running on.
"""
import hashlib
import io
import sys
import tarfile
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whatisit import cli, fetch  # noqa: E402


class TestPlatformMapping:
    @pytest.mark.parametrize("key,expected", [
        (("Linux", "x86_64"), "llama-b10333-bin-ubuntu-x64.tar.gz"),
        (("Linux", "aarch64"), "llama-b10333-bin-ubuntu-arm64.tar.gz"),
        (("Darwin", "arm64"), "llama-b10333-bin-macos-arm64.tar.gz"),
        (("Darwin", "x86_64"), "llama-b10333-bin-macos-x64.tar.gz"),
    ])
    def test_known_platforms_resolve(self, key, expected):
        assert fetch.asset_name("b10333", key) == expected
        assert fetch.asset_url("b10333", key).endswith(expected)

    @pytest.mark.parametrize("key", [
        ("FreeBSD", "x86_64"), ("Windows", "x86_64"), ("Linux", "s390x"),
    ])
    def test_unsupported_platform_has_no_asset(self, key):
        assert fetch.asset_name("b10333", key) is None
        assert fetch.asset_url("b10333", key) is None

    @pytest.mark.parametrize("raw,expected", [
        ("AMD64", "x86_64"), ("x86_64", "x86_64"), ("aarch64", "aarch64"),
    ])
    def test_machine_aliases_normalize(self, raw, expected):
        assert fetch.platform_key("Linux", raw) == ("Linux", expected)

    def test_darwin_arm_is_named_arm64(self):
        assert fetch.platform_key("Darwin", "aarch64") == ("Darwin", "arm64")

    def test_unsupported_platform_gives_instructions_not_a_traceback(self):
        text = fetch.manual_instructions(("FreeBSD", "x86_64"))
        assert "Build llama.cpp from source" in text
        assert "whatisit setup --model" in text


class TestRuntimePlan:
    def test_modern_glibc_uses_upstream(self, monkeypatch):
        monkeypatch.setattr(fetch, "is_musl", lambda: False)
        monkeypatch.setattr(fetch, "glibc_version", lambda: (2, 39))
        assert fetch.runtime_plan(("Linux", "x86_64"))["kind"] == "upstream"

    def test_old_glibc_does_not_use_upstream(self, monkeypatch):
        """The whole point: never download a binary that cannot start."""
        monkeypatch.setattr(fetch, "is_musl", lambda: False)
        monkeypatch.setattr(fetch, "glibc_version", lambda: (2, 17))
        monkeypatch.setattr(fetch, "has_avx2", lambda: True)
        plan = fetch.runtime_plan(("Linux", "x86_64"))
        assert plan["kind"] == "compat"
        assert "2.17" in plan["reason"]
        assert "ubuntu-x64" not in (plan["url"] or "")

    def test_old_glibc_without_a_compat_build_refuses(self, monkeypatch):
        monkeypatch.setattr(fetch, "is_musl", lambda: False)
        monkeypatch.setattr(fetch, "glibc_version", lambda: (2, 17))
        plan = fetch.runtime_plan(("Linux", "aarch64"))
        assert plan["kind"] == "none"
        assert "2.17" in plan["reason"]

    def test_old_glibc_without_avx2_refuses(self, monkeypatch):
        monkeypatch.setattr(fetch, "is_musl", lambda: False)
        monkeypatch.setattr(fetch, "glibc_version", lambda: (2, 17))
        monkeypatch.setattr(fetch, "has_avx2", lambda: False)
        plan = fetch.runtime_plan(("Linux", "x86_64"))
        assert plan["kind"] == "none"
        assert "AVX2" in plan["reason"]

    def test_musl_is_refused(self, monkeypatch):
        monkeypatch.setattr(fetch, "is_musl", lambda: True)
        plan = fetch.runtime_plan(("Linux", "x86_64"))
        assert plan["kind"] == "none"
        assert "musl" in plan["reason"]

    def test_unknown_glibc_warns_but_proceeds(self, monkeypatch):
        monkeypatch.setattr(fetch, "is_musl", lambda: False)
        monkeypatch.setattr(fetch, "glibc_version", lambda: None)
        plan = fetch.runtime_plan(("Linux", "x86_64"))
        assert plan["kind"] == "upstream"
        assert plan["warn"]

    def test_macos_skips_the_glibc_question(self, monkeypatch):
        monkeypatch.setattr(fetch, "glibc_version", lambda: (2, 17))
        assert fetch.runtime_plan(("Darwin", "arm64"))["kind"] == "upstream"

    def test_unsupported_platform_plan(self):
        plan = fetch.runtime_plan(("Windows", "x86_64"))
        assert plan["kind"] == "none" and "Windows" in plan["reason"]


class _FakeResponse(io.BytesIO):
    def __init__(self, payload, headers=None):
        super().__init__(payload)
        self.headers = headers or {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestDownload:
    def test_verified_download_lands_at_the_destination(self, monkeypatch, tmp_path):
        payload = b"hello model"
        monkeypatch.setattr(fetch, "_open", lambda *a, **k: _FakeResponse(payload))
        dest = tmp_path / "m.gguf"
        fetch.download("https://example/x", dest, sha256=hashlib.sha256(payload).hexdigest())
        assert dest.read_bytes() == payload
        assert not dest.with_name("m.gguf.part").exists()

    def test_checksum_mismatch_deletes_and_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fetch, "_open", lambda *a, **k: _FakeResponse(b"corrupt"))
        dest = tmp_path / "m.gguf"
        with pytest.raises(fetch.FetchError, match="checksum mismatch"):
            fetch.download("https://example/x", dest, sha256="0" * 64)
        assert not dest.exists()
        assert not dest.with_name("m.gguf.part").exists()

    def test_interrupted_download_leaves_nothing_behind(self, monkeypatch, tmp_path):
        class Dying(io.BytesIO):
            headers = {"content-length": "999"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=-1):
                raise OSError("connection reset")

        monkeypatch.setattr(fetch, "_open", lambda *a, **k: Dying())
        dest = tmp_path / "m.gguf"
        with pytest.raises(fetch.FetchError):
            fetch.download("https://example/x", dest, sha256="0" * 64)
        assert not dest.exists()
        assert not dest.with_name("m.gguf.part").exists()

    def test_keyboard_interrupt_leaves_nothing_behind(self, monkeypatch, tmp_path):
        class Interrupted(io.BytesIO):
            headers = {"content-length": "999"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=-1):
                raise KeyboardInterrupt

        monkeypatch.setattr(fetch, "_open", lambda *a, **k: Interrupted())
        dest = tmp_path / "m.gguf"
        with pytest.raises(KeyboardInterrupt):
            fetch.download("https://example/x", dest)
        assert not dest.exists()
        assert not dest.with_name("m.gguf.part").exists()

    def test_no_network_becomes_a_fetch_error(self, monkeypatch, tmp_path):
        def boom(*a, **k):
            raise urllib.error.URLError("Name or service not known")

        monkeypatch.setattr(fetch, "_open", boom)
        with pytest.raises(fetch.FetchError, match="download failed"):
            fetch.download("https://example/x", tmp_path / "m.gguf")

    def test_release_digests_network_failure_is_a_fetch_error(self, monkeypatch):
        def boom(*a, **k):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(fetch, "_open", boom)
        with pytest.raises(fetch.FetchError, match="GitHub release API"):
            fetch.release_digests("b10333")


def _make_tar(tmp_path, top="llama-b10333", extra=None):
    src = tmp_path / "src" / top
    src.mkdir(parents=True)
    (src / "llama-server").write_text("#!/bin/sh\n")
    (src / "libllama.so").write_text("so")
    archive = tmp_path / "r.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname=top)
        if extra:
            info = tarfile.TarInfo(extra)
            info.size = 3
            tf.addfile(info, io.BytesIO(b"bad"))
    return archive


class TestExtract:
    def test_returns_the_directory_holding_llama_server(self, tmp_path):
        archive = _make_tar(tmp_path)
        out = fetch.extract_runtime(archive, tmp_path / "dest")
        assert (out / "llama-server").is_file()
        # The .so files must stay beside the binary; it links against them.
        assert (out / "libllama.so").is_file()

    def test_rejects_path_traversal(self, tmp_path):
        archive = _make_tar(tmp_path, extra="../escaped")
        with pytest.raises(fetch.FetchError, match="escapes"):
            fetch.extract_runtime(archive, tmp_path / "dest")
        assert not (tmp_path / "escaped").exists()

    def test_soname_symlink_chain_survives(self, tmp_path):
        """The .so version links must be extracted, not skipped.

        llama-server links against the SONAME libllama-common.so.0, which is a
        symlink to the real .so.0.17.0. Drop it and the extraction looks fine
        but the binary cannot start.
        """
        src = tmp_path / "src" / "llama-b1"
        src.mkdir(parents=True)
        (src / "llama-server").write_text("#!/bin/sh\n")
        (src / "libllama-common.so.0.17.0").write_text("real")
        archive = tmp_path / "r.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(src, arcname="llama-b1")
            link = tarfile.TarInfo("llama-b1/libllama-common.so.0")
            link.type, link.linkname = tarfile.SYMTYPE, "libllama-common.so.0.17.0"
            tf.addfile(link)

        out = fetch.extract_runtime(archive, tmp_path / "dest")
        soname = out / "libllama-common.so.0"
        assert soname.is_symlink()
        assert soname.resolve().read_text() == "real"

    def test_symlink_escaping_the_directory_is_rejected(self, tmp_path):
        src = tmp_path / "src" / "llama-b1"
        src.mkdir(parents=True)
        (src / "llama-server").write_text("#!/bin/sh\n")
        archive = tmp_path / "r.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(src, arcname="llama-b1")
            evil = tarfile.TarInfo("llama-b1/passwd")
            evil.type, evil.linkname = tarfile.SYMTYPE, "../../../../etc/passwd"
            tf.addfile(evil)
        with pytest.raises(fetch.FetchError, match="link escapes"):
            fetch.extract_runtime(archive, tmp_path / "dest")

    def test_absolute_symlink_is_rejected(self, tmp_path):
        src = tmp_path / "src" / "llama-b1"
        src.mkdir(parents=True)
        (src / "llama-server").write_text("#!/bin/sh\n")
        archive = tmp_path / "r.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(src, arcname="llama-b1")
            evil = tarfile.TarInfo("llama-b1/shadow")
            evil.type, evil.linkname = tarfile.SYMTYPE, "/etc/shadow"
            tf.addfile(evil)
        with pytest.raises(fetch.FetchError, match="absolute link target"):
            fetch.extract_runtime(archive, tmp_path / "dest")

    def test_archive_without_llama_server_is_an_error(self, tmp_path):
        src = tmp_path / "s"
        src.mkdir()
        (src / "readme").write_text("x")
        archive = tmp_path / "r.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(src, arcname="s")
        with pytest.raises(fetch.FetchError, match="not found"):
            fetch.extract_runtime(archive, tmp_path / "dest")


class _Args:
    """Stand-in for the parsed namespace."""

    def __init__(self, **kw):
        defaults = dict(model=None, bin_dir=None, copy=False, auto=False,
                        size="1.5b", llama_version=fetch.LLAMA_BUILD,
                        runtime_only=False, model_only=False, dry_run=False,
                        print_urls=False)
        defaults.update(kw)
        self.__dict__.update(defaults)


@pytest.fixture
def home(monkeypatch, tmp_path):
    for v in ("WHATISIT_CONFIG_DIR", "WHATISIT_DATA_DIR", "WHATISIT_MODEL",
              "NL2SH_CONFIG_DIR", "NL2SH_DATA_DIR", "NL2SH_MODEL",
              "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def _no_downloads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a download was attempted")

    monkeypatch.setattr(fetch, "download", boom)
    monkeypatch.setattr(fetch, "_open", boom)


class TestSetupCommand:
    def test_dry_run_downloads_nothing(self, home, monkeypatch, capsys):
        _no_downloads(monkeypatch)
        monkeypatch.setattr(fetch, "runtime_plan",
                            lambda *a, **k: {"kind": "upstream", "url": "u", "sha256": None,
                                             "size": 1, "reason": "", "warn": ""})
        assert cli.cmd_setup(_Args(dry_run=True), {}) == 0
        out = capsys.readouterr().out
        assert "would fetch" in out and "nothing was changed" in out
        assert not (home / "data" / "models").exists()

    def test_print_urls_downloads_nothing(self, home, monkeypatch, capsys):
        _no_downloads(monkeypatch)
        monkeypatch.setattr(fetch, "runtime_plan",
                            lambda *a, **k: {"kind": "compat", "url": "https://x/c.tar.gz",
                                             "sha256": "abc", "size": 1, "reason": "",
                                             "warn": ""})
        assert cli.cmd_setup(_Args(print_urls=True), {}) == 0
        out = capsys.readouterr().out
        assert "https://x/c.tar.gz" in out
        assert fetch.MODELS["1.5b"]["sha256"] in out
        assert not (home / "data" / "models").exists()

    def test_insufficient_disk_refuses_before_downloading(self, home, monkeypatch, capsys):
        _no_downloads(monkeypatch)
        monkeypatch.setattr(fetch, "runtime_plan",
                            lambda *a, **k: {"kind": "upstream", "url": "u", "sha256": None,
                                             "size": 1, "reason": "", "warn": ""})
        monkeypatch.setattr(fetch, "existing_llama_server", lambda: None)
        monkeypatch.setattr(fetch, "free_bytes", lambda p: 1024)
        rc = cli.cmd_setup(_Args(auto=True), {})
        assert rc == 1
        assert "not enough disk space" in capsys.readouterr().err

    def test_no_runtime_available_explains_and_exits_nonzero(self, home, monkeypatch, capsys):
        _no_downloads(monkeypatch)
        monkeypatch.setattr(fetch, "runtime_plan",
                            lambda *a, **k: {"kind": "none", "url": None, "sha256": None,
                                             "size": 0, "reason": "musl libc", "warn": ""})
        rc = cli.cmd_setup(_Args(auto=True), {})
        assert rc == 1
        err = capsys.readouterr().err
        assert "musl libc" in err and "setup --model" in err

    def test_auto_needs_no_tty(self, home, monkeypatch, capsys):
        """--auto must work in a Dockerfile or CI step with stdin closed."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        calls = []

        def fake_download(url, dest, sha256=None, expected_size=None, progress=None):
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return dest

        monkeypatch.setattr(fetch, "download", fake_download)
        monkeypatch.setattr(fetch, "existing_llama_server", lambda: None)
        monkeypatch.setattr(fetch, "free_bytes", lambda p: 10 ** 12)
        rc = cli.cmd_setup(_Args(auto=True, model_only=True), {})
        assert rc == 0
        assert len(calls) == 1
        assert (home / "data" / "models" / fetch.MODELS["1.5b"]["file"]).exists()

    def test_without_auto_and_without_tty_nothing_is_downloaded(self, home, monkeypatch):
        _no_downloads(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(fetch, "existing_llama_server", lambda: None)
        monkeypatch.setattr(fetch, "free_bytes", lambda p: 10 ** 12)
        assert cli.cmd_setup(_Args(model_only=True), {}) == 1

    def test_second_run_is_a_no_op(self, home, monkeypatch, capsys):
        models = home / "data" / "models"
        models.mkdir(parents=True)
        (models / fetch.MODELS["1.5b"]["file"]).write_bytes(b"already here")
        _no_downloads(monkeypatch)
        rc = cli.cmd_setup(_Args(auto=True, model_only=True), {})
        assert rc == 0
        assert "model present" in capsys.readouterr().out

    def test_existing_llama_server_on_path_is_reused(self, home, monkeypatch, capsys):
        _no_downloads(monkeypatch)
        found = home / "usr" / "bin" / "llama-server"
        found.parent.mkdir(parents=True)
        found.write_text("#!/bin/sh\n")
        (found.parent / "llama-cli").write_text("#!/bin/sh\n")
        monkeypatch.setattr(fetch, "existing_llama_server", lambda: found)
        monkeypatch.setattr(fetch, "runtime_plan",
                            lambda *a, **k: {"kind": "upstream", "url": "u", "sha256": None,
                                             "size": 1, "reason": "", "warn": ""})
        rc = cli.cmd_setup(_Args(auto=True, runtime_only=True), {})
        assert rc == 0
        assert (home / "data" / "bin" / "llama-server").resolve() == found.resolve()
        assert "linked runtime" in capsys.readouterr().out

    def test_compat_runtime_switches_the_transport_to_tcp(self, home, monkeypatch, tmp_path):
        """That build cannot bind a UNIX socket; without this the first query
        fails with 'couldn't bind HTTP server socket'."""
        monkeypatch.setattr(fetch, "runtime_plan",
                            lambda *a, **k: {"kind": "compat", "url": "https://x/c.tar.gz",
                                             "sha256": "abc", "size": 10, "reason": "old glibc",
                                             "warn": ""})
        monkeypatch.setattr(fetch, "existing_llama_server", lambda: None)
        monkeypatch.setattr(fetch, "free_bytes", lambda p: 10 ** 12)
        monkeypatch.setattr(fetch, "download",
                            lambda url, dest, **k: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                    dest.write_bytes(b"x"), dest)[-1])

        unpacked = tmp_path / "unpacked"
        unpacked.mkdir()
        (unpacked / "llama-server").write_text("#!/bin/sh\n")
        monkeypatch.setattr(fetch, "extract_runtime", lambda a, d: unpacked)

        cfg = {}
        assert cli.cmd_setup(_Args(auto=True, runtime_only=True), cfg) == 0
        assert cfg["force_tcp"] is True

    def test_upstream_runtime_leaves_the_transport_alone(self, home, monkeypatch, tmp_path):
        monkeypatch.setattr(fetch, "runtime_plan",
                            lambda *a, **k: {"kind": "upstream", "url": "https://x/u.tar.gz",
                                             "sha256": None, "size": 10, "reason": "", "warn": ""})
        # Upstream path calls asset_url/asset_name against the real host
        # platform; pin them so Windows (and other unsupported hosts) still
        # exercise the force_tcp omission branch.
        monkeypatch.setattr(fetch, "asset_url", lambda *a, **k: "https://x/u.tar.gz")
        monkeypatch.setattr(fetch, "asset_name", lambda *a, **k: "u.tar.gz")
        monkeypatch.setattr(fetch, "existing_llama_server", lambda: None)
        monkeypatch.setattr(fetch, "free_bytes", lambda p: 10 ** 12)
        monkeypatch.setattr(fetch, "release_digests",
                            lambda b, **k: {"u.tar.gz": "abc"})
        monkeypatch.setattr(fetch, "download",
                            lambda url, dest, **k: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                    dest.write_bytes(b"x"), dest)[-1])
        unpacked = tmp_path / "unpacked"
        unpacked.mkdir()
        (unpacked / "llama-server").write_text("#!/bin/sh\n")
        monkeypatch.setattr(fetch, "extract_runtime", lambda a, d: unpacked)

        cfg = {}
        assert cli.cmd_setup(_Args(auto=True, runtime_only=True), cfg) == 0
        assert "force_tcp" not in cfg

    def test_non_default_size_is_reachable_through_the_fixed_slot(self, home, monkeypatch):
        def fake_download(url, dest, sha256=None, expected_size=None, progress=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return dest

        monkeypatch.setattr(fetch, "download", fake_download)
        monkeypatch.setattr(fetch, "free_bytes", lambda p: 10 ** 12)
        assert cli.cmd_setup(_Args(auto=True, model_only=True, size="3b"), {}) == 0
        models = home / "data" / "models"
        from whatisit import config as cfg_mod
        slot = models / cfg_mod.MODEL_NAME
        assert slot.is_symlink()
        assert slot.resolve() == (models / fetch.MODELS["3b"]["file"]).resolve()


class TestManualPathStillWorks:
    def test_explicit_model_is_registered_without_any_download(self, home, monkeypatch, capsys):
        _no_downloads(monkeypatch)
        src = home / "mine.gguf"
        src.write_bytes(b"model bytes")
        rc = cli.cmd_setup(_Args(model=str(src)), {})
        assert rc == 0
        from whatisit import config as cfg_mod
        slot = home / "data" / "models" / cfg_mod.MODEL_NAME
        assert slot.resolve() == src.resolve()
        assert "linked model" in capsys.readouterr().out

    def test_explicit_bin_dir_is_registered(self, home, monkeypatch):
        _no_downloads(monkeypatch)
        b = home / "llama" / "bin"
        b.mkdir(parents=True)
        (b / "llama-server").write_text("#!/bin/sh\n")
        (b / "llama-cli").write_text("#!/bin/sh\n")
        assert cli.cmd_setup(_Args(bin_dir=str(b)), {}) == 0
        assert (home / "data" / "bin" / "llama-server").resolve() == (b / "llama-server").resolve()

    def test_missing_explicit_model_fails_cleanly(self, home, monkeypatch, capsys):
        _no_downloads(monkeypatch)
        rc = cli.cmd_setup(_Args(model=str(home / "nope.gguf")), {})
        assert rc == 1
        assert "no such file" in capsys.readouterr().err

    def test_legacy_env_var_is_still_honoured(self, home, monkeypatch):
        """NL2SH_MODEL with WHATISIT_MODEL unset: the cluster depends on this."""
        from whatisit import config as cfg_mod
        monkeypatch.delenv("WHATISIT_MODEL", raising=False)
        m = home / "legacy.gguf"
        m.write_bytes(b"x")
        monkeypatch.setenv("NL2SH_MODEL", str(m))
        assert cfg_mod.find_model() == m


def test_runtime_size_is_the_download_size_not_the_extracted_size():
    """The prompt offers a size and the progress bar counts one; they must
    agree. RUNTIME_BYTES is the extracted estimate and is only for the disk
    check, so it must not leak into the offered figure."""
    for key in fetch.ASSET_SUFFIX:
        plan = fetch.runtime_plan(key=key)
        assert plan["size"] == fetch.ASSET_BYTES[key]
        assert plan["size"] != fetch.RUNTIME_BYTES
        assert plan["size"] < fetch.RUNTIME_BYTES
