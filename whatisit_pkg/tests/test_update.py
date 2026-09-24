import hashlib
import json

import pytest

from whatisit import config as cfg_mod
from whatisit import fetch, update

NOW = 1_000_000.0
OLD, NEW = "a" * 64, "b" * 64


@pytest.fixture
def spawned(monkeypatch):
    calls = []
    monkeypatch.setattr(update.engine, "spawn_detached", calls.append)
    return calls


@pytest.fixture
def data(monkeypatch, tmp_path, spawned):
    for v in ("WHATISIT_MODEL", "NL2SH_MODEL", "WHATISIT_OPENAI_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WHATISIT_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(update, "__version__", "0.4.0")
    monkeypatch.setattr(fetch, "MODELS", {
        "1.5b": {"file": cfg_mod.MODEL_NAME, "sha256": NEW},
        "3b": {"file": "nl2sh-3b-Q4_K_M.gguf", "sha256": NEW.upper()},
    })
    d = tmp_path / "data"
    d.mkdir()
    return d


def write_state(d, **state):
    (d / "update.json").write_text(json.dumps(state))


def read_state(d):
    return json.loads((d / "update.json").read_text())


def install_model(d, name=cfg_mod.MODEL_NAME, sha=OLD):
    models = d / "models"
    models.mkdir(exist_ok=True)
    path = models / name
    path.write_bytes(b"gguf")
    if name != cfg_mod.MODEL_NAME:
        (models / cfg_mod.MODEL_NAME).symlink_to(path)
    path = path.resolve()
    state = read_state(d) if (d / "update.json").exists() else {}
    write_state(d, **{**state, "model": {"key": update._stat_key(path), "sha256": sha}})
    return path


class TestVersion:
    @pytest.mark.parametrize("a,b", [("0.10.0", "0.9.0"), ("1.0", "0.99.9"), ("0.4.1", "0.4")])
    def test_newer(self, a, b):
        assert update._version(a) > update._version(b)

    @pytest.mark.parametrize("v", ["0.5.0rc1", "", None, 5, "latest"])
    def test_unparseable_is_never_newer(self, v):
        assert not update._version(v) > update._version("0.4.0")

    def test_trailing_zeros_are_equal(self):
        assert update._version("0.5") == update._version("0.5.0")


class TestNotices:
    def test_first_run_schedules_a_check_and_says_nothing(self, data, spawned):
        assert update.notices({}, now=NOW) == []
        assert spawned == ["whatisit.update"]
        assert read_state(data)["checked"] == NOW

    def test_checked_today_does_not_spawn(self, data, spawned):
        write_state(data, checked=NOW - 60)
        update.notices({}, now=NOW)
        assert spawned == []

    @pytest.mark.parametrize("checked", [NOW - update.DAY, NOW + 3600, "junk"])
    def test_stale_future_or_garbage_check_spawns(self, data, spawned, checked):
        write_state(data, checked=checked)
        update.notices({}, now=NOW)
        assert spawned == ["whatisit.update"]

    def test_newer_release_is_announced_once_a_day(self, data):
        write_state(data, checked=NOW, latest="0.5.0")
        assert update.notices({}, now=NOW) == [
            "whatisit 0.5.0 is out (you have 0.4.0). pip install -U whatisit"]
        assert update.notices({}, now=NOW + 60) == []
        assert len(update.notices({}, now=NOW + update.DAY)) == 1

    @pytest.mark.parametrize("latest", ["0.4.0", "0.3.9", "0.5.0rc1", None])
    def test_no_notice_unless_strictly_newer(self, data, latest):
        write_state(data, checked=NOW, latest=latest)
        assert update.notices({}, now=NOW) == []

    def test_turned_off_means_no_network_and_no_notice(self, data, spawned):
        write_state(data, latest="9.0.0")
        assert update.notices({"update_check": False}, now=NOW) == []
        assert spawned == []

    def test_corrupt_state_file_is_ignored(self, data, spawned):
        (data / "update.json").write_text("[1, 2")
        assert update.notices({}, now=NOW) == []
        assert spawned == ["whatisit.update"]

    @pytest.mark.parametrize("state", [
        {"model": "x"},
        {"model": {"key": None, "sha256": ["x"]}},
        {"latest": "1" * 5000},
        {"latest": ["0.5.0"], "notified": {}},
    ])
    def test_hostile_values_never_raise(self, data, state):
        install_model(data)
        model = read_state(data)["model"]
        if isinstance(state.get("model"), dict):
            state["model"]["key"] = model["key"]
        write_state(data, checked=NOW, **state)
        assert update.notices({}, now=NOW) == []

    def test_unwritable_state_does_not_check_every_query(self, data, spawned):
        (data / "update.json").mkdir()
        update.notices({}, now=NOW)
        update.notices({}, now=NOW + 1)
        assert spawned == []


class TestModelNotice:
    def test_old_model_is_announced(self, data):
        install_model(data)
        write_state(data, **{**read_state(data), "checked": NOW})
        assert update.notices({}, now=NOW) == ["a newer model is out. run: whatisit setup"]

    def test_pinned_model_is_quiet(self, data):
        install_model(data, sha=NEW)
        write_state(data, **{**read_state(data), "checked": NOW})
        assert update.notices({}, now=NOW) == []

    def test_non_default_size_names_its_flag(self, data):
        install_model(data, name="nl2sh-3b-Q4_K_M.gguf")
        write_state(data, **{**read_state(data), "checked": NOW})
        assert update.notices({}, now=NOW) == [
            "a newer model is out. run: whatisit setup --size 3b"]

    def test_model_changed_since_hashing_is_not_judged(self, data):
        path = install_model(data)
        write_state(data, **{**read_state(data), "checked": NOW})
        path.write_bytes(b"a different, longer file")
        assert update.notices({}, now=NOW) == []

    def test_a_model_the_user_picked_is_left_alone(self, data, monkeypatch, tmp_path):
        mine = tmp_path / "community.gguf"
        mine.write_bytes(b"gguf")
        (data / "models").mkdir()
        (data / "models" / cfg_mod.MODEL_NAME).symlink_to(mine)
        write_state(data, checked=NOW,
                    model={"key": update._stat_key(mine.resolve()), "sha256": OLD})
        assert update.notices({}, now=NOW) == []

    def test_a_pinned_name_registered_from_elsewhere_is_left_alone(self, data, tmp_path):
        mine = tmp_path / "mine" / cfg_mod.MODEL_NAME
        mine.parent.mkdir()
        mine.write_bytes(b"my own quant")
        (data / "models").mkdir()
        (data / "models" / cfg_mod.MODEL_NAME).symlink_to(mine)
        assert update.installed_model() is None

    def test_model_env_override_is_left_alone(self, data, monkeypatch):
        install_model(data)
        assert update.installed_model() is not None
        monkeypatch.setenv("WHATISIT_MODEL", str(data / "models" / cfg_mod.MODEL_NAME))
        assert update.installed_model() is None

    def test_remote_endpoint_skips_the_model(self, data):
        install_model(data)
        write_state(data, **{**read_state(data), "checked": NOW})
        assert update.notices({"openai_base_url": "http://127.0.0.1:1/v1"}, now=NOW) == []


class TestModelSha:
    def test_hashes_once_per_file_version(self, data, monkeypatch):
        path = data / "m.gguf"
        path.write_bytes(b"gguf")
        calls = []
        real = fetch.sha256_file
        monkeypatch.setattr(fetch, "sha256_file", lambda p: calls.append(p) or real(p))
        assert update.model_sha(path) == hashlib.sha256(b"gguf").hexdigest()
        assert update.model_sha(path) == hashlib.sha256(b"gguf").hexdigest()
        assert len(calls) == 1

    def test_never_hashes_when_told_not_to(self, data, monkeypatch):
        path = data / "m.gguf"
        path.write_bytes(b"gguf")
        monkeypatch.setattr(fetch, "sha256_file", lambda p: pytest.fail("hashed"))
        assert update.model_sha(path, compute=False) is None

    def test_missing_file(self, data):
        assert update.model_sha(data / "gone.gguf") is None

    def test_remembered_sha_is_trusted_without_hashing(self, data, monkeypatch):
        path = data / "m.gguf"
        path.write_bytes(b"gguf")
        monkeypatch.setattr(fetch, "sha256_file", lambda p: pytest.fail("hashed"))
        update.remember_sha(path, NEW)
        assert update.model_sha(path) == NEW


def fake_pypi(monkeypatch, version, seen=None):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"info": {"version": version}}).encode()

    def urlopen(req, timeout):
        if seen is not None:
            seen.append((req.full_url, timeout))
        return Resp()

    monkeypatch.setattr(update.urllib.request, "urlopen", urlopen)


class TestRefresh:
    def test_records_the_latest_version_and_hashes_the_model(self, data, monkeypatch):
        install_model(data)
        (data / "update.json").unlink()
        seen = []
        fake_pypi(monkeypatch, "0.5.0", seen)
        update.refresh()
        state = read_state(data)
        assert seen == [("https://pypi.org/pypi/whatisit/json", 10)]
        assert state["latest"] == "0.5.0"
        assert state["model"]["sha256"] == hashlib.sha256(b"gguf").hexdigest()

    @pytest.mark.parametrize("version", ["1" * 5000, ["0.5.0"], "0.5.0rc1"])
    def test_does_not_store_a_version_it_cannot_use(self, data, monkeypatch, version):
        write_state(data)
        fake_pypi(monkeypatch, version)
        update.refresh()
        assert "latest" not in read_state(data)

    def test_remote_endpoint_skips_hashing(self, data, monkeypatch):
        install_model(data)
        (data / "update.json").unlink()
        cfg_mod.save_config({"openai_base_url": "http://127.0.0.1:1/v1"})
        fake_pypi(monkeypatch, "0.4.0")
        monkeypatch.setattr(fetch, "sha256_file", lambda p: pytest.fail("hashed"))
        update.refresh()

    def test_offline_keeps_what_it_knew(self, data, monkeypatch):
        write_state(data, latest="0.5.0")

        def offline(req, timeout):
            raise OSError("no route")

        monkeypatch.setattr(update.urllib.request, "urlopen", offline)
        update.refresh()
        assert read_state(data)["latest"] == "0.5.0"
