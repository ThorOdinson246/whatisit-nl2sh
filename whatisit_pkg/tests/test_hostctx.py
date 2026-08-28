"""Tests for whatisit.hostctx: distro detection, package manager mapping,
and stable/volatile block formatting.

Every test isolates itself via monkeypatch -- none may read or write the real
~/.config or ~/.local/share.
"""
from __future__ import annotations

import time

import pytest

from whatisit import config as cfg_mod
from whatisit import hostctx


def _mock_os_release(monkeypatch, content: str | None):
    """Replace hostctx.Path so Path('/etc/os-release') returns fake data.

    The fake Path object also implements .name so _probe()'s shell resolution
    does not AttributeError.
    """
    class FakePath:
        def __init__(self, p):
            self._p = p
        def read_text(self, encoding=None, errors=None):
            if self._p == "/etc/os-release":
                if content is None:
                    raise OSError("no such file")
                return content
            raise OSError(f"unexpected path: {self._p}")
        @property
        def name(self):
            return self._p.split("/")[-1]
    monkeypatch.setattr(hostctx, "Path", FakePath)


def _mock_platform(monkeypatch, system="Darwin", release="23.0.0", machine="x86_64"):
    fake = type("P", (), {
        "system": lambda self: system,
        "release": lambda self: release,
        "machine": lambda self: machine,
    })()
    monkeypatch.setattr(hostctx, "platform", fake)


# ---------------------------------------------------------------------------
# _distro_info
# ---------------------------------------------------------------------------

class TestDistroInfo:
    def test_reads_os_release_fields(self, monkeypatch):
        os_release = (
            'NAME="Ubuntu"\n'
            'VERSION="22.04.3 LTS (Jammy Jellyfish)"\n'
            'ID=ubuntu\n'
            'ID_LIKE=debian\n'
            'PRETTY_NAME="Ubuntu 22.04.3 LTS"\n'
            'VERSION_ID="22.04"\n'
        )
        _mock_os_release(monkeypatch, os_release)
        info = hostctx._distro_info()
        assert info["id"] == "ubuntu"
        assert info["name"] == "Ubuntu"
        assert info["version"] == "22.04.3 LTS (Jammy Jellyfish)"
        assert info["version_id"] == "22.04"
        assert info["id_like"] == ["debian"]

    def test_falls_back_to_platform_when_os_release_missing(self, monkeypatch):
        _mock_os_release(monkeypatch, None)
        _mock_platform(monkeypatch)
        info = hostctx._distro_info()
        assert info["id"] == ""
        assert info["name"] == "Darwin"
        assert info["version"] == ""
        assert info["version_id"] == ""
        assert info["id_like"] == []

    def test_pretty_name_is_fallback_for_name(self, monkeypatch):
        os_release = 'PRETTY_NAME="Fedora Linux 39"\nID=fedora\n'
        _mock_os_release(monkeypatch, os_release)
        info = hostctx._distro_info()
        assert info["name"] == "Fedora Linux 39"


# ---------------------------------------------------------------------------
# _distro
# ---------------------------------------------------------------------------

class TestDistro:
    def test_returns_pretty_name(self, monkeypatch):
        os_release = 'PRETTY_NAME="Archcraft"\nID=archcraft\n'
        _mock_os_release(monkeypatch, os_release)
        assert hostctx._distro() == "Archcraft"

    def test_falls_back_to_platform_system(self, monkeypatch):
        _mock_os_release(monkeypatch, None)
        _mock_platform(monkeypatch)
        assert hostctx._distro() == "Darwin"


# ---------------------------------------------------------------------------
# _canonical_pkg
# ---------------------------------------------------------------------------

class TestCanonicalPkg:
    @pytest.mark.parametrize("distro_id,id_like,expected", [
        ("ubuntu", ["debian"], "apt"),
        ("debian", [], "apt"),
        ("fedora", [], "dnf"),
        ("centos", [], "dnf"),
        ("arch", [], "pacman"),
        ("manjaro", [], "pacman"),
        ("archcraft", ["arch"], "pacman"),
        ("alpine", [], "apk"),
        ("void", [], "xbps"),
        ("opensuse-leap", [], "zypper"),
        ("opensuse-tumbleweed", [], "zypper"),
    ])
    def test_maps_distros_to_canonical_pkg(self, distro_id, id_like, expected):
        assert hostctx._canonical_pkg(distro_id, id_like) == expected

    def test_unmapped_linux_distro_returns_unknown(self, monkeypatch):
        _mock_platform(monkeypatch, system="Linux")
        assert hostctx._canonical_pkg("nixos", []) == "unknown"
        assert hostctx._canonical_pkg("", []) == "unknown"

    def test_unmapped_darwin_returns_brew(self, monkeypatch):
        _mock_platform(monkeypatch, system="Darwin")
        assert hostctx._canonical_pkg("nixos", []) == "brew"
        assert hostctx._canonical_pkg("", []) == "brew"

    def test_id_like_fallback(self):
        """archcraft is not in the map, but arch is via ID_LIKE."""
        assert hostctx._canonical_pkg("archcraft", ["arch"]) == "pacman"

    def test_first_match_wins(self):
        """If a distro ID maps directly, ID_LIKE is not consulted."""
        assert hostctx._canonical_pkg("ubuntu", ["debian"]) == "apt"


# ---------------------------------------------------------------------------
# _probe
# ---------------------------------------------------------------------------

class TestProbe:
    def test_includes_distro_version(self, monkeypatch):
        os_release = 'NAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n'
        _mock_os_release(monkeypatch, os_release)
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: None)
        facts = hostctx._probe()
        assert facts["distro"] == "Ubuntu"
        assert facts["distro_version"] == "22.04"
        assert facts["pkg"] == "apt"  # declared by map; no binary found

    def test_falls_back_to_declared_pkg_when_no_binary_found(self, monkeypatch):
        os_release = 'NAME="Archcraft"\nID=archcraft\nID_LIKE=arch\n'
        _mock_os_release(monkeypatch, os_release)
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: None)
        facts = hostctx._probe()
        assert facts["pkg"] == "pacman"  # declared, even though binary not found

    def test_found_binary_wins_over_declared(self, monkeypatch):
        os_release = 'NAME="Ubuntu"\nID=ubuntu\n'
        _mock_os_release(monkeypatch, os_release)
        monkeypatch.setattr(hostctx.shutil, "which",
                            lambda x: "/usr/bin/apt" if x == "apt" else None)
        facts = hostctx._probe()
        assert facts["pkg"] == "apt"

    def test_unknown_pkg_falls_back_to_brew_on_darwin(self, monkeypatch):
        """Unmapped distros default to brew on macOS (the universal answer)."""
        os_release = 'NAME="Unknown"\nID=unknown\n'
        _mock_os_release(monkeypatch, os_release)
        _mock_platform(monkeypatch, system="Darwin")
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: None)
        facts = hostctx._probe()
        assert facts["pkg"] == "brew"

    def test_unmapped_linux_distro_returns_unknown(self, monkeypatch):
        """An unmapped Linux distro must NOT inherit macOS brew guidance."""
        os_release = 'NAME="Unknown"\nID=unknown\n'
        _mock_os_release(monkeypatch, os_release)
        _mock_platform(monkeypatch, system="Linux")
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: None)
        facts = hostctx._probe()
        assert facts["pkg"] == "unknown"

    def test_declared_pkg_wins_over_foreign_binary(self, monkeypatch):
        """A foreign binary on PATH (e.g. apt installed on Arch) must not
        override the distro-declared manager when its own binary is present."""
        os_release = 'NAME="Arch"\nID=arch\n'
        _mock_os_release(monkeypatch, os_release)
        _mock_platform(monkeypatch, system="Linux")
        monkeypatch.setattr(hostctx.shutil, "which",
                            lambda x: "/usr/bin/pacman" if x == "pacman"
                            else ("/usr/bin/apt" if x == "apt" else None))
        facts = hostctx._probe()
        assert facts["pkg"] == "pacman"

    def test_includes_present_and_missing_tools(self, monkeypatch):
        monkeypatch.setattr(hostctx.shutil, "which",
                            lambda x: "/usr/bin/git" if x == "git" else None)
        facts = hostctx._probe()
        assert "git" in facts["present"]
        assert "apt" in facts["missing"]


# ---------------------------------------------------------------------------
# stable_block
# ---------------------------------------------------------------------------

class TestStableBlock:
    def test_includes_version_when_present(self):
        facts = {
            "distro": "Ubuntu",
            "distro_version": "22.04",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": ["git", "apt"],
            "missing": ["brew", "pacman"],
        }
        block = hostctx.stable_block(facts)
        assert "Ubuntu 22.04" in block

    def test_omits_version_when_empty(self):
        facts = {
            "distro": "Fedora",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "dnf",
            "present": ["git"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        # The version should NOT appear as a number after the distro name.
        # "OS: Fedora (x86_64)" is fine; "OS: Fedora 39 (x86_64)" would include one.
        assert "Fedora 39" not in block
        assert "OS: Fedora (x86_64)" in block

    def test_apt_guidance_appears(self):
        facts = {
            "distro": "Ubuntu",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": ["git"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert "apt install" in block
        assert "apt-get" in block

    def test_pacman_guidance_appears(self):
        facts = {
            "distro": "Archcraft",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "zsh",
            "pkg": "pacman",
            "present": ["git", "pacman"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert "pacman -S" in block
        assert "pacman -Syy" in block

    def test_brew_guidance_appears(self):
        facts = {
            "distro": "macOS",
            "distro_version": "",
            "arch": "arm64",
            "shell": "zsh",
            "pkg": "brew",
            "present": ["git", "brew"],
            "missing": ["apt", "pacman"],
        }
        block = hostctx.stable_block(facts)
        assert "brew install" in block
        assert "apt-get" not in block

    def test_netstat_to_ss_steer(self):
        facts = {
            "distro": "Ubuntu",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": ["ss", "git"],
            "missing": ["netstat", "lsof"],
        }
        block = hostctx.stable_block(facts)
        assert "ss -lptn" in block
        assert "netstat is unavailable" in block

    def test_lsof_missing_recommends_ss_or_fuser(self):
        facts = {
            "distro": "Archcraft",
            "distro_version": "",
            "arch": "x86_64",
            "shell": "zsh",
            "pkg": "pacman",
            "present": ["ss", "git"],
            "missing": ["lsof"],
        }
        block = hostctx.stable_block(facts)
        assert "lsof is unavailable" in block
        assert "fuser" in block

    def test_alpine_guidance(self):
        facts = {
            "distro": "Alpine Linux",
            "distro_version": "3.19",
            "arch": "x86_64",
            "shell": "sh",
            "pkg": "apk",
            "present": ["git"],
            "missing": ["apt", "yum"],
        }
        block = hostctx.stable_block(facts)
        assert "apk add" in block
        assert "apt-get" not in block

    def test_dnf_guidance(self):
        facts = {
            "distro": "Fedora Linux",
            "distro_version": "39",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "dnf",
            "present": ["git"],
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert "dnf install" in block
        assert "yum install" in block

    def test_block_is_capped_at_max_chars(self):
        # Feed an artificially long present list to test the cap.
        long_tools = ["tool" + str(i) for i in range(200)]
        facts = {
            "distro": "Test",
            "distro_version": "1.0",
            "arch": "x86_64",
            "shell": "bash",
            "pkg": "apt",
            "present": long_tools,
            "missing": [],
        }
        block = hostctx.stable_block(facts)
        assert len(block) <= hostctx.MAX_STABLE_CHARS

    def test_delegates_to_stable_facts_when_no_facts_given(self, monkeypatch, tmp_path):
        """When called with no facts, stable_block uses the cached probe."""
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts = hostctx.stable_facts(refresh=True)
        block = hostctx.stable_block()
        assert facts["distro"] in block
        assert facts["pkg"] in block


# ---------------------------------------------------------------------------
# stable_facts caching
# ---------------------------------------------------------------------------

class TestStableFactsCache:
    def test_returns_cached_value_within_ttl(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts = hostctx.stable_facts(refresh=True)
        # Should return the same dict (not re-probe).
        cached = hostctx.stable_facts()
        assert cached["generated"] == facts["generated"]
        assert cached["distro"] == facts["distro"]

    def test_refresh_bypasses_cache(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts1 = hostctx.stable_facts(refresh=True)
        # Manually advance the timestamp to simulate time passing.
        cache_path = hostctx._cache_path()
        import json as _json
        raw = _json.loads(cache_path.read_text())
        raw["generated"] = time.time() - hostctx.CACHE_TTL - 1
        cache_path.write_text(_json.dumps(raw))
        facts2 = hostctx.stable_facts(refresh=True)
        assert facts2["generated"] > facts1["generated"]

    def test_corrupt_cache_file_is_rebuilt(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        cache_path = hostctx._cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{not json}")
        facts = hostctx.stable_facts(refresh=True)
        assert "distro" in facts

    def test_missing_cache_is_built(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        facts = hostctx.stable_facts(refresh=True)
        assert facts["generated"] > 0
        assert "distro" in facts
        assert "present" in facts
        assert "missing" in facts


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

class TestBuild:
    def test_returns_plain_prompt_when_disabled(self, monkeypatch):
        system, user = hostctx.build("install numpy", enabled=False)
        assert system == cfg_mod.SYSTEM_PROMPT
        assert user == "install numpy"

    def test_includes_system_prompt_plus_facts_when_enabled(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "data"
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(data_dir))
        system, user = hostctx.build("install numpy", enabled=True)
        assert system.startswith(cfg_mod.SYSTEM_PROMPT)
        assert "<host_environment>" in system
        assert "install numpy" in user

    def test_volatile_block_contains_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "foo.txt").write_text("hello")
        (tmp_path / "bar").mkdir()
        system, user = hostctx.build("list files", enabled=True)
        assert "foo.txt" in user
        assert "bar/" in user


class _GBNFParser:
    """A small self-contained GBNF parser + matcher for the grammar subset
    used by PKG_MGR_GRAMMARS. Lets the tests assert real acceptance/rejection
    of whole command strings instead of grepping the grammar text for
    substrings (which could never see that `other-cmd ::= char+` accepted
    everything)."""

    def __init__(self, grammar: str):
        self.text = grammar
        self.pos = 0
        self.rules: dict[str, tuple] = {}
        self._parse_rules()

    # ---- parsing ----
    def _skip_ws(self):
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c in " \t\r\n":
                self.pos += 1
            elif c == "#":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            else:
                break

    def _parse_rules(self):
        # Each GBNF rule in PKG_MGR_GRAMMARS sits on its own line, so parse
        # rule-by-rule instead of scanning one stream (a stream scanner cannot
        # tell where one rule's expression ends and the next rule's header
        # begins, since both look like bare identifiers).
        for line in self.text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "::=" not in stripped:
                continue
            name, _, body = stripped.partition("::=")
            name = name.strip()
            assert name, (name, stripped)
            self.pos = 0
            self.text = body
            self._skip_ws()
            self.rules[name] = self._alternation()
        assert "root" in self.rules

    def _name(self):
        start = self.pos
        while self.pos < len(self.text) and (
                self.text[self.pos].isalnum() or self.text[self.pos] in "-_"):
            self.pos += 1
        return self.text[start:self.pos]

    def _alternation(self):
        branches = [self._sequence()]
        while True:
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == "|":
                self.pos += 1
                branches.append(self._sequence())
            else:
                break
        return ("alt", branches) if len(branches) > 1 else branches[0]

    def _sequence(self):
        items = []
        while True:
            self._skip_ws()
            if self.pos >= len(self.text):
                break
            if self.text[self.pos] in "|)":
                break
            items.append(self._element())
        return ("seq", items)

    def _element(self):
        self._skip_ws()
        c = self.text[self.pos]
        if c == '"':
            node = self._literal()
        elif c == "[":
            node = self._charclass()
        elif c == "(":
            self.pos += 1
            inner = self._alternation()
            self._skip_ws()
            assert self.text[self.pos] == ")"
            self.pos += 1
            node = ("group", inner)
        else:
            node = ("ref", self._name())
        return self._suffix(node)

    def _suffix(self, node):
        if self.pos < len(self.text) and self.text[self.pos] in "*+?":
            op = self.text[self.pos]
            self.pos += 1
            return ("rep", node, op)
        return node

    def _literal(self):
        self.pos += 1  # opening quote
        out = []
        escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
        while self.pos < len(self.text) and self.text[self.pos] != '"':
            c = self.text[self.pos]
            if c == "\\":
                self.pos += 1
                out.append(escapes.get(self.text[self.pos], self.text[self.pos]))
            else:
                out.append(c)
            self.pos += 1
        assert self.pos < len(self.text)
        self.pos += 1  # closing quote
        return ("lit", "".join(out))

    def _charclass(self):
        self.pos += 1  # '['
        negated = False
        if self.pos < len(self.text) and self.text[self.pos] == "^":
            negated = True
            self.pos += 1
        chars, ranges = set(), []
        while self.pos < len(self.text) and self.text[self.pos] != "]":
            a = self.text[self.pos]
            if a == "\\":
                a = self.text[self.pos + 1]
                self.pos += 2
            else:
                self.pos += 1
            if (self.pos < len(self.text) and self.text[self.pos] == "-"
                    and self.pos + 1 < len(self.text)
                    and self.text[self.pos + 1] != "]"):
                self.pos += 1
                b = self.text[self.pos]
                if b == "\\":
                    b = self.text[self.pos + 1]
                    self.pos += 2
                else:
                    self.pos += 1
                ranges.append((a, b))
            else:
                chars.add(a)
        assert self.pos < len(self.text) and self.text[self.pos] == "]"
        self.pos += 1
        return ("class", chars, ranges, negated)

    # ---- matching ----
    def matches(self, s: str) -> bool:
        return len(s) in self._match(self.rules["root"], s, 0)

    def _match(self, node, s: str, pos: int) -> set[int]:
        kind = node[0]
        if kind == "alt":
            ends = set()
            for branch in node[1]:
                ends |= self._match(branch, s, pos)
            return ends
        if kind == "seq":
            ends = {pos}
            for item in node[1]:
                nxt = set()
                for p in ends:
                    nxt |= self._match(item, s, p)
                ends = nxt
                if not ends:
                    break
            return ends
        if kind == "lit":
            lit = node[1]
            return {pos + len(lit)} if s.startswith(lit, pos) else set()
        if kind == "class":
            if pos >= len(s):
                return set()
            chars, ranges, negated = node[1], node[2], node[3]
            hit = s[pos] in chars or any(a <= s[pos] <= b for a, b in ranges)
            hit = not hit if negated else hit
            return {pos + 1} if hit else set()
        if kind == "group":
            return self._match(node[1], s, pos)
        if kind == "ref":
            return self._match(self.rules[node[1]], s, pos)
        # repetition
        child, op = node[1], node[2]
        if op == "?":
            return {pos} | self._match(child, s, pos)
        frontier = {pos}
        ends = {pos} if op == "*" else set()
        prev = set()
        for _ in range(len(s) + 1):
            nxt = set()
            for p in frontier:
                nxt |= self._match(child, s, p)
            if nxt:
                ends |= nxt
            if not nxt or nxt <= prev:
                break
            prev = nxt
            frontier = nxt
        return ends


class TestGrammar:
    """Real GBNF acceptance/rejection of whole command strings.

    The grammars must accept the host's own install verbs and reject foreign
    ones -- the whole point of the constraint -- and must never accept
    `other-cmd` catch-alls that swallow foreign installers (the pre-fix bug).
    """

    def _assert_accepts(self, pkg_mgr, cmd):
        g = hostctx.grammar_for_pkg(pkg_mgr)
        assert g is not None, f"no grammar for {pkg_mgr}"
        assert _GBNFParser(g).matches(cmd), f"{pkg_mgr} grammar rejected {cmd!r}"

    def _assert_rejects(self, pkg_mgr, cmd):
        g = hostctx.grammar_for_pkg(pkg_mgr)
        assert g is not None, f"no grammar for {pkg_mgr}"
        assert not _GBNFParser(g).matches(cmd), f"{pkg_mgr} grammar accepted {cmd!r}"

    def test_host_install_verbs_accepted(self):
        self._assert_accepts("pacman", "pacman -S htop")
        self._assert_accepts("pacman", "sudo pacman -S htop")
        self._assert_accepts("pacman", "pacman -S htop tmux")
        self._assert_accepts("apt", "apt install htop")
        self._assert_accepts("apt", "sudo apt-get install htop")
        self._assert_accepts("dnf", "sudo dnf install htop")
        self._assert_accepts("apk", "apk add htop")
        self._assert_accepts("brew", "brew install htop")
        self._assert_accepts("zypper", "sudo zypper install htop")
        self._assert_accepts("xbps", "xbps-install htop")

    def test_foreign_install_verbs_rejected(self):
        """Regression for the `other-cmd ::= char+` no-op: a foreign installer
        must be rejected under the host grammar."""
        self._assert_rejects("pacman", "apt-get install htop")
        self._assert_rejects("pacman", "apt install htop")
        self._assert_rejects("pacman", "sudo dnf install htop")
        self._assert_rejects("pacman", "apk add htop")
        self._assert_rejects("pacman", "brew install htop")
        self._assert_rejects("apt", "pacman -S htop")
        self._assert_rejects("dnf", "pacman -S htop")
        self._assert_rejects("apk", "apt-get install htop")
        self._assert_rejects("brew", "apt-get install htop")

    def test_non_install_commands_rejected(self):
        """The grammar is strict: only install verbs. A non-install command
        must not pass, which is why is_install_request gates the grammar on."""
        self._assert_rejects("pacman", "find . -size +100M")
        self._assert_rejects("pacman", "ls -la")

    def test_multiple_commands_accepted(self):
        self._assert_accepts("pacman", "pacman -S htop\npacman -S tmux\n")

    def test_pkg_names_with_plus_and_dot(self):
        """pkg-name must accept libc++1 and python3.11-style names."""
        self._assert_accepts("pacman", "pacman -S libc++1")
        self._assert_accepts("brew", "brew install python3.11")

    def test_dnf_pkg_names_with_digits(self):
        """Regression: the dnf grammar's char class read [a-zA-Z0_9_.+-]
        (`0_`, not `0-9`) -- it happened to accept letters by accident but was
        not the intended class."""
        self._assert_accepts("dnf", "dnf install python3.11")
        self._assert_accepts("dnf", "sudo dnf install libxml2")

    def test_unknown_pkg_returns_none(self):
        assert hostctx.grammar_for_pkg("nonexistent") is None

    def test_all_known_pkgmgrs_have_grammar(self):
        for mgr in ("pacman", "apt", "dnf", "apk", "brew", "zypper", "xbps"):
            assert hostctx.grammar_for_pkg(mgr) is not None

    def test_grammar_has_optional_trailing_newline(self):
        g = hostctx.grammar_for_pkg("pacman")
        assert _GBNFParser(g).matches("pacman -S htop")
        assert _GBNFParser(g).matches("pacman -S htop\n")


class TestIsInstallRequest:
    def test_detects_install_intent(self):
        assert hostctx.is_install_request("install htop")
        assert hostctx.is_install_request("please install the docker package")
        assert hostctx.is_install_request("How do I install htop?")

    def test_ignores_non_install_intent(self):
        assert not hostctx.is_install_request("list files bigger than 100MB")
        assert not hostctx.is_install_request("show disk usage")
        assert not hostctx.is_install_request("find the largest log file")

    def test_remove_is_not_install_intent(self):
        """Removing uses per-manager verbs (pacman -R, apt remove) that no
        single grammar can express, so it must not trigger the grammar."""
        assert not hostctx.is_install_request("remove htop")


class TestPostprocess:
    """Tests for regex post-processing of wrong-distro commands."""

    def test_apt_to_pacman(self):
        assert hostctx.postprocess_command("apt-get install htop", "pacman") == "pacman -S htop"

    def test_sudo_apt_to_pacman(self):
        got = hostctx.postprocess_command("sudo apt-get install htop", "pacman")
        assert got == "sudo pacman -S htop"

    def test_sudo_apt_to_brew_drops_sudo(self):
        """Homebrew aborts under sudo ("Don't run this as root!"); a leftover
        sudo from a rewritten apt command must be stripped."""
        got = hostctx.postprocess_command("sudo apt-get install htop", "brew")
        assert got == "brew install htop"

    def test_sudo_dnf_to_brew_drops_sudo(self):
        got = hostctx.postprocess_command("sudo dnf install htop", "brew")
        assert got == "brew install htop"

    def test_sudo_apt_to_apk_drops_sudo(self):
        """apk runs as root on Alpine; a leftover sudo must be stripped."""
        got = hostctx.postprocess_command("sudo apt-get install htop", "apk")
        assert got == "apk add htop"

    def test_sudo_pacman_to_apk_drops_sudo(self):
        got = hostctx.postprocess_command("sudo pacman -S htop", "apk")
        assert got == "apk add htop"

    def test_no_rewrite_keeps_native_sudo(self):
        """A native sudo command that was NOT rewritten is left alone."""
        got = hostctx.postprocess_command("sudo dnf install -y htop", "dnf")
        assert got == "sudo dnf install -y htop"

    def test_dnf_to_pacman(self):
        assert hostctx.postprocess_command("dnf install htop", "pacman") == "pacman -S htop"

    def test_pacman_to_apt(self):
        assert hostctx.postprocess_command("pacman -S htop", "apt") == "apt install htop"

    def test_strips_deb_flags(self):
        got = hostctx.postprocess_command("apt-get install -y htop", "pacman")
        assert got == "pacman -S htop"
        got = hostctx.postprocess_command(
            "apt-get install --no-install-recommends htop", "pacman")
        assert got == "pacman -S htop"
        got = hostctx.postprocess_command("apt-get install -qq htop", "pacman")
        assert got == "pacman -S htop"

    def test_non_install_commands_unchanged(self):
        # Commands that don't match any rewrite rule pass through
        cmd = "find . -name '*.py' -exec wc -l {} ;"
        got = hostctx.postprocess_command(cmd, "pacman")
        assert got == cmd
        assert hostctx.postprocess_command("ps aux | grep nginx", "pacman") == "ps aux | grep nginx"
        assert hostctx.postprocess_command("ls -la > output.txt", "pacman") == "ls -la > output.txt"
        assert hostctx.postprocess_command("echo $HOME", "pacman") == "echo $HOME"

    def test_does_not_mangle_native_flags_on_non_apt_hosts(self):
        """Regression: deb-flag stripping must NOT run on commands that were
        not rewritten from an apt/legacy install syntax -- otherwise we corrupt
        native commands like `grep -qq` or `dnf install -y`."""
        # `-qq` is valid for grep; must survive untouched (no rewrite happened).
        cmd = "grep -qq pattern file"
        assert hostctx.postprocess_command(cmd, "pacman") == cmd
        assert hostctx.postprocess_command(cmd, "dnf") == cmd
        # `-y` is valid for dnf/yum/zypper (not only apt); native installs are
        # preserved because no rewrite took place.
        dnf_cmd = "dnf install -y htop"
        zypper_cmd = "zypper install -y htop"
        assert hostctx.postprocess_command(dnf_cmd, "dnf") == dnf_cmd
        assert hostctx.postprocess_command(zypper_cmd, "zypper") == zypper_cmd
        assert hostctx.postprocess_command("pacman -S htop", "pacman") == "pacman -S htop"

    def test_preserves_interior_double_spaces_when_no_rewrite(self):
        """Regression: whitespace collapsing must run only after an actual
        rewrite. An untouched command may carry meaningful runs of spaces
        inside quoted arguments (grep 'foo  bar', awk -F'  ')."""
        cmd = "grep 'foo  bar' report.txt"
        assert hostctx.postprocess_command(cmd, "pacman") == cmd
        assert hostctx.postprocess_command("sed 's/a  b/x/'", "pacman") == "sed 's/a  b/x/'"
        assert hostctx.postprocess_command(
            "awk -F'  ' '{print $1}'", "pacman") == "awk -F'  ' '{print $1}'"
        assert hostctx.postprocess_command("echo 'two  spaces'", "apt") == "echo 'two  spaces'"

    def test_strips_deb_flags_only_after_a_rewrite(self):
        """After translating an apt command, apt-only flags are removed."""
        got = hostctx.postprocess_command("apt-get install -y htop", "pacman")
        assert got == "pacman -S htop"
        got = hostctx.postprocess_command(
            "apt-get install --no-install-recommends htop", "pacman")
        assert got == "pacman -S htop"

    def test_multi_package_install(self):
        got = hostctx.postprocess_command("apt-get install htop nmon", "pacman")
        assert got == "pacman -S htop nmon"


# ---------------------------------------------------------------------------
# pkg_guidance_line() + build(pkg_line=...)
# ---------------------------------------------------------------------------

class TestPkgGuidanceLine:
    def _facts(self, pkg):
        return {"pkg": pkg, "distro": "Arch Linux", "distro_version": "",
                "arch": "x86_64", "shell": "zsh", "present": [], "missing": []}

    def test_exact_constraint_text_for_pacman(self):
        """The line joins the cached system prompt: pin its full wording so an
        accidental edit cannot silently change what every host sends."""
        expected = ("<constraint>\n"
                    "This machine manages packages exclusively with pacman; "
                    "install, update, and remove software only with pacman.\n"
                    "</constraint>")
        assert hostctx.pkg_guidance_line(self._facts("pacman")) == expected

    def test_unknown_pkg_returns_empty(self):
        assert hostctx.pkg_guidance_line(self._facts("unknown")) == ""

    def test_does_not_name_foreign_managers(self):
        """stable_block()'s note: naming a banned tool primes a small model
        toward it. The guidance line must mention only the host's manager."""
        line = hostctx.pkg_guidance_line(self._facts("pacman"))
        for foreign in ("apt", "dnf", "yum", "zypper", "apk", "brew"):
            assert foreign not in line

    def test_front_loads_the_manager_name(self):
        """The shipped 1.5B model weighs early tokens heavily: the manager
        name must appear in the first sentence half, not the tail."""
        line = hostctx.pkg_guidance_line(self._facts("dnf"))
        first_half = line[:len(line) // 2]
        assert "dnf" in first_half

    def test_byte_stable_across_calls(self, monkeypatch, tmp_path):
        """The line joins the prefix-cached system prompt; it must be
        deterministic so the cache stays valid."""
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        facts = self._facts("apt")
        assert hostctx.pkg_guidance_line(facts) == hostctx.pkg_guidance_line(facts)

    def test_uses_real_facts_when_none_given(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(hostctx, "_distro_info",
                            lambda: {"id": "arch", "name": "Arch Linux",
                                     "version": "", "version_id": "",
                                     "id_like": []})
        monkeypatch.setattr(hostctx.shutil, "which",
                            lambda x: x == "pacman")
        monkeypatch.setattr(hostctx.platform, "system", lambda: "Linux")
        line = hostctx.pkg_guidance_line()
        assert "pacman" in line


class TestBuildPkgLine:
    """build(pkg_line=True) folds in only the one-line guidance -- the opt-in
    distro_guidance mode that keeps the measured-harmful facts block off."""

    def _probe_pacman(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(hostctx, "_distro_info",
                            lambda: {"id": "arch", "name": "Arch Linux",
                                     "version": "", "version_id": "",
                                     "id_like": []})
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: x == "pacman")
        monkeypatch.setattr(hostctx.platform, "system", lambda: "Linux")

    def test_pkg_line_appended_without_facts_block(self, monkeypatch, tmp_path):
        self._probe_pacman(monkeypatch, tmp_path)
        system, user = hostctx.build("install numpy", enabled=False,
                                     include_volatile=False, pkg_line=True)
        assert "<host_environment>" not in system
        assert "<constraint>" in system
        assert "pacman" in system
        assert user == "install numpy"

    def test_pkg_line_keeps_volatile_out_of_the_system_prompt(self, monkeypatch, tmp_path):
        """The guidance line joins the cached prefix; volatile facts must stay
        on the user side exactly as with the full facts block."""
        self._probe_pacman(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        system, user = hostctx.build("list files", enabled=False, pkg_line=True)
        assert "Working directory" not in system
        assert "Working directory" in user

    def test_no_line_when_pkg_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHATISIT_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(hostctx, "_distro_info",
                            lambda: {"id": "nixos", "name": "NixOS",
                                     "version": "", "version_id": "",
                                     "id_like": []})
        monkeypatch.setattr(hostctx.shutil, "which", lambda x: False)
        monkeypatch.setattr(hostctx.platform, "system", lambda: "Linux")
        system, user = hostctx.build("install numpy", enabled=False,
                                     include_volatile=False, pkg_line=True)
        assert system == cfg_mod.SYSTEM_PROMPT
        assert user == "install numpy"

    def test_full_block_subsumes_the_line(self, monkeypatch, tmp_path):
        self._probe_pacman(monkeypatch, tmp_path)
        system, _ = hostctx.build("install numpy", enabled=True, pkg_line=True)
        assert "<host_environment>" in system
        # exactly one constraint mentioning pacman comes from stable_block()
        assert system.count("This machine manages packages") == 0

    def test_disabled_and_no_pkg_line_is_plain(self):
        system, user = hostctx.build("install numpy", enabled=False, pkg_line=False)
        assert system == cfg_mod.SYSTEM_PROMPT
        assert user == "install numpy"
