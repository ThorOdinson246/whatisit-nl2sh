r"""Tell the model about the machine it is running on.

Why this exists, and why it is shaped this way.

The single most common failure in real use was the model emitting a placeholder
(`cp -r /path/to/dir /tmp`) or naming a tool the host does not have
(`netstat -tuln | grep 5000` on a box with no netstat). Both are the same defect:
the model is guessing at facts it could simply be told.

Measured on the target hardware: putting a host-facts block in the SYSTEM prompt
is nearly free, because llama-server prefix-caches it. A 685-token block cost
11.1 s on the first query and then 1.25 / 0.54 / 1.92 s -- i.e. baseline. Paid
once per session. In the same test the block changed
`netstat -tuln | grep 5000` into `ss -lptn | grep 5000`, fixing a real observed
failure, because the block said netstat was absent and ss shows pids.

Two rules follow from that measurement:

STABLE facts (distro, package manager, which tools exist, shell) go in the
system prompt so the cached prefix stays byte-identical across queries.

VOLATILE facts (cwd, directory listing, git branch) must NOT go there -- they
change per query and would invalidate the cache every time, turning a once-per-
session cost into a per-query one. They are appended to the user turn instead.

Budget: the stable block is capped (see MAX_STABLE_CHARS). The 685-token block
used in the experiment was deliberately inflated to measure the worst case; a
real one is ~150 tokens, about 2.4 s of one-time prefill.

STATUS: DISABLED BY DEFAULT, on measurement.

I originally ranked this the highest-value change on the strength of 4 hand-run
examples. A proper 295-task run reversed that, and the anecdote was simply too
small to rank on:

    no context   0.849 mean / 54.2% pass
    context      0.815 mean / 45.1% pass    d=-0.034, McNemar p=0.0004

Two implementation bugs accounted for part of it (a false working-directory claim
and a key=value format the model read as shell variables); fixing both recovered
39.0% -> 45.1% but did not close the gap. Ruled out as the cause: only 3 of 41
regressions involve a tool the context declared missing.

What actually happens is that the extra tokens make a 1.5B model produce more
elaborate answers, and elaboration breaks simple correct ones:
    `sha512sum f`            -> `echo -n "f" | sha512sum`   (hashes the NAME)
    `echo 'hello' > world.txt` -> `touch /testbed/world.txt`  (drops the content)
    `find .. | xargs wc -l`  -> `find .. -exec wc -l {} \;`  (working idiom traded away)

But the benchmark is close to blind to the BENEFIT this is for. Measured on the
task text: 73% of ALFA tasks already name a concrete target path, so context is
pure noise there, and the underspecified 27% are trivia (`ls`, `pwd`, `date`)
that need no context either. Of real typed queries, 79% are underspecified --
the opposite distribution.

So ALFA measures this feature's cost in full and its benefit barely. The measured
harm is real and sufficient reason not to enable it by default; it is NOT
sufficient reason to conclude the idea is wrong. Revisit only against an eval
built from underspecified requests.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import config as cfg_mod

CACHE_TTL = 24 * 3600      # installed tools change rarely
MAX_STABLE_CHARS = 900     # ~200 tokens; keeps first-query prefill near 2-3 s
MAX_ENTRIES = 12           # directory listing truncation

# Tools worth telling the model about, chosen from observed failures rather than
# from a generic "useful commands" list: each one is either a tool the model
# reached for and the host lacked, or the correct modern replacement for one.
PROBE_TOOLS = [
    "ss", "netstat", "lsof", "ip", "ifconfig",       # the port-lookup failures
    "docker", "podman", "kubectl",
    "git", "rg", "fd", "jq", "tree", "ncdu",
    "zip", "unzip", "tar", "xz", "rsync", "curl", "wget",
    "python3", "node", "shellcheck",
    "squeue", "sbatch",                              # HPC: scheduler present?
    "systemctl", "brew", "apt", "dnf", "yum", "pacman", "apk",
]

PKG_MANAGERS = [("apt", "apt"), ("dnf", "dnf"), ("yum", "yum"),
                ("pacman", "pacman"), ("apk", "apk"), ("brew", "brew"),
                ("zypper", "zypper")]


def _distro_info() -> dict:
    """Return structured distro metadata from /etc/os-release.

    Falls back to platform.system() / platform.release() when the file is
    absent (e.g. macOS, WSL without os-release, minimal containers).
    """
    try:
        data = dict(
            line.split("=", 1)
            for line in Path("/etc/os-release").read_text(errors="replace").splitlines()
            if "=" in line
        )
        return {
            "id": data.get("ID", "").strip('"').lower(),
            "name": data.get("NAME", data.get("PRETTY_NAME", "")).strip('"'),
            "version": data.get("VERSION", "").strip('"'),
            "version_id": data.get("VERSION_ID", "").strip('"'),
            "id_like": [v.strip('"').lower() for v in data.get("ID_LIKE", "").split()],
        }
    except (OSError, ValueError):
        return {
            "id": "",
            "name": platform.system(),
            "version": "",
            "version_id": "",
            "id_like": [],
        }


def _distro() -> str:
    info = _distro_info()
    if info["name"]:
        return info["name"]
    return platform.system()


# Canonical package manager for each well-known distro ID. Checked in order; the
# first match wins. This is a DECLARATIVE hint, not a runtime check -- the real
# presence check is done by PKG_MANAGERS / shutil.which below. It lets the
# stable block name the RIGHT package manager even on distros (e.g. Archcraft,
# NixOS) where the binary name differs from the distro ID.
DISTRO_PKG_MAP: dict[str, str] = {
    "ubuntu": "apt",
    "debian": "apt",
    "linuxmint": "apt",
    "pop": "apt",
    "zorin": "apt",
    "elementary": "apt",
    "mx": "apt",
    "raspbian": "apt",
    "kali": "apt",
    "fedora": "dnf",
    "centos": "dnf",
    "rhel": "dnf",
    "rocky": "dnf",
    "almalinux": "dnf",
    "amazon": "dnf",
    "opensuse-leap": "zypper",
    "opensuse-tumbleweed": "zypper",
    "opensuse": "zypper",
    "sles": "zypper",
    "arch": "pacman",
    "manjaro": "pacman",
    "endeavouros": "pacman",
    "cachyos": "pacman",
    "garuda": "pacman",
    "artix": "pacman",
    "alpine": "apk",
    "void": "xbps",
}


def _canonical_pkg(distro_id: str, id_like: list[str]) -> str:
    """Map a distro ID (and its ID_LIKE family) to the canonical package manager.

    This is a DECLARATIVE map: the actual binary-presence check is done by the
    PKG_MANAGERS loop in _probe(). Here we only pick the name the model should
    use when it talks about this machine's package manager. That matters for
    distros like Archcraft (ID=archcraft, ID_LIKE=arch) where the binary is
    `pacman` but the ID is not in DISTRO_PKG_MAP by itself.
    """
    checked: set[str] = set()
    for candidate in (distro_id,) + tuple(id_like):
        if candidate in checked:
            continue
        checked.add(candidate)
        if pkg := DISTRO_PKG_MAP.get(candidate):
            return pkg
    # Homebrew is the universal answer on macOS. On an unmapped Linux distro
    # (NixOS, Gentoo, ...) assuming brew would hand the model macOS guidance
    # and grammar; "unknown" lets _probe fall back to whatever binary exists.
    if platform.system() == "Darwin":
        return "brew"
    return "unknown"


def _cache_path() -> Path:
    return cfg_mod.data_dir() / "hostctx.json"


def _probe() -> dict:
    info = _distro_info()
    present = [t for t in PROBE_TOOLS if shutil.which(t)]
    missing = [t for t in PROBE_TOOLS if t not in present]
    # The canonical pkg name from the distro map; prefer it when its binary is
    # actually present, so an unrelated foreign binary on PATH (e.g. an apt
    # package installed on Arch) cannot override the declared manager. PATH
    # discovery is only a fallback for distros whose declared binary is absent
    # (custom spins whose binary name differs from the ID).
    decl_pkg = _canonical_pkg(info["id"], info["id_like"])
    decl_bin = next((bin_ for bin_, name in PKG_MANAGERS if name == decl_pkg), None)
    if decl_pkg != "unknown" and decl_bin is not None and shutil.which(decl_bin):
        pkg = decl_pkg
    else:
        found_pkg = next((name for bin_, name in PKG_MANAGERS if shutil.which(bin_)), "unknown")
        pkg = found_pkg if found_pkg != "unknown" else decl_pkg
    return {
        "generated": time.time(),
        "distro": info["name"] or platform.system(),
        "distro_id": info["id"],
        "distro_version": info["version_id"] or info["version"],
        "kernel": platform.release(),
        "arch": platform.machine(),
        "shell": Path(os.environ.get("SHELL", "/bin/sh")).name,
        "pkg": pkg,
        "present": present,
        "missing": missing,
    }


def stable_facts(refresh: bool = False) -> dict:
    """Probe the host, cached to disk. Cheap after the first call."""
    p = _cache_path()
    if not refresh and p.exists():
        try:
            d = json.loads(p.read_text())
            if time.time() - d.get("generated", 0) < CACHE_TTL:
                return d
        except (json.JSONDecodeError, OSError):
            pass
    d = _probe()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2))
    except OSError:
        pass
    return d


# Few-shot examples keyed by package manager, injected into the system prompt.
# Hypothesis (not benchmarked): a concrete example steers the model toward the
# host's command style more reliably than an abstract constraint. Each example
# mirrors the exact command style the model should emit.
FEW_SHOT_EXAMPLES: dict[str, str] = {
    "apt": "User: install htop\nAssistant: sudo apt install htop",
    "dnf": "User: install htop\nAssistant: sudo dnf install htop",
    "pacman": "User: install htop\nAssistant: sudo pacman -S htop",
    "apk": "User: install htop\nAssistant: apk add htop",
    "brew": "User: install htop\nAssistant: brew install htop",
    "zypper": "User: install htop\nAssistant: sudo zypper install htop",
    "xbps": "User: install htop\nAssistant: xbps-install htop",
}


def stable_block(facts: dict | None = None) -> str:
    """The part that goes in the system prompt. Must be stable across queries."""
    f = facts or stable_facts()
    # Include version when present so the model can give version-specific
    # commands (e.g. apt vs. apt-get, dnf vs. yum, brew vs. port).
    version_tag = f" {f['distro_version']}" if f.get("distro_version") else ""
    lines = [
        "<host_environment>",
        f"OS: {f['distro']}{version_tag} ({f['arch']})",
        f"Shell: {f['shell']}",
        f"Package manager: {f['pkg']}",
        f"Available tools: {' '.join(f['present'])}",
        "</host_environment>",
    ]
    # NOTE: We deliberately omit a "Banned tools" line. Hypothesis (not
    # benchmarked): small models (<3B) prime on negative constraints --
    # mentioning `apt` in the prompt may increase the probability the model
    # outputs `apt-get`, even inside a "banned" list. The <example> block and
    # <constraint> tags provide positive guidance without activating the wrong
    # concepts.
    # Steer to the modern tool when the legacy one is genuinely unavailable.
    if "ss" in f["present"] and "netstat" in f["missing"]:
        lines.append("<constraint>")
        lines.append("For listening ports use `ss -lptn` (shows the owning pid); "
                     "netstat is unavailable.")
        lines.append("</constraint>")
    if "lsof" in f["missing"] and "ss" in f["present"]:
        lines.append("<constraint>")
        lines.append("lsof is unavailable; use `ss -lptn` or `fuser` instead.")
        lines.append("</constraint>")
    # Distro-specific package manager guidance: some distros ship legacy names
    # alongside modern ones, and the model should use the canonical one.
    if f["pkg"] == "apt":
        lines.append("<constraint>")
        lines.append("Install packages with `apt install <pkg>` (not `apt-get install`).")
        lines.append("</constraint>")
    elif f["pkg"] == "dnf":
        lines.append("<constraint>")
        lines.append("Install packages with `dnf install <pkg>` (not `yum install`).")
        lines.append("</constraint>")
    elif f["pkg"] == "pacman":
        lines.append("<constraint>")
        lines.append("Install packages with `pacman -S <pkg>`; "
                      "use `pacman -Syy` to force a full refresh.")
        lines.append("</constraint>")
    elif f["pkg"] == "apk":
        lines.append("<constraint>")
        lines.append("Install packages with `apk add <pkg>`.")
        lines.append("</constraint>")
    elif f["pkg"] == "brew":
        lines.append("<constraint>")
        lines.append("Install packages with `brew install <pkg>`.")
        lines.append("</constraint>")
    # Few-shot example: one concrete usage example for this distro's pkg mgr.
    # Neutral guidance only -- its accuracy effect is not separately measured.
    if example := FEW_SHOT_EXAMPLES.get(f["pkg"]):
        lines.append("<example>")
        lines.append(example)
        lines.append("</example>")
    block = "\n".join(lines)
    return block[:MAX_STABLE_CHARS]


def volatile_block(cwd: Path | None = None) -> str:
    """Per-query facts. Deliberately NOT in the system prompt -- see module docstring."""
    cwd = Path(cwd or Path.cwd())
    # Prose labels, NOT `key=value`. Measured: a `cwd=/testbed` line made the
    # model treat the key as a shell variable and emit `mkdir -p $cwd/test_dir`
    # and `for i in $(echo $cwd_entries ...)`. The context format itself was
    # teaching it to reference variables that do not exist.
    lines = [f"Working directory is {cwd}"]
    try:
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in cwd.iterdir()
                         if not p.name.startswith("."))
        shown = entries[:MAX_ENTRIES]
        more = f" (+{len(entries)-len(shown)} more)" if len(entries) > len(shown) else ""
        lines.append(f"It contains: {' '.join(shown)}{more}" if shown else "It is empty.")
    except OSError:
        pass
    if git := _git_state(cwd):
        lines.append(git)
    return "\n".join(lines)


def _git_state(cwd: Path) -> str:
    """Branch and dirtiness, or '' if not a repo. Short timeout: a slow NFS repo
    must never delay a command suggestion."""
    if not shutil.which("git"):
        return ""
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(cwd),
                           capture_output=True, text=True, timeout=1.5)
        if r.returncode != 0:
            return ""
        branch = r.stdout.strip()
        s = subprocess.run(["git", "status", "--porcelain"], cwd=str(cwd),
                           capture_output=True, text=True, timeout=1.5)
        dirty = "dirty" if s.stdout.strip() else "clean"
        return f"It is a git repo on branch {branch}, working tree {dirty}."
    except (subprocess.TimeoutExpired, OSError):
        return ""


# Regex substitutions that translate a model's output from one distro's syntax
# to another. Applied as a duct-tape fallback when the 1.5B model gets the
# intent right but the syntax wrong (e.g. `apt-get install` on Arch).
# Each pattern is deliberately broad to catch variants the model might emit.
# Debian/Ubuntu flags that have no equivalent on most other distros (-y, -qq,
# --no-install-recommends) are consumed inline by the apt/dnf patterns so they
# are dropped during replacement without a separate global strip pass.
# There are deliberately NO `sudo ...` variants: the bare pattern matches the
# tail of the sudo form (`\bapt...` matches inside `sudo apt install`), so a
# sudo twin could never fire. Whether the leading `sudo` survives is handled
# explicitly in postprocess_command via _NO_SUDO.
_DEB_FLAGS_RE = r"(?:\s+(?:-y|-qq|--no-install-recommends))*"

# Managers that must never be invoked through sudo. Homebrew aborts outright
# ("Don't run this as root!"); apk is normally run as root on Alpine, where
# sudo may not even be installed.
_NO_SUDO = {"brew", "apk"}

PKG_MGR_REWRITE: list[tuple[str, str, str]] = [
    # pacman host: translate Debian/Ubuntu/Fedora syntax -> Arch
    ("pacman",
     re.compile(rf"\bapt(-get)?\s+install{_DEB_FLAGS_RE}\b"), "pacman -S"),
    ("pacman",
     re.compile(rf"\bdnf\s+install{_DEB_FLAGS_RE}\b"), "pacman -S"),
    # apt host: translate Arch syntax -> Debian
    ("apt",
     re.compile(r"\bpacman\s+-S\b"), "apt install"),
    # dnf host: translate Arch or Debian syntax -> Fedora
    ("dnf",
     re.compile(r"\bpacman\s+-S\b"), "dnf install"),
    ("dnf",
     re.compile(rf"\bapt(-get)?\s+install{_DEB_FLAGS_RE}\b"), "dnf install"),
    # apk host: translate everything else -> Alpine
    ("apk",
     re.compile(rf"\bapt(-get)?\s+install{_DEB_FLAGS_RE}\b"), "apk add"),
    ("apk",
     re.compile(r"\bpacman\s+-S\b"), "apk add"),
    ("apk",
     re.compile(rf"\bdnf\s+install{_DEB_FLAGS_RE}\b"), "apk add"),
    # brew host: translate Linux syntax -> macOS
    ("brew",
     re.compile(rf"\bapt(-get)?\s+install{_DEB_FLAGS_RE}\b"), "brew install"),
    ("brew",
     re.compile(r"\bpacman\s+-S\b"), "brew install"),
    ("brew",
     re.compile(rf"\bdnf\s+install{_DEB_FLAGS_RE}\b"), "brew install"),
    ("brew",
     re.compile(r"\bapk\s+add\b"), "brew install"),
]


def postprocess_command(cmd: str, pkg_mgr: str) -> str:
    """Rewrite a command from the wrong distro's syntax to the host's.

    Debian-only flags (-y, -qq, --no-install-recommends) are consumed inline by
    the PKG_MGR_REWRITE patterns, so they are dropped only when the install verb
    was actually rewritten. Native commands that happen to contain these flags
    (e.g. ``grep -qq``, ``dnf install -y``) are left intact.
    """
    rewritten = False
    for host_pkg, pattern, replacement in PKG_MGR_REWRITE:
        if host_pkg == pkg_mgr:
            new_cmd = pattern.sub(replacement, cmd)
            if new_cmd != cmd:
                cmd = new_cmd
                rewritten = True
    # Managers that must never run under sudo (brew, apk) get their leading
    # sudo dropped after a rewrite -- the bare patterns match the tail of the
    # sudo form, so `sudo apt install htop` becomes `sudo brew install htop`
    # and the leftover sudo would make Homebrew abort.
    if rewritten and pkg_mgr in _NO_SUDO:
        cmd = re.sub(r"^sudo\s+", "", cmd)
    # Collapse double-spaces left by the substitution. Only safe when we
    # actually rewrote: an untouched command may hold meaningful runs of
    # spaces inside quotes (grep 'foo  bar', awk -F'  ').
    if rewritten:
        cmd = re.sub(r"  +", " ", cmd).strip()
    return cmd


# GBNF grammars for install-intent prompts (see is_install_request). For those
# prompts this IS a hard constraint: the model cannot emit a foreign installer
# (e.g. `apt-get install htop` on Arch) even if its training bias pulls it that
# direction. Non-install queries are not grammar-constrained at all; the regex
# rewrite in postprocess_command remains the backstop for every path.
PKG_MGR_GRAMMARS: dict[str, str] = {
    # Rule names use dashes (not underscores) because GBNF identifiers only
    # accept [a-zA-Z0-9-] in some llama.cpp builds.
    "pacman": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo-install | install
sudo-install ::= "sudo " install
install      ::= "pacman -S " pkg-name (" " pkg-name)*
pkg-name     ::= [a-zA-Z0-9_.+-]+
''',
    "apt": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo-install | install
sudo-install ::= "sudo " install
install      ::= ("apt " | "apt-get ") "install " pkg-name (" " pkg-name)*
pkg-name     ::= [a-zA-Z0-9_.+-]+
''',
    "dnf": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo-install | install
sudo-install ::= "sudo " install
install      ::= "dnf install " pkg-name (" " pkg-name)*
pkg-name     ::= [a-zA-Z0-9_.+-]+
''',
    "apk": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= install
install      ::= "apk add " pkg-name (" " pkg-name)*
pkg-name     ::= [a-zA-Z0-9_.+-]+
''',
    "brew": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= install
install      ::= "brew install " pkg-name (" " pkg-name)*
pkg-name     ::= [a-zA-Z0-9_.+-]+
''',
    "zypper": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= sudo-install | install
sudo-install ::= "sudo " install
install      ::= "zypper install " pkg-name (" " pkg-name)*
pkg-name     ::= [a-zA-Z0-9_.+-]+
''',
    "xbps": r'''
root         ::= command ("\n" command)* "\n"?
command      ::= install
install      ::= "xbps-install " pkg-name (" " pkg-name)*
pkg-name     ::= [a-zA-Z0-9_.+-]+
''',
}


# Install-intent detection used to gate the grammars above. Installing is the
# one intent whose output the grammar can hard-constrain; removing/updating use
# different verbs per manager (pacman -R vs apt remove) that no single grammar
# can express, so they are deliberately left to the regex rewrite backstop.
_INSTALL_RE = re.compile(
    r"\b(install|installs|installing|installed|installation|reinstall)\b",
    re.IGNORECASE)


def is_install_request(prompt: str) -> bool:
    """True when the user is asking to install packages."""
    return bool(_INSTALL_RE.search(prompt))


def grammar_for_pkg(pkg_mgr: str) -> str | None:
    """Return a GBNF grammar string for the given package manager, or None."""
    return PKG_MGR_GRAMMARS.get(pkg_mgr)


def pkg_guidance_line(facts: dict | None = None) -> str:
    """One stable sentence naming the host's package manager, or '' if unknown.

    This is the distro_guidance alternative to the full facts block: the block
    is measured harmful on the shipped model (see module docstring), but a
    wrong-distro install command remains a real failure. Three constraints
    shape the wording. The shipped model is small (Qwen2.5-1.5B finetune) and
    weighs early tokens heavily, so the manager's name sits inside the first
    few tokens rather than at the end of the line. Foreign package managers
    are deliberately NOT named -- see the note in stable_block(): naming a
    wrong tool, even to ban it, primes a small model toward it. And the line
    must be byte-identical per host, because it joins the prefix-cached system
    prompt; any per-query variation would turn the once-per-session prefill
    cost into a per-query one.
    """
    f = facts or stable_facts()
    pkg = f.get("pkg", "unknown")
    if pkg == "unknown":
        return ""
    return ("<constraint>\n"
            f"This machine manages packages exclusively with {pkg}; "
            f"install, update, and remove software only with {pkg}.\n"
            "</constraint>")


def build(prompt: str, enabled: bool = True, cwd: Path | None = None,
          include_volatile: bool = True, pkg_line: bool = False) -> tuple[str, str]:
    """Return (system_prompt, user_message) with context folded in.

    pkg_line folds in just pkg_guidance_line() instead of the full block, for
    the opt-in distro_guidance mode. It has no effect when the full block is
    already enabled (the block subsumes the line).
    """
    if not enabled and not pkg_line:
        return cfg_mod.SYSTEM_PROMPT, prompt
    parts = [cfg_mod.SYSTEM_PROMPT]
    if enabled:
        parts.append(stable_block())
    else:
        line = pkg_guidance_line()
        if line:
            parts.append(line)
    if include_volatile:
        user = volatile_block(cwd) + "\n\n<request>\n" + prompt + "\n</request>"
    else:
        user = prompt
    return "\n\n".join(parts), user
