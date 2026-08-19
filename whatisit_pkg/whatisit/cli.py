"""whatisit -- natural language to shell command, fully local.

Usage:
    whatisit find files larger than 100MB in this folder
    whatisit "find files over 100MB modified this week"
    whatisit -n 3 compress this folder      # show alternatives
    whatisit -e count lines in every python file   # run it, after confirming

Subcommands: setup, doctor, stop, config
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as cfg_mod
from . import engine, fetch
from .safety import check

# Colour only when attached to a terminal, and honour NO_COLOR. Tracked per
# stream: colour is chosen against stdout but half the output goes to stderr,
# so `whatisit foo 2>err.log` would otherwise write escape codes into the file.
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_TTY_ERR = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def out(msg: str) -> None:
    """The command itself, and anything the user asked to see."""
    print(msg)


def warn(msg: str = "", end: str = "\n") -> None:
    """Everything else.

    `eval "$(whatisit -q ...)"` runs whatever reaches stdout, so a stray
    print() there is executed rather than read. #19 shipped exactly that.
    """
    print(msg if _TTY_ERR else _ANSI_RE.sub("", msg), file=sys.stderr, end=end)


def _is_windows() -> bool:
    """Isolated so tests can fake Windows without rewriting global os.name.

    Monkeypatching os.name="nt" on Linux breaks pathlib.Path.home() (used by
    config_dir during main()), which then raises RuntimeError looking for a
    Windows home directory that does not exist on the CI host.
    """
    return os.name == "nt"


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


BOLD = lambda s: _c("1", s)
DIM = lambda s: _c("2", s)
RED = lambda s: _c("1;31", s)
YELLOW = lambda s: _c("33", s)
GREEN = lambda s: _c("32", s)
CYAN = lambda s: _c("36", s)


def print_findings(findings) -> bool:
    """Print safety findings. Returns True if any were DANGER."""
    danger = False
    for sev, why in findings:
        if sev == "DANGER":
            danger = True
            warn(f"  {RED('!! DANGER')}  {why}")
        else:
            warn(f"  {YELLOW('!  caution')} {why}")
    return danger


def _log_query(prompt: str, cmds: list, elapsed: float, mode: str) -> None:
    """Append one query to a local JSONL, for later hand-judging.

    `verdict` is left null deliberately: correctness here can only be decided
    by a human or by execution in the harness, never by the model that produced
    the answer.
    """
    import datetime
    import json
    path = cfg_mod.data_dir() / "queries.jsonl"
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "nl": prompt, "candidates": cmds, "latency_s": round(elapsed, 3),
           "mode": mode, "verdict": None}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        # Create at 0600 from open() rather than chmod-ing after: write-then-
        # chmod leaves the first line -- a real shell request, which can carry
        # hostnames, paths and secrets -- readable at the umask default
        # (group-readable on a shared NFS home) for the width of that window.
        # O_NOFOLLOW is POSIX-only; Windows has no equivalent flag on os.open.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if os.name != "nt":
            flags |= os.O_NOFOLLOW
        fd = os.open(str(path), flags, 0o600)
        try:
            os.fchmod(fd, 0o600)   # also covers a pre-existing, wrongly-permissioned file
        except (OSError, AttributeError):
            pass
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the tool


def _warn_stray_flags(args) -> None:
    """Say when a flag ended up inside the request instead of acting as a flag.

    Flags are only read before the request text, for the reasons in QueryArgs.
    That is the right trade, but it makes `whatisit list files -e` send "-e" to
    the model, which answers something strange. Silence there is the worst
    outcome: the user sees a wrong command and no clue why. Goes to stderr, so
    `$(whatisit -q ...)` stays clean.
    """
    stray = getattr(args, "stray_flags", None)
    if not stray:
        return
    first = stray[0]
    rest = " ".join(w for w in args.words if w not in stray)
    warn(DIM(f"  note: {first} was read as part of your request, not as a flag."))
    warn(DIM(f"        flags go first:  whatisit {first} {rest}"))


def _emit_debug(prompt: str, system: str, user_msg: str, grammar: str | None) -> None:
    """Print the exact prompt + grammar the model sees, for diagnosing failures."""
    print(DIM(f"  --debug: system prompt length = {len(system)}"), file=sys.stderr)
    print(DIM(f"  --debug: grammar = {'none' if grammar is None else 'set'}"), file=sys.stderr)
    print(DIM(f"  --debug: user prompt:\n{user_msg}"), file=sys.stderr)
    if grammar:
        print(DIM(f"  --debug: GBNF grammar:\n{grammar}"), file=sys.stderr)
    print(DIM(f"  --debug: request:\n{prompt}"), file=sys.stderr)


def cmd_query(args, cfg: dict) -> int:
    prompt = " ".join(args.words).strip()
    if not prompt:
        warn("whatisit: nothing to do -- give me a request in plain English")
        return 2

    # Per-invocation overrides from CLI flags. These win over the saved config
    # file but do not persist to it (that is `whatisit config --set`'s job).
    if args.model is not None:
        os.environ["WHATISIT_MODEL"] = args.model
    if args.threads is not None:
        cfg["threads"] = args.threads
    if args.port is not None:
        cfg["server_port"] = args.port
    if args.ctx_size is not None:
        cfg["ctx_size"] = args.ctx_size
    if args.host_context is not None:
        cfg["host_context"] = args.host_context
    if args.grammar is not None:
        cfg["use_grammar"] = args.grammar
    if args.debug:
        # Show the exact prompt sent to the model and the active GBNF grammar
        # (if any). Intended for diagnosing why the 1.5B model misbehaves; goes
        # to stderr so `-q` output is unaffected.
        system, user_msg = engine.hostctx.build(prompt,
                                                enabled=cfg.get("host_context", True))
        pkg = "unknown"
        grammar = None
        if cfg.get("host_context", True) and cfg.get("use_grammar", True):
            try:
                pkg = engine.hostctx.stable_facts().get("pkg", "unknown")
            except OSError:
                pass
            if pkg != "unknown" and hasattr(engine.hostctx, "grammar_for_pkg"):
                grammar = engine.hostctx.grammar_for_pkg(pkg)
        _emit_debug(prompt, system, user_msg, grammar)

    # Remote mode sends the request (and host context, if enabled) somewhere
    # else, which is the one thing this tool otherwise promises never to do.
    # The warning is opt-in and goes to stderr so -q/$(...) output stays clean.
    remote = cfg_mod.remote_config(cfg)
    if remote is not None:
        for w in engine.remote_warnings(remote):
            warn(DIM(f"  note: {w}"))

    try:
        cmds, elapsed, mode = engine.generate(
            prompt, cfg, n=args.num, force_oneshot=args.oneshot, quiet=args.quiet,
            for_execution=args.execute or args.quiet)
    except FileNotFoundError as e:
        warn(f"whatisit: {e}")
        return 3
    except Exception as e:
        warn(f"whatisit: generation failed: {e}")
        return 4

    if not cmds:
        warn("whatisit: the model returned nothing usable. Try rephrasing.")
        return 5

    # --quiet keeps STDOUT bare so it composes: x=$(whatisit -q ...).
    # It must still run the safety check: eval "$(whatisit -q ...)" is the
    # documented scripting idiom, so this is the one path that pipes straight
    # into a shell. Warnings go to stderr, leaving stdout clean.
    if args.quiet:
        findings = check(cmds[0])
        if any(sev == "DANGER" for sev, _ in findings):
            warn("whatisit: refusing to emit a command flagged DANGER:")
            print_findings(findings)
            return 6
        out(cmds[0])
        if findings:
            sys.stdout.flush()
            print_findings(findings)
        _warn_stray_flags(args)
        return 0

    for i, c in enumerate(cmds):
        label = f"{DIM(f'{i+1}.')} " if len(cmds) > 1 else ""
        out(f"{label}{BOLD(CYAN(c))}")
        findings = check(c)
        if findings:
            # The command goes to stdout and warnings to stderr (so that
            # `whatisit ... | sh` still works). Without this flush the two streams
            # interleave and the warning appears ABOVE the command it is about.
            sys.stdout.flush()
            print_findings(findings)
            sys.stderr.flush()

    # Always flush before anything else reaches stderr, so the command is
    # never printed after messages that refer to it.
    sys.stdout.flush()
    _warn_stray_flags(args)

    # Opt-in only (`whatisit config --set log_queries=true`). Shell requests can
    # contain hostnames, paths and credentials, so this is never on by default
    # and stays on the local disk.
    if cfg.get("log_queries"):
        _log_query(prompt, cmds, elapsed, mode)

    if args.timing:
        warn(DIM(f"  [{elapsed:.2f}s, {mode} mode]"))

    if not args.execute:
        return 0

    if _is_windows():
        warn("whatisit: --execute is disabled on Windows")
        return 7

    # ---- execution path ----
    # With several candidates on screen, do NOT assume #1. The numbering only
    # ranks confidence, and #1 is the greedy answer rather than a verified one;
    # picking silently would run something the user never chose.
    if len(cmds) > 1:
        if not sys.stdin.isatty():
            warn("whatisit: several candidates -- rerun without -n, or pick one yourself.")
            return 6
        try:
            pick = input(f"\n{BOLD('Run which?')} [1-{len(cmds)}, or N to cancel] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if not pick.isdigit() or not 1 <= int(pick) <= len(cmds):
            warn("whatisit: not running.")
            return 0
        chosen = cmds[int(pick) - 1]
    else:
        chosen = cmds[0]

    findings = check(chosen)
    danger = any(sev == "DANGER" for sev, _ in findings)

    if danger:
        warn(f"\n{RED('Refusing to auto-run a command flagged DANGER.')}")
        warn(DIM("Copy it and run it yourself if you are certain."))
        return 6

    if cfg.get("confirm_execute", True):
        if not sys.stdin.isatty():
            warn("whatisit: refusing to execute without an interactive confirmation.")
            return 6
        # Default stays explicit: "y"/"yes" is required. -y/--yes or
        # confirm_default=true opts into treating an empty answer as yes.
        empty_is_yes = args.yes or cfg.get("confirm_default", False)
        accept = ("", "y", "yes") if empty_is_yes else ("y", "yes")
        prompt = "[Y/n]" if empty_is_yes else "[y/N]"
        try:
            ans = input(f"\n{BOLD('Run this?')} {prompt} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if ans not in accept:
            warn("whatisit: not running.")
            return 0

    warn(DIM(f"$ {chosen}"))
    shell = os.environ.get("SHELL", "/bin/bash")
    return subprocess.run([shell, "-c", chosen]).returncode


def _confirm(prompt: str, auto: bool) -> bool:
    """Ask before spending someone's bandwidth. --auto is standing consent."""
    if auto:
        return True
    if not sys.stdin.isatty():
        warn(f"  {YELLOW('skipped')}: {prompt} (no terminal to ask; use --auto)")
        return False
    try:
        return input(f"  {prompt} [Y/n] ").strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _progress(got: int, total: int) -> None:
    if not total:
        return
    bar = got * 30 // total
    warn(f"\r    [{'#' * bar}{'.' * (30 - bar)}] {got * 100 // total:3d}%  "
         f"{fetch.fmt_size(got)} / {fetch.fmt_size(total)}", end="")
    if got >= total:
        warn()


def _model_targets(size: str, models_dir: Path):
    """(real file, the fixed slot name the engine looks for)."""
    spec = fetch.MODELS[size]
    return models_dir / spec["file"], models_dir / cfg_mod.MODEL_NAME


def cmd_setup_print_urls(args) -> int:
    """Everything needed to fetch by hand, for machines with no route out."""
    plan = fetch.runtime_plan()
    spec = fetch.MODELS[args.size]
    print(BOLD("whatisit setup --print-urls"))
    if plan["kind"] == "none":
        print(f"  runtime: {plan['reason']}")
        print(fetch.manual_instructions())
    else:
        digest = plan["sha256"]
        if plan["kind"] == "upstream":
            try:
                digest = fetch.release_digests(args.llama_version).get(
                    fetch.asset_name(args.llama_version))
            except fetch.FetchError:
                digest = None
        print("  runtime:")
        print(f"    url    {plan['url']}")
        print(f"    sha256 {digest or '(fetch from the GitHub release page)'}")
    print("  model:")
    print(f"    url    {fetch.model_url(args.size)}")
    print(f"    sha256 {spec['sha256']}")
    print(f"    size   {fetch.fmt_size(spec['size'])}")
    print("\n  Then, once both are on this machine:")
    print(f"    whatisit setup --model ./{spec['file']} --bin-dir ./llama/bin")
    return 0


def cmd_setup(args, cfg: dict) -> int:
    if getattr(args, "print_urls", False):
        return cmd_setup_print_urls(args)

    print(BOLD("whatisit setup"))
    models_dir = cfg_mod.data_dir() / "models"
    bin_dir = cfg_mod.data_dir() / "bin"

    # An explicit path means the manual path, which is unchanged. Otherwise
    # bare `setup` fetches what is missing.
    if not (args.model or args.bin_dir):
        return _setup_auto(args, cfg, models_dir, bin_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    target = models_dir / cfg_mod.MODEL_NAME
    # An explicit --model always wins. The target name is fixed, so without
    # this the existence check below matches the already-registered file and
    # switching models is a silent no-op that still reports success.
    if args.model and target.exists():
        src = Path(args.model).expanduser().resolve()
        if src.exists() and (not target.is_symlink() or
                             target.resolve() != src):
            target.unlink()
            print(f"  replacing registered model with {src.name}")
    if target.exists():
        shown = target.resolve().name if target.is_symlink() else target.name
        print(f"  model already present: {shown} "
              f"({target.stat().st_size / 1e6:.0f} MB)")
    elif args.model:
        src = Path(args.model).expanduser().resolve()
        if not src.exists():
            warn(f"  {RED('no such file')}: {src}")
            return 1
        # Symlink rather than copy by default: the model is ~1 GB and is
        # often already on shared or external storage. --copy overrides.
        if args.copy:
            print(f"  copying {src} -> {target} (~1 GB, please wait)")
            shutil.copy2(src, target)
        else:
            target.symlink_to(src)
            print(f"  linked model: {target} -> {src}")
    else:
        print(f"  {YELLOW('no model yet.')} Provide one with:")
        print(f"    whatisit setup --model /path/to/{cfg_mod.MODEL_NAME}")
        print(DIM("  (a released build would download it from the model hub here)"))

    for name, suffix in (("llama-server", "LLAMA_SERVER"),
                         ("llama-cli", "LLAMA_CLI")):
        dest = bin_dir / name
        src = cfg_mod.env(suffix) or (args.bin_dir and str(Path(args.bin_dir) / name))
        if dest.exists():
            print(f"  runtime present: {dest}")
        elif src and Path(src).exists():
            dest.symlink_to(Path(src).resolve())
            print(f"  linked runtime: {dest} -> {src}")

    return _finish_setup(cfg)


def _finish_setup(cfg: dict) -> int:
    cfg.setdefault("threads", 0)
    p = cfg_mod.save_config(cfg)
    print(f"  config written: {p}")
    print(f"  threads: {cfg_mod.resolve_threads(cfg)} "
          + DIM(f"(of {os.cpu_count()} cores; decode is memory-bound, not core-bound)"))
    print(f"\n{GREEN('Ready.')} Try:  whatisit list files changed this week")
    return 0


def _setup_auto(args, cfg: dict, models_dir: Path, bin_dir: Path) -> int:
    """Fetch whatever is missing, asking first and verifying afterwards."""
    spec = fetch.MODELS[args.size]
    model_file, slot = _model_targets(args.size, models_dir)
    server = bin_dir / "llama-server"

    want_model = not args.runtime_only
    want_runtime = not args.model_only

    need_model = want_model and not model_file.exists()
    need_runtime = want_runtime and not server.exists()

    if want_model and not need_model:
        print(f"  model present: {model_file.name} "
              f"({model_file.stat().st_size / 1e6:.0f} MB)")
    if want_runtime and not need_runtime:
        print(f"  runtime present: {server}")

    plan = {"kind": "none", "reason": "", "warn": "", "url": None,
            "sha256": None, "size": 0}
    if need_runtime:
        plan = fetch.runtime_plan()
        if plan["warn"]:
            out(f"  {YELLOW('note')}: {plan['warn']}")
        if plan["kind"] == "none":
            out(f"  {RED('no runtime available')}: {plan['reason']}")
            out(fetch.manual_instructions())
            if fetch.platform_key()[0] == "Linux":
                out(fetch.source_build_instructions())
            return 1
        if plan["kind"] == "compat":
            print(f"  {YELLOW('note')}: {plan['reason']}")
            print("  using the compatibility build instead")

        # Reusing an existing llama-server skips a download entirely.
        found = fetch.existing_llama_server()
        if found and _confirm(f"found {found} -- use it?", args.auto):
            bin_dir.mkdir(parents=True, exist_ok=True)
            for name in ("llama-server", "llama-cli"):
                src = found.parent / name
                dest = bin_dir / name
                if src.exists() and not dest.exists():
                    dest.symlink_to(src.resolve())
                    print(f"  linked runtime: {dest} -> {src}")
            need_runtime = False

    if not need_model and not need_runtime:
        if args.dry_run:
            print("  nothing to fetch")
            return 0
        return _finish_setup(cfg)

    bytes_needed = (spec["size"] if need_model else 0) + \
                   (fetch.RUNTIME_BYTES if need_runtime else 0)
    if args.dry_run:
        print("  would fetch:")
        if need_runtime:
            print(f"    runtime  {plan['url']}")
        if need_model:
            print(f"    model    {fetch.model_url(args.size)}  "
                  f"({fetch.fmt_size(spec['size'])})")
        print(f"  total about {fetch.fmt_size(bytes_needed)}; nothing was changed")
        return 0

    # Refuse before starting a download that cannot finish.
    free = fetch.free_bytes(cfg_mod.data_dir())
    if free < bytes_needed * 1.1:
        print(f"  {RED('not enough disk space')}: need about "
              f"{fetch.fmt_size(bytes_needed * 1.1)}, "
              f"{fetch.fmt_size(free)} free at {cfg_mod.data_dir()}",
              file=sys.stderr)
        return 1

    models_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    try:
        if need_runtime:
            _fetch_runtime(args, plan, bin_dir)
            if plan["kind"] == "compat":
                # That build predates UNIX-socket support in llama-server, so
                # the default transport would start and immediately fail to
                # bind. Recorded here rather than discovered at first query.
                cfg["force_tcp"] = True
                print("  note: this build serves over a local TCP port")
        if need_model:
            _fetch_model(args, spec, model_file, slot)
    except fetch.FetchError as e:
        warn(f"  {RED('failed')}: {e}")
        warn(fetch.manual_instructions())
        return 1

    return _finish_setup(cfg)


def _fetch_runtime(args, plan: dict, bin_dir: Path) -> None:
    sha = plan["sha256"]
    if plan["kind"] == "upstream":
        # Verify against GitHub's own published digest rather than a list
        # maintained here, which would go stale on every version bump.
        name = fetch.asset_name(args.llama_version)
        sha = fetch.release_digests(args.llama_version).get(name)
        if not sha:
            raise fetch.FetchError(f"no published checksum for {name}")
        url = fetch.asset_url(args.llama_version)
    else:
        url = plan["url"]
    if not url:
        raise fetch.FetchError("no published runtime for this platform")

    print(f"  llama.cpp runtime  ({fetch.fmt_size(plan['size'] or 0)})")
    if not _confirm("download it?", args.auto):
        raise fetch.FetchError("declined; nothing was downloaded")

    staging = cfg_mod.data_dir() / "runtime"
    archive = staging / url.rsplit("/", 1)[-1]
    fetch.download(url, archive, sha256=sha, progress=_progress)
    src_dir = fetch.extract_runtime(archive, staging / "unpacked")
    archive.unlink(missing_ok=True)

    # The binaries are dynamically linked against the .so files beside them and
    # find them through RUNPATH=$ORIGIN, so the directory has to stay whole.
    # Putting its contents directly in bin/ keeps that true and matches where
    # the engine already looks.
    for item in src_dir.iterdir():
        dest = bin_dir / item.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        shutil.move(str(item), str(dest))
    shutil.rmtree(staging / "unpacked", ignore_errors=True)
    print(f"  runtime installed: {bin_dir}")


def _fetch_model(args, spec: dict, model_file: Path, slot: Path) -> None:
    print(f"  model {spec['file']}  ({fetch.fmt_size(spec['size'])})")
    if not _confirm("download it?", args.auto):
        raise fetch.FetchError("declined; nothing was downloaded")
    fetch.download(fetch.model_url(args.size), model_file,
                   sha256=spec["sha256"], expected_size=spec["size"],
                   progress=_progress)
    print(f"  model installed: {model_file}")
    # The engine resolves a fixed slot name, so a non-default size still has to
    # be reachable through it.
    if slot != model_file:
        if slot.exists() or slot.is_symlink():
            slot.unlink()
        slot.symlink_to(model_file)
        print(f"  registered: {slot.name} -> {model_file.name}")


def _doctor_remote(cfg: dict, remote: dict) -> int:
    """doctor output for an OpenAI-compatible backend. Local llama.cpp is unused."""
    try:
        base = engine.normalize_endpoint_url(remote["base_url"])
    except ValueError as e:
        print(f"  {RED('FAIL')}  remote     {e}")
        print(f"  info  config     {cfg_mod.config_path()}")
        print(f"\n{RED('Not ready.')}")
        return 1
    model = remote["model"]
    ok = True
    print(f"  {GREEN('ok')}    backend    OpenAI-compatible  {base}")
    if model:
        print(f"  {GREEN('ok')}    model      {model}")
    else:
        print(f"  {RED('FAIL')}  model      not set -- set openai_model or WHATISIT_OPENAI_MODEL")
        ok = False
    print(f"  info  auth       {'authenticated' if remote['api_key'] else 'no API key'}")
    try:
        names = engine.list_remote_models(remote)
    except Exception as e:
        print(f"  {RED('FAIL')}  endpoint   could not reach it: {e}")
        ok = False
    else:
        if model and model not in names:
            print(f"  {YELLOW('warn')}  endpoint   reachable, but {model} is not among "
                  f"{len(names)} advertised model(s)")
        else:
            print(f"  {GREEN('ok')}    endpoint   reachable ({len(names)} model(s))")
    print(f"  info  config     {cfg_mod.config_path()}")
    print(f"\n{GREEN('All good.') if ok else RED('Not ready.')}")
    return 0 if ok else 1


def cmd_doctor(args, cfg: dict) -> int:
    print(BOLD("whatisit doctor"))

    # Remote mode means no local model or llama.cpp is required at all; the
    # local diagnostics below are irrelevant (and would wrongly report failure).
    remote = cfg_mod.remote_config(cfg)
    if remote is not None:
        return _doctor_remote(cfg, remote)

    ok = True

    model = cfg_mod.find_model()
    if model:
        # The registered path is a fixed slot name, so a 3B installed here still
        # sits at nl2sh-1.5b-....gguf. Report what the slot actually points at,
        # otherwise doctor names the wrong model.
        actual = model.resolve().name if model.is_symlink() else model.name
        suffix = f"  [{actual}]" if actual != model.name else ""
        print(f"  {GREEN('ok')}    model      {model} "
              f"({model.stat().st_size / 1e6:.0f} MB){suffix}")
    else:
        print(f"  {RED('FAIL')}  model      not found -- run `whatisit setup --model ...`")
        ok = False

    srv = cfg_mod.env("LLAMA_SERVER") or str(cfg_mod.data_dir() / "bin" / "llama-server")
    if Path(srv).exists():
        print(f"  {GREEN('ok')}    server     {srv}")
    else:
        print(f"  {YELLOW('warn')}  server     not found -- falling back to slow one-shot mode")

    cli = cfg_mod.find_llama_cli()
    print(f"  {GREEN('ok') if cli else RED('FAIL')}    cli        {cli or 'not found'}")
    ok = ok and bool(cli or Path(srv).exists())

    bundled = Path(__file__).resolve().parent.parent.parent / "runtime" / "lib"
    print(f"  {'ok' if bundled.is_dir() else 'warn'}    libs       "
          f"{bundled if bundled.is_dir() else 'using system libstdc++'}")

    port = engine.running_port()
    where = f"listening on {port}" if port else "not running"
    print(f"  {'ok' if port else 'idle'}  server pid {where}")
    print(f"  info  threads    {cfg_mod.resolve_threads(cfg)} (of {os.cpu_count()} cores)")
    print(f"  info  config     {cfg_mod.config_path()}")

    # Both directories present means the one-time move was skipped rather than
    # run: it never overwrites. Only worth saying here, where someone is
    # already asking why the install looks wrong.
    for kind in ("config", "data"):
        legacy = cfg_mod.legacy_dir(kind)
        current = cfg_mod.config_dir() if kind == "config" else cfg_mod.data_dir()
        if legacy.is_dir() and current.is_dir():
            print(f"  {YELLOW('warn')}  {kind:<10} {legacy} also exists and is unused; "
                  f"{current} is the live one")
    print(f"\n{GREEN('All good.') if ok else RED('Not ready.')}")
    return 0 if ok else 1


def cmd_stop(args, cfg: dict) -> int:
    print("whatisit: server stopped." if engine.stop_server() else "whatisit: no server running.")
    return 0


def cmd_config(args, cfg: dict) -> int:
    if args.set:
        for kv in args.set:
            if "=" not in kv:
                warn(f"whatisit: expected key=value, got {kv!r}")
                return 2
            k, v = kv.split("=", 1)
            if v.lower() in ("true", "false"):
                cfg[k] = v.lower() == "true"
            else:
                try:
                    cfg[k] = int(v)
                except ValueError:
                    try:
                        cfg[k] = float(v)
                    except ValueError:
                        cfg[k] = v
        print(f"whatisit: wrote {cfg_mod.save_config(cfg)}")
    for k in sorted(cfg):
        v = cfg[k]
        # Never print a stored API key; say it is set instead. Secrets belong in
        # the environment (WHATISIT_OPENAI_API_KEY), but if one is in the
        # config file it still must not be echoed back.
        if k == "openai_api_key" and v:
            v = "********"
        print(f"  {k} = {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="whatisit",
        description="Natural language to shell command, local by default. "
                    "Optionally, any OpenAI-compatible endpoint.",
        epilog="examples:\n"
               "  whatisit find files larger than 100MB in this folder\n"
               "  whatisit -n 3 'find files bigger than 100MB'\n"
               "  whatisit -e 'count lines in every python file'\n"
               "  eval \"$(whatisit -q 'show disk usage')\"",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="your request, in plain English")
    ap.add_argument("-n", "--num", type=int, default=1, metavar="N",
                    help="show N alternative commands (default 1)")
    ap.add_argument("-e", "--execute", action="store_true",
                    help="run the command after confirming (never auto-runs DANGER)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print only the bare command, for $(...) use")
    ap.add_argument("-t", "--timing", action="store_true", help="report latency")
    ap.add_argument("--oneshot", action="store_true",
                    help="bypass the resident server (slower; for debugging)")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="with -e, treat an empty confirm answer as yes ([Y/n])")
    ap.add_argument("--port", type=int, metavar="PORT",
                    help="fixed TCP port for the resident server (1-65535)")
    ap.add_argument("--threads", type=int, metavar="N",
                    help="override saved thread count for this invocation")
    ap.add_argument("--ctx-size", type=int, metavar="N",
                    help="override the saved context size for this invocation")
    ap.add_argument("--model", metavar="PATH",
                    help="override the registered model for this invocation")
    ap.add_argument("--host-context", dest="host_context", action="store_true",
                    help="enable host context for this invocation")
    ap.add_argument("--no-host-context", dest="host_context", action="store_false",
                    help="disable host context for this invocation")
    ap.add_argument("--grammar", dest="grammar", action="store_true",
                    help="enable the GBNF grammar for this invocation")
    ap.add_argument("--no-grammar", dest="grammar", action="store_false",
                    help="disable the GBNF grammar for this invocation")
    ap.add_argument("--debug", action="store_true",
                    help="print the exact prompt and grammar sent to the model")

    sub = ap.add_subparsers(dest="sub")
    s = sub.add_parser("setup", help="first-run setup: fetch the runtime and model")
    s.add_argument("--model", help="path to a GGUF model you already have")
    s.add_argument("--bin-dir", help="directory containing llama-server / llama-cli")
    s.add_argument("--copy", action="store_true", help="copy the model instead of symlinking")
    s.add_argument("--auto", action="store_true",
                   help="assume yes to every download; needs no terminal")
    s.add_argument("--size", choices=sorted(fetch.MODELS), default="1.5b",
                   help="which model to fetch (default 1.5b)")
    s.add_argument("--llama-version", metavar="bXXXX", default=fetch.LLAMA_BUILD,
                   help="pin a specific llama.cpp build")
    s.add_argument("--runtime-only", action="store_true", help="fetch only llama.cpp")
    s.add_argument("--model-only", action="store_true", help="fetch only the model")
    s.add_argument("--dry-run", action="store_true",
                   help="print what would be fetched and change nothing")
    s.add_argument("--print-urls", action="store_true",
                   help="print URLs and checksums, then exit (for offline machines)")
    s.set_defaults(func=cmd_setup)

    sub.add_parser("doctor", help="check the installation").set_defaults(func=cmd_doctor)
    sub.add_parser("stop", help="stop the resident model server").set_defaults(func=cmd_stop)
    c = sub.add_parser("config", help="show or change settings")
    c.add_argument("--set", nargs="+", metavar="K=V")
    c.set_defaults(func=cmd_config)
    return ap


SUBCOMMANDS = {"setup", "doctor", "stop", "config"}
_FLAGS_NOARG = {"-e", "--execute", "-q", "--quiet", "-t", "--timing", "--oneshot",
                "--host-context", "--no-host-context",
                "--grammar", "--no-grammar", "--debug", "-y", "--yes"}
_FLAGS_ARG = {"-n", "--num"}
_FLAGS_QUERY_ARG = {"--port", "--threads", "--ctx-size", "--model"}


class QueryArgs:
    """Hand-parsed query invocation.

    argparse cannot be used for the query path. Two reasons, both found by
    actually typing realistic requests:
      1. A subparser turns any trailing word that happens to name a
         subcommand into one -- `whatisit how do I stop a stuck process` is
         parsed as the `stop` subcommand.
      2. Real requests contain things that look like flags
         (`find files -name test`), which argparse would reject.
    So: consume flags only while they appear BEFORE the request text, then
    take everything remaining verbatim.
    """

    def __init__(self, argv: list[str]):
        self.num, self.execute, self.quiet = 1, False, False
        self.timing, self.oneshot = False, False
        self.port, self.threads, self.ctx_size, self.model = None, None, None, None
        self.host_context, self.grammar, self.debug, self.yes = None, None, False, False
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "--":                      # explicit end of flags
                i += 1
                break
            if a in _FLAGS_NOARG:
                if a == "--host-context":
                    self.host_context = True
                elif a == "--no-host-context":
                    self.host_context = False
                elif a == "--grammar":
                    self.grammar = True
                elif a == "--no-grammar":
                    self.grammar = False
                elif a in ("-y", "--yes"):
                    self.yes = True
                else:
                    setattr(self, {"-e": "execute", "--execute": "execute",
                                   "-q": "quiet", "--quiet": "quiet",
                                   "-t": "timing", "--timing": "timing",
                                   "--oneshot": "oneshot",
                                   "--debug": "debug"}[a], True)
            elif a in _FLAGS_ARG:
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} needs a number")
                self.num = int(argv[i + 1])
                i += 1
            elif a in _FLAGS_QUERY_ARG:
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} needs a value")
                val = argv[i + 1]
                if a == "--port":
                    self.port = int(val)
                    if not 1 <= self.port <= 65535:
                        raise ValueError(f"--port must be 1..65535, got {self.port}")
                elif a == "--threads":
                    self.threads = int(val)
                    if self.threads < 0:
                        raise ValueError(f"--threads must be >= 0, got {self.threads}")
                elif a == "--ctx-size":
                    self.ctx_size = int(val)
                    if self.ctx_size <= 0:
                        raise ValueError(f"--ctx-size must be > 0, got {self.ctx_size}")
                else:
                    self.model = val
                i += 1
            elif a.startswith("-n") and len(a) > 2 and a[2:].isdigit():
                self.num = int(a[2:])          # -n3
            else:
                break                          # request text starts here
            i += 1
        self.words = argv[i:]

        # A `--` anywhere in the request is an attempt to end the flags, not
        # part of the question. Drop the first one so `whatisit list files -- -e`
        # does what it looks like.
        explicit = "--" in self.words
        if explicit:
            cut = self.words.index("--")
            self.words = self.words[:cut] + self.words[cut + 1:]

        # Flags only count before the request text (see the docstring), so a
        # trailing `-e` silently becomes part of the question and the model
        # answers something odd. Record it so the caller can say so rather
        # than leaving the user to work it out. A `--` means the user already
        # said they meant it literally, so stay quiet in that case.
        known = _FLAGS_NOARG | _FLAGS_ARG | _FLAGS_QUERY_ARG
        self.stray_flags = [] if explicit else [w for w in self.words if w in known]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Before load_config, so an existing config.json is read from its new home.
    # Goes to stderr to keep `$(whatisit -q ...)` substitutions clean.
    cfg_mod.migrate_legacy_dirs(echo=warn)
    cfg = cfg_mod.load_config()

    if not argv:
        build_parser().print_help()
        return 0

    # A subcommand only counts as one when it is the very first token, so
    # `whatisit config` manages settings while `whatisit show me the git config`
    # stays a question.
    if argv[0] in SUBCOMMANDS:
        args = build_parser().parse_args(argv)
        return args.func(args, cfg)

    if argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    try:
        args = QueryArgs(argv)
    except ValueError as e:
        warn(f"whatisit: {e}")
        return 2
    if not args.words:
        build_parser().print_help()
        return 0
    return cmd_query(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
