from __future__ import annotations

import argparse
import copy
import logging
import pickle
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set

RUN_COMMANDS_FILENAME = "run_commands.pickle"
SCHEMA_VERSION = 1

_CONTROL_DESTS = {
    "ignore_saved_run_command",
    "no_update_saved_run_command",
    "update_saved_run_command",
}


def add_run_command_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ignore_saved_run_command",
        action="store_true",
        help="Ignore saved arguments from run_commands.pickle for this phase.",
    )
    parser.add_argument(
        "--no_update_saved_run_command",
        action="store_true",
        help="Do not update run_commands.pickle after this phase finishes.",
    )


def apply_saved_run_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    phase: str,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    if getattr(args, "ignore_saved_run_command", False):
        return args

    exp_root = resolve_experiment_root(args)
    if exp_root is None:
        logging.warning("run_commands.pickle not used for %s: experiment path could not be resolved.", phase)
        return args

    stored = load_phase_run_command(exp_root, phase)
    if not stored:
        logging.warning(
            "No saved run command found for %s at %s; using parser/metadata defaults.",
            phase,
            exp_root / RUN_COMMANDS_FILENAME,
        )
        return args

    saved_args = stored.get("args") or {}
    if not isinstance(saved_args, dict):
        return args

    explicit = _explicit_dests(parser, argv if argv is not None else sys.argv[1:])
    for key, value in saved_args.items():
        if key in _CONTROL_DESTS or key in explicit or not hasattr(args, key):
            continue
        setattr(args, key, copy.deepcopy(value))
    return args


def snapshot_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        k: copy.deepcopy(v)
        for k, v in vars(args).items()
        if k not in _CONTROL_DESTS
    }


def save_run_command_from_args(
    args: argparse.Namespace,
    phase: str,
    args_snapshot: Optional[Dict[str, Any]] = None,
    argv: Optional[Sequence[str]] = None,
) -> None:
    if getattr(args, "no_update_saved_run_command", False):
        return
    exp_root = resolve_experiment_root(args)
    if exp_root is None:
        logging.info("run_commands.pickle not updated for %s: experiment path could not be resolved.", phase)
        return
    save_phase_run_command(
        exp_root=exp_root,
        phase=phase,
        args=args_snapshot if args_snapshot is not None else snapshot_args(args),
        argv=argv if argv is not None else sys.argv[1:],
    )


def require_args(parser: argparse.ArgumentParser, args: argparse.Namespace, names: Iterable[str]) -> None:
    missing = []
    for name in names:
        value = getattr(args, name, None)
        if value is None or value == "":
            missing.append(f"--{name}")
    if missing:
        parser.error(
            "Missing required arguments after applying run_commands.pickle: "
            + ", ".join(missing)
        )


def resolve_experiment_root(args: argparse.Namespace) -> Optional[Path]:
    experiment_path = getattr(args, "experiment_path", None)
    if experiment_path:
        return Path(experiment_path)

    output_path = getattr(args, "output_path", None)
    experiment_name = getattr(args, "experiment_name", None)
    if output_path and experiment_name:
        return Path(output_path) / experiment_name

    return None


def load_run_commands(exp_root: Path) -> Dict[str, Any]:
    path = exp_root / RUN_COMMANDS_FILENAME
    if not path.exists():
        return _empty_store()
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        logging.warning("Could not read %s: %s. Starting with an empty command store.", path, exc)
        return _empty_store()

    if not isinstance(payload, dict):
        return _empty_store()
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("phases", {})
    return payload


def load_phase_run_command(exp_root: Path, phase: str) -> Dict[str, Any]:
    store = load_run_commands(exp_root)
    phase_store = store.get("phases", {}).get(phase, {})
    if not isinstance(phase_store, dict):
        return {}
    latest = phase_store.get("latest", {})
    return latest if isinstance(latest, dict) else {}


def save_phase_run_command(
    exp_root: Path,
    phase: str,
    args: Dict[str, Any],
    argv: Optional[Sequence[str]] = None,
) -> None:
    exp_root.mkdir(parents=True, exist_ok=True)
    store = load_run_commands(exp_root)
    phases = store.setdefault("phases", {})
    phase_store = phases.setdefault(phase, {})
    history = phase_store.setdefault("history", [])

    record = {
        "phase": phase,
        "updated_at": datetime.now().isoformat(),
        "args": copy.deepcopy(args),
        "argv": list(argv or []),
        "command": " ".join(shlex.quote(str(part)) for part in (argv or [])),
    }
    phase_store["latest"] = record
    history.append(record)

    path = exp_root / RUN_COMMANDS_FILENAME
    with path.open("wb") as f:
        pickle.dump(store, f)


def _empty_store() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "phases": {}}


def _explicit_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> Set[str]:
    option_to_dest: Dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit: Set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit
