from __future__ import annotations

import ast
import configparser
import inspect
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from streamline.p1_data_process.p1_runner import P1Runner
from streamline.p2_impute_scale.p2_runner import P2Runner
from streamline.p3_feature_learning.p3_runner import P3Runner
from streamline.p4_feature_importance.p4_runner import P4Runner
from streamline.p5_feature_selection.p5_runner import P5Runner
from streamline.p6_modeling.p6_runner import P6Runner
from streamline.p7_ensembles.p7_runner import P7Runner
from streamline.p8_summary_statistics.p8_runner import P8Runner
from streamline.p9_compare_datasets.p9_runner import P9Runner
from streamline.p10_replication.p10_runner import P10Runner
from streamline.p11_reporting.p11_runner import P11Runner
from streamline.p6_modeling.utils.loader import normalize_modeling_type
from streamline.utils.run_commands import save_phase_run_command, snapshot_effective_args


PHASE_ALIASES = {
    "p1": "p1_data_process",
    "p1_data": "p1_data_process",
    "p1_data_process": "p1_data_process",
    "p2": "p2_impute_scale",
    "p2_impute_scale": "p2_impute_scale",
    "p3": "p3_feature_learning",
    "p3_feature_learning": "p3_feature_learning",
    "p4": "p4_feature_importance",
    "p4_feature_importance": "p4_feature_importance",
    "p5": "p5_feature_selection",
    "p5_feature_selection": "p5_feature_selection",
    "p6": "p6_modeling",
    "p6_modeling": "p6_modeling",
    "p7": "p7_ensembles",
    "p7_ensembles": "p7_ensembles",
    "p8": "p8_summary_statistics",
    "p8_summary_statistics": "p8_summary_statistics",
    "p9": "p9_compare_datasets",
    "p9_compare_datasets": "p9_compare_datasets",
    "p10": "p10_replication",
    "p10_replication": "p10_replication",
    "p11": "p11_reporting",
    "p11_reporting": "p11_reporting",
}

DEFAULT_PHASE_ORDER = [
    "p1_data_process",
    "p2_impute_scale",
    "p3_feature_learning",
    "p4_feature_importance",
    "p5_feature_selection",
    "p6_modeling",
    "p7_ensembles",
    "p8_summary_statistics",
    "p9_compare_datasets",
    "p10_replication",
    "p11_reporting",
]

PHASE_RUNNERS = {
    "p1_data_process": P1Runner,
    "p2_impute_scale": P2Runner,
    "p3_feature_learning": P3Runner,
    "p4_feature_importance": P4Runner,
    "p5_feature_selection": P5Runner,
    "p6_modeling": P6Runner,
    "p7_ensembles": P7Runner,
    "p8_summary_statistics": P8Runner,
    "p9_compare_datasets": P9Runner,
    "p10_replication": P10Runner,
    "p11_reporting": P11Runner,
}

CONTROL_KEYS = {
    "enabled",
    "report_modes",
    "skip_for_outcome_types",
}

PHASE_TOGGLE_KEYS = {
    "p1_data_process": ("do_p1", "do_data_process", "do_eda"),
    "p2_impute_scale": ("do_p2", "do_impute_scale", "do_dataprep"),
    "p3_feature_learning": ("do_p3", "do_feature_learning"),
    "p4_feature_importance": ("do_p4", "do_feature_importance", "do_feat_imp"),
    "p5_feature_selection": ("do_p5", "do_feature_selection", "do_feat_sel"),
    "p6_modeling": ("do_p6", "do_modeling", "do_model"),
    "p7_ensembles": ("do_p7", "do_ensembles"),
    "p8_summary_statistics": ("do_p8", "do_summary_statistics", "do_stats"),
    "p9_compare_datasets": ("do_p9", "do_compare_datasets", "do_compare_dataset"),
    "p10_replication": ("do_p10", "do_replication", "do_replicate"),
    "p11_reporting": ("do_p11", "do_reporting", "do_report", "do_rep_report"),
}

PHASES_THROUGH_STANDARD_REPORT = {
    "p1_data_process",
    "p2_impute_scale",
    "p3_feature_learning",
    "p4_feature_importance",
    "p5_feature_selection",
    "p6_modeling",
    "p7_ensembles",
    "p8_summary_statistics",
    "p9_compare_datasets",
    "p11_reporting",
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    suffix = path.suffix.lower()
    if suffix not in {".cfg", ".ini"}:
        raise ValueError("STREAMLINE pipeline configs must end in .cfg or .ini")
    loaded = load_cfg_config(path)
    if not isinstance(loaded, dict):
        raise ValueError("STREAMLINE config root must be a mapping/object.")
    return expand_config_values(loaded)


def load_cfg_config(path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    with path.open() as file:
        parser.read_file(file)

    loaded: dict[str, Any] = {"run": {}, "phase_controls": {}, "phases": {}}
    for section in parser.sections():
        section_key = section.strip()
        normalized_section = section_key.lower()
        values = {
            key.strip(): parse_cfg_value(value)
            for key, value in parser.items(section)
        }
        if normalized_section in {"run", "global"}:
            loaded[normalized_section] = values
        elif normalized_section in {"phases", "phase_controls"}:
            loaded["phase_controls"].update(values)
        elif normalized_section in PHASE_ALIASES:
            loaded["phases"][normalized_section] = values
        else:
            raise ValueError(
                f"Unknown config section [{section_key}]. Use [run], [phases], or a phase section like [p1]."
            )
    return loaded


def parse_cfg_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def expand_config_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config_values(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def normalize_phase_name(name: str) -> str:
    normalized = PHASE_ALIASES.get(str(name).strip())
    if normalized is None:
        known = ", ".join(sorted(PHASE_ALIASES))
        raise ValueError(f"Unknown phase '{name}'. Known phases: {known}")
    return normalized


def normalize_phase_list(values: Any) -> list[str]:
    if values in (None, "", []):
        return []
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    return [normalize_phase_name(value) for value in values]


def parse_config_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping/object.")
    return dict(value)


def runner_constructor_defaults(runner_class) -> dict[str, Any]:
    signature = inspect.signature(runner_class.__init__)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is not inspect.Parameter.empty:
            kwargs[name] = deepcopy(parameter.default)
    return kwargs


def runner_kwargs(runner_class, common_config: dict[str, Any], phase_config: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(runner_class.__init__)
    allowed = {name for name in signature.parameters if name != "self"}
    kwargs = runner_constructor_defaults(runner_class)
    kwargs.update({key: value for key, value in common_config.items() if key in allowed})
    kwargs.update({key: value for key, value in phase_config.items() if key in allowed and key not in CONTROL_KEYS})
    return kwargs


class PipelineRunner:
    def __init__(
        self,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        *,
        dry_run: bool = False,
        start_at: str | None = None,
        stop_after: str | None = None,
        only: list[str] | str | None = None,
        skip: list[str] | str | None = None,
    ):
        if config is None:
            if config_path is None:
                raise ValueError("Provide config_path or config.")
            config = load_config(config_path)
        self.config_path = None if config_path is None else str(config_path)
        self.config = deepcopy(config)
        self.dry_run = bool(dry_run)
        self.start_at = normalize_phase_name(start_at) if start_at else None
        self.stop_after = normalize_phase_name(stop_after) if stop_after else None
        self.only = normalize_phase_list(only)
        self.skip = set(normalize_phase_list(skip))

        global_config = parse_config_mapping(self.config.get("global"), "global")
        run_config = parse_config_mapping(self.config.get("run"), "run")
        self.common_config = {**global_config, **run_config}
        self.phase_configs = parse_config_mapping(self.config.get("phases"), "phases")
        self.phase_controls = parse_config_mapping(self.config.get("phase_controls"), "phase_controls")

    def run(self) -> list[str]:
        phases = self.resolve_phase_order()
        completed = []
        logging.info("STREAMLINE pipeline phase order: %s", ", ".join(phases))
        for phase in phases:
            if not self.phase_is_enabled(phase):
                logging.info("Skipping disabled phase: %s", phase)
                continue
            if self.should_skip_regression_ensemble(phase):
                logging.info("Skipping Phase 7 because regression/continuous ensembles are not supported.")
                continue
            self.run_phase(phase)
            completed.append(phase)
        return completed

    def resolve_phase_order(self) -> list[str]:
        if self.only:
            phases = list(self.only)
        else:
            phases = normalize_phase_list(
                self.common_config.get("phase_order")
                or self.phase_controls.get("phase_order")
                or self.common_config.get("phases")
                or DEFAULT_PHASE_ORDER
            )
            if self.start_at:
                if self.start_at not in phases:
                    raise ValueError(f"start_at phase '{self.start_at}' is not present in the configured phase order.")
                phases = phases[phases.index(self.start_at):]
            if self.stop_after:
                if self.stop_after not in phases:
                    raise ValueError(f"stop_after phase '{self.stop_after}' is not present in the configured phase order.")
                phases = phases[: phases.index(self.stop_after) + 1]
        return [phase for phase in phases if phase not in self.skip]

    def phase_config(self, phase: str) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        aliases = [alias for alias, canonical in PHASE_ALIASES.items() if canonical == phase]
        for source in (self.config, self.phase_configs):
            for alias in aliases:
                value = source.get(alias)
                if isinstance(value, dict):
                    merged.update(value)
        return merged

    def phase_is_enabled(self, phase: str) -> bool:
        config = self.phase_config(phase)
        if "enabled" in config:
            return bool(config["enabled"])
        toggle = self.phase_toggle_value(phase)
        if toggle is not None:
            return bool(toggle)
        if phase == "p10_replication":
            return bool(config.get("rep_data_path") and config.get("dataset_for_rep"))
        return True

    def phase_toggle_value(self, phase: str) -> Any:
        for key in PHASE_TOGGLE_KEYS[phase]:
            if key in self.phase_controls:
                return self.phase_controls[key]
        if "do_all" in self.phase_controls:
            return self.phase_controls["do_all"]
        if "do_till_report" in self.phase_controls:
            if phase in PHASES_THROUGH_STANDARD_REPORT:
                return self.phase_controls["do_till_report"]
            if phase == "p10_replication":
                return self.phase_controls.get("do_replicate", False)
        return None

    def should_skip_regression_ensemble(self, phase: str) -> bool:
        if phase != "p7_ensembles":
            return False
        outcome_type = str(self.common_config.get("outcome_type", "")).lower()
        p6_config = self.phase_config("p6_modeling")
        p6_outcome_type = p6_config.get("outcome_type", self.common_config.get("outcome_type"))
        p6_model_type = p6_config.get("model_type", self.common_config.get("model_type"))
        model_type = normalize_modeling_type(outcome_type=p6_outcome_type, model_type=p6_model_type)
        return outcome_type in {"continuous", "regression"} or model_type == "Regression"

    def run_phase(self, phase: str) -> None:
        runner_class = PHASE_RUNNERS[phase]
        config = self.phase_config(phase)
        if phase == "p6_modeling":
            self.apply_p6_defaults(config)
        kwargs = runner_kwargs(runner_class, self.common_config, config)
        if phase == "p11_reporting":
            self.run_reporting(kwargs, config)
            return

        logging.info("Starting %s", phase)
        if self.dry_run:
            print(f"[dry-run] {phase}: {runner_class.__name__}({kwargs})")
            return
        runner = runner_class(**kwargs)
        runner.run()
        self.save_phase_run_arguments(phase, snapshot_effective_args(kwargs, runner))
        logging.info("Finished %s", phase)

    def experiment_root_from_kwargs(self, kwargs: dict[str, Any]) -> Path | None:
        experiment_path = kwargs.get("experiment_path")
        if experiment_path:
            return Path(str(experiment_path))
        output_path = kwargs.get("output_path")
        experiment_name = kwargs.get("experiment_name")
        if output_path and experiment_name:
            return Path(str(output_path)) / str(experiment_name)
        common_output = self.common_config.get("output_path")
        common_name = self.common_config.get("experiment_name")
        if common_output and common_name:
            return Path(str(common_output)) / str(common_name)
        return None

    def config_command_argv(self, phase: str) -> list[str]:
        argv = ["run.py"]
        if self.config_path:
            argv.extend(["--config", self.config_path])
        argv.extend(["--only", phase])
        return argv

    def save_phase_run_arguments(self, phase: str, kwargs: dict[str, Any]) -> None:
        exp_root = self.experiment_root_from_kwargs(kwargs)
        if exp_root is None:
            logging.info("run_commands.pickle not updated for %s: experiment path could not be resolved.", phase)
            return
        save_phase_run_command(
            exp_root=exp_root,
            phase=phase,
            args=dict(kwargs),
            argv=self.config_command_argv(phase),
        )

    def apply_p6_defaults(self, phase_config: dict[str, Any]) -> None:
        if "outcome_type" in phase_config:
            return
        outcome_type = self.common_config.get("outcome_type")
        if outcome_type:
            phase_config["outcome_type"] = outcome_type

    def run_reporting(self, kwargs: dict[str, Any], config: dict[str, Any]) -> None:
        modes = config.get("report_modes")
        if modes is None:
            report_mode = config.get("report_mode", kwargs.get("report_mode", "standard"))
            modes = [report_mode]
        if isinstance(modes, str):
            modes = [item.strip() for item in modes.split(",") if item.strip()]
        for mode in modes:
            mode_kwargs = dict(kwargs)
            mode_kwargs["report_mode"] = mode
            logging.info("Starting p11_reporting (%s)", mode)
            if self.dry_run:
                print(f"[dry-run] p11_reporting: P11Runner({mode_kwargs})")
                continue
            runner = P11Runner(**mode_kwargs)
            runner.run()
            self.save_phase_run_arguments("p11_reporting", snapshot_effective_args(mode_kwargs, runner))
            logging.info("Finished p11_reporting (%s)", mode)
