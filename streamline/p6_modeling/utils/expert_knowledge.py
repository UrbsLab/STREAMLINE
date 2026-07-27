from __future__ import annotations

import glob
import inspect
import logging
import os

from streamline.p6_modeling.utils.categorical import normalize_model_id


EXPERT_KNOWLEDGE_PARAMETER_NAMES = (
    "expert_knowledge",
    "expertknowledge",
    "expert_param",
    "expertparam",
    "expertKnowledge",
)
EXPERT_KNOWLEDGE_SOURCE_PRIORITY = (
    "multiswrfdb",
    "multiswrfdbstar",
    "multisurf",
    "multisurfstar",
    "mutualinformation",
)


def model_constructor_parameter_names(ModelCls):
    signature = inspect.signature(ModelCls.__init__)
    return {
        name for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def model_class_label(ModelCls) -> str:
    small = getattr(ModelCls, "small_name", "")
    name = getattr(ModelCls, "model_name", "")
    if small and name:
        return f"{small} ({name})"
    return small or name or str(ModelCls)


def model_expert_knowledge_parameter_name(ModelCls):
    constructor_names = model_constructor_parameter_names(ModelCls)
    for parameter_name in EXPERT_KNOWLEDGE_PARAMETER_NAMES:
        if parameter_name in constructor_names:
            return parameter_name
    if getattr(ModelCls, "uses_expertparam", False):
        logging.warning(
            "[P6] %s marks uses_expertparam=True but has no supported expert "
            "knowledge constructor parameter.",
            model_class_label(ModelCls),
        )
    return None


def normalize_expert_knowledge_source_name(value):
    return normalize_model_id(value).replace("*", "star").replace("_", "")


def find_expert_knowledge_score_file(dataset_dir: str, cv_idx: int):
    feature_importance_dir = os.path.join(dataset_dir, "feature_importance")
    if not os.path.isdir(feature_importance_dir):
        return None

    score_paths = glob.glob(
        os.path.join(feature_importance_dir, "*", f"*_scores_cv_{int(cv_idx)}.csv")
    )
    if not score_paths:
        return None

    priority = {
        normalize_expert_knowledge_source_name(source): index
        for index, source in enumerate(EXPERT_KNOWLEDGE_SOURCE_PRIORITY)
    }

    def sort_key(path):
        parent_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path).split("_scores_cv_")[0]
        parent_key = normalize_expert_knowledge_source_name(parent_name)
        file_key = normalize_expert_knowledge_source_name(file_name)
        source_rank = min(priority.get(parent_key, 999), priority.get(file_key, 999))
        return source_rank, parent_key, file_key, path

    return sorted(score_paths, key=sort_key)[0]


def find_cv_training_file(dataset_dir: str, cv_idx: int):
    dataset_name = os.path.basename(dataset_dir.rstrip("/"))
    expected = os.path.join(
        dataset_dir,
        "CVDatasets",
        f"{dataset_name}_CV_{int(cv_idx)}_Train.csv",
    )
    if os.path.exists(expected):
        return expected

    candidates = sorted(glob.glob(os.path.join(
        dataset_dir,
        "CVDatasets",
        f"*_CV_{int(cv_idx)}_Train.csv",
    )))
    return candidates[0] if candidates else None


def load_cv_feature_names(dataset_dir: str, outcome_label: str, instance_label: str | None, cv_idx: int):
    train_path = find_cv_training_file(dataset_dir, cv_idx)
    if train_path is None:
        return None

    try:
        import pandas as pd
        columns = pd.read_csv(train_path, nrows=0).columns.tolist()
    except Exception as exc:
        logging.warning("[P6] Could not read CV feature columns from %s: %s", train_path, exc)
        return None

    excluded = {outcome_label}
    if instance_label:
        excluded.add(instance_label)
    return [column for column in columns if column not in excluded]


def load_expert_knowledge_scores(dataset_dir: str, outcome_label: str, instance_label: str | None, cv_idx: int):
    score_path = find_expert_knowledge_score_file(dataset_dir, cv_idx)
    if score_path is None:
        return None

    feature_names = load_cv_feature_names(dataset_dir, outcome_label, instance_label, cv_idx)
    if not feature_names:
        return None

    try:
        import pandas as pd
        scores_df = pd.read_csv(score_path)
    except Exception as exc:
        logging.warning("[P6] Could not read expert knowledge scores from %s: %s", score_path, exc)
        return None

    if "feature" not in scores_df.columns or "score" not in scores_df.columns:
        logging.warning(
            "[P6] Expert knowledge score file %s must include feature and score columns.",
            score_path,
        )
        return None

    score_values = pd.to_numeric(scores_df["score"], errors="coerce").fillna(0.0)
    score_map = dict(zip(scores_df["feature"].astype(str), score_values.astype(float)))
    expert_knowledge = [float(score_map.get(feature, 0.0)) for feature in feature_names]

    missing_scores = [feature for feature in feature_names if feature not in score_map]
    if missing_scores:
        logging.warning(
            "[P6] Expert knowledge file %s is missing %d/%d modeled features; "
            "using 0.0 for missing scores.",
            score_path,
            len(missing_scores),
            len(feature_names),
        )

    logging.info(
        "[P6] Loaded %d expert knowledge scores from %s.",
        len(expert_knowledge),
        score_path,
    )
    return expert_knowledge
