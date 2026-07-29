from __future__ import annotations

import pandas as pd

from streamline.p6_modeling.modeling import create_model_instance
from streamline.p6_modeling.utils.expert_knowledge import (
    find_expert_knowledge_score_file,
    load_expert_knowledge_scores,
)


class DummyExpertKnowledgeModel:
    small_name = "DummyEK"
    model_name = "Dummy Expert Knowledge"
    uses_expertparam = True

    def __init__(
        self,
        random_state=None,
        n_jobs=None,
        scoring_metric=None,
        metric_direction=None,
        expert_knowledge=None,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.expert_knowledge = expert_knowledge


class DummyExpertParamModel:
    small_name = "DummyEP"
    model_name = "Dummy Expert Param"
    uses_expertparam = True

    def __init__(
        self,
        random_state=None,
        n_jobs=None,
        scoring_metric=None,
        metric_direction=None,
        expertparam=None,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring_metric = scoring_metric
        self.metric_direction = metric_direction
        self.expertparam = expertparam


def make_dataset_with_feature_importance(tmp_path):
    dataset_dir = tmp_path / "Experiment" / "demo_dataset"
    cv_dir = dataset_dir / "CVDatasets"
    multiswrfdb_dir = dataset_dir / "feature_importance" / "multiswrfdb"
    mutualinformation_dir = dataset_dir / "feature_importance" / "mutualinformation"
    cv_dir.mkdir(parents=True)
    multiswrfdb_dir.mkdir(parents=True)
    mutualinformation_dir.mkdir(parents=True)

    train_df = pd.DataFrame(
        {
            "InstanceID": ["i1", "i2"],
            "feature_b": [1, 0],
            "feature_a": [0, 1],
            "feature_c": [3, 4],
            "Class": [0, 1],
        }
    )
    train_df.to_csv(cv_dir / "demo_dataset_CV_0_Train.csv", index=False)

    pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b", "feature_c"],
            "score": [0.2, 0.8, 0.1],
        }
    ).to_csv(multiswrfdb_dir / "multiswrfdb_scores_cv_0.csv", index=False)

    pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b", "feature_c"],
            "score": [9.0, 9.0, 9.0],
        }
    ).to_csv(mutualinformation_dir / "mutualinformation_scores_cv_0.csv", index=False)

    return dataset_dir


def test_phase6_loads_expert_knowledge_in_cv_feature_order(tmp_path):
    dataset_dir = make_dataset_with_feature_importance(tmp_path)

    score_path = find_expert_knowledge_score_file(str(dataset_dir), 0)
    assert score_path.endswith("multiswrfdb_scores_cv_0.csv")

    expert_knowledge = load_expert_knowledge_scores(
        str(dataset_dir),
        outcome_label="Class",
        instance_label="InstanceID",
        cv_idx=0,
    )

    assert expert_knowledge == [0.8, 0.2, 0.1]


def test_phase6_passes_expert_knowledge_to_models_with_expert_parameter(tmp_path):
    dataset_dir = make_dataset_with_feature_importance(tmp_path)

    model = create_model_instance(
        DummyExpertKnowledgeModel,
        random_state=17,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        model_params={},
        dataset_dir=str(dataset_dir),
        outcome_label="Class",
        instance_label="InstanceID",
        cv_idx=0,
    )

    assert model.expert_knowledge == [0.8, 0.2, 0.1]


def test_phase6_supports_expertparam_spelling_and_manual_override(tmp_path):
    dataset_dir = make_dataset_with_feature_importance(tmp_path)

    auto_model = create_model_instance(
        DummyExpertParamModel,
        random_state=17,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        model_params={},
        dataset_dir=str(dataset_dir),
        outcome_label="Class",
        instance_label="InstanceID",
        cv_idx=0,
    )
    assert auto_model.expertparam == [0.8, 0.2, 0.1]

    override_model = create_model_instance(
        DummyExpertParamModel,
        random_state=17,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        model_params={"dummyep": {"expertparam": [1.0, 2.0, 3.0]}},
        dataset_dir=str(dataset_dir),
        outcome_label="Class",
        instance_label="InstanceID",
        cv_idx=0,
    )
    assert override_model.expertparam == [1.0, 2.0, 3.0]

    alias_override_model = create_model_instance(
        DummyExpertParamModel,
        random_state=17,
        scoring_metric="balanced_accuracy",
        metric_direction="maximize",
        model_params={"dummyep": {"expert_knowledge": [4.0, 5.0, 6.0]}},
        dataset_dir=str(dataset_dir),
        outcome_label="Class",
        instance_label="InstanceID",
        cv_idx=0,
    )
    assert alias_override_model.expertparam == [4.0, 5.0, 6.0]
