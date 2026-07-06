import pickle
import sys
import types

import numpy as np
import pandas as pd

from streamline.p4_feature_importance.importance import FeatureImportance
import streamline.p4_feature_importance.p4_runner as p4_runner_module
from streamline.p4_feature_importance.p4_runner import P4Runner
from streamline.p4_feature_importance.registry.multisurf import MultiSURF
from streamline.p4_feature_importance.registry.multisurfstar import MultiSURFStar
from streamline.p4_feature_importance.registry.multiswrfdb import MultiSWRFDB
from streamline.p5_feature_selection.registry.default import DefaultFeatureSelector


def install_fake_skrebate(monkeypatch):
    module = types.ModuleType("skrebate")
    module.created = []
    module.fit_arrays = []
    module.turf_created = []

    class FakeRebate:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.rank_absolute = kwargs.get("rank_absolute", False)
            module.created.append((self.__class__.__name__, kwargs))

        def fit(self, X, y):
            module.fit_arrays.append(np.array(X, copy=True))
            self.feature_importances_ = np.arange(X.shape[1], dtype=float)
            self.top_features_ = np.argsort(self.feature_importances_)[::-1]
            return self

    class MultiSURF(FakeRebate):
        pass

    class MultiSURFstar(FakeRebate):
        pass

    class MultiSWRFDB(FakeRebate):
        pass

    class MultiSWRFDBstar(FakeRebate):
        pass

    class TURF:
        def __init__(self, relief_object, pct=0.5, num_scores_to_return=100):
            self.relief_object = relief_object
            self.pct = pct
            self.num_scores_to_return = num_scores_to_return
            module.turf_created.append(
                {
                    "relief_class": relief_object.__class__.__name__,
                    "pct": pct,
                    "num_scores_to_return": num_scores_to_return,
                }
            )

        def fit(self, X, y):
            self.relief_object.fit(X, y)
            self.feature_importances_ = np.arange(X.shape[1], dtype=float) + 10.0
            self.top_features_ = np.argsort(self.feature_importances_)[::-1]
            return self

    module.MultiSURF = MultiSURF
    module.MultiSURFstar = MultiSURFstar
    module.MultiSWRFDB = MultiSWRFDB
    module.MultiSWRFDBstar = MultiSWRFDBstar
    module.TURF = TURF
    monkeypatch.setitem(sys.modules, "skrebate", module)
    return module


def write_cv_fixture(tmp_path):
    exp_root = tmp_path / "Experiment"
    dataset_dir = exp_root / "Dataset"
    cv_dir = dataset_dir / "CVDatasets"
    exploratory_dir = dataset_dir / "exploratory"
    cv_dir.mkdir(parents=True)
    exploratory_dir.mkdir(parents=True)

    train = pd.DataFrame(
        {
            "Class": [0, 1, 0, 1],
            "InstanceID": ["i1", "i2", "i3", "i4"],
            "quant": [0.1, 0.2, 0.3, 0.4],
            "cat_a": ["low", "high", "low", "medium"],
            "cat_b": [1, 2, 1, 3],
            "other": [4.0, 5.0, 6.0, 7.0],
        }
    )
    test = train.copy()
    train_path = cv_dir / "Dataset_CV_0_Train.csv"
    test_path = cv_dir / "Dataset_CV_0_Test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    with open(exploratory_dir / "categorical_features.pickle", "wb") as f:
        pickle.dump(["cat_a", "cat_b"], f)
    return exp_root, train_path, test_path


def test_rebate_receives_streamline_categorical_feature_indexes(monkeypatch, tmp_path):
    fake_skrebate = install_fake_skrebate(monkeypatch)
    exp_root, train_path, test_path = write_cv_fixture(tmp_path)

    FeatureImportance(
        cv_train_path=str(train_path),
        cv_test_path=str(test_path),
        experiment_path=str(exp_root),
        model_id="multiswrfdb",
        model_params={"categorical_features": [0], "n_jobs": 2},
        outcome_type="Binary",
        instance_label="InstanceID",
        random_state=3,
    ).run()

    created_name, kwargs = fake_skrebate.created[-1]
    assert created_name == "MultiSWRFDB"
    assert kwargs["categorical_features"] == [1, 2]
    assert kwargs["label_type"] == "binary"
    assert kwargs["n_jobs"] == 2
    assert np.isfinite(fake_skrebate.fit_arrays[-1][:, 1]).all()

    score_path = exp_root / "Dataset" / "feature_importance" / "multiswrfdb" / "multiswrfdb_scores_cv_0.csv"
    assert score_path.exists()
    with open(exp_root / "Dataset" / "feature_importance" / "multiswrfdb" / "selector_cv0.pickle", "rb") as f:
        payload = pickle.load(f)
    assert payload["categorical_features"] == ["cat_a", "cat_b"]
    assert payload["categorical_feature_indices"] == [1, 2]
    assert (exp_root / "Dataset" / "runtime" / "runtime_feature_importance_multiswrfdb_cv0.txt").exists()


def test_rebate_turf_uses_full_feature_count_by_default(monkeypatch, tmp_path):
    fake_skrebate = install_fake_skrebate(monkeypatch)
    exp_root, train_path, test_path = write_cv_fixture(tmp_path)

    FeatureImportance(
        cv_train_path=str(train_path),
        cv_test_path=str(test_path),
        experiment_path=str(exp_root),
        model_id="MultiSWRFDB*",
        model_params={"use_turf": True, "turf_pct": 0.25},
        outcome_type="Binary",
        instance_label="InstanceID",
    ).run()

    assert fake_skrebate.turf_created[-1] == {
        "relief_class": "MultiSWRFDBstar",
        "pct": 0.25,
        "num_scores_to_return": 4,
    }
    assert (
        exp_root / "Dataset" / "feature_importance" / "multiswrfdbstar" / "multiswrfdbstar_scores_cv_0.csv"
    ).exists()


def test_rebate_wrappers_accept_string_categorical_indexes():
    model = MultiSWRFDB(categorical_features=["1", "bad"])
    model.columns = ["quant", "cat"]
    X = pd.DataFrame({"quant": [1.0, 2.0], "cat": ["a", "b"]})

    X_array = model.normalize_rebate_matrix(X)

    assert X_array.shape == (2, 2)
    assert np.isfinite(X_array[:, 1]).all()


def test_multisurf_wrappers_do_not_pass_neighbor_param_to_skrebate():
    assert "n_neighbors" not in MultiSURF(n_neighbors=7).build_rebate_params(3)
    assert "n_neighbors" not in MultiSURFStar(n_neighbors=7).build_rebate_params(3)


def test_p4_runner_applies_rebate_n_jobs_defaults_for_active_models(tmp_path):
    output_path = tmp_path / "out"
    exp_root = output_path / "DemoExp"
    exp_root.mkdir(parents=True)

    runner = P4Runner(
        output_path=str(output_path),
        experiment_name="DemoExp",
        models="mutualinformation,multiswrfdb,multiswrfdbstar",
    )

    assert runner.models_params["multiswrfdb"]["n_jobs"] == 1
    assert runner.models_params["multiswrfdbstar"]["n_jobs"] == 1
    assert "mutualinformation" not in runner.models_params


def test_p4_runner_defaults_to_all_registered_importance_models(monkeypatch, tmp_path):
    output_path = tmp_path / "out"
    exp_root = output_path / "DemoExp"
    exp_root.mkdir(parents=True)

    monkeypatch.setattr(
        p4_runner_module,
        "list_importances",
        lambda: {"z_model": object, "a_model": object},
    )
    monkeypatch.setattr(p4_runner_module, "resolve_importance_id", lambda model_id: model_id)

    runner = P4Runner(output_path=str(output_path), experiment_name="DemoExp")

    assert runner.models == ["a_model", "z_model"]


def test_default_feature_selector_can_cap_more_than_two_algorithms():
    selector = DefaultFeatureSelector(export_scores=False)
    selected = {
        "mutualinformation": [["a"]],
        "multiswrfdb": [["b"]],
        "multiswrfdbstar": [["c"]],
    }
    ranks = {
        "mutualinformation": [["a", "b", "c"]],
        "multiswrfdb": [["b", "c", "a"]],
        "multiswrfdbstar": [["c", "a", "b"]],
    }

    cv_selected, informative, uninformative = selector._select_union_cap(
        selected,
        3,
        ranks,
        ["mutualinformation", "multiswrfdb", "multiswrfdbstar"],
        1,
    )

    assert cv_selected == [["a", "b", "c"]]
    assert informative == [3]
    assert uninformative == [0]
