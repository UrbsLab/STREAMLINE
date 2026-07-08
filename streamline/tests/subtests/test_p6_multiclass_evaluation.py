import numpy as np

from streamline.p6_modeling.utils.submodels import MulticlassClassificationModel


class _DummyMulticlassEstimator:
    def __init__(self):
        self.classes_ = np.array([0, 1, 2])
        self._proba = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.75, 0.15],
                [0.05, 0.10, 0.85],
                [0.15, 0.65, 0.20],
                [0.80, 0.10, 0.10],
                [0.10, 0.20, 0.70],
            ]
        )

    def predict(self, x_test):
        return self.classes_[np.argmax(self._proba[: len(x_test)], axis=1)]

    def predict_proba(self, x_test):
        return self._proba[: len(x_test)]


class _DummyMulticlassModel(MulticlassClassificationModel):
    def __init__(self):
        super().__init__(model=None, model_name="Dummy Multiclass")
        self.model = _DummyMulticlassEstimator()

    def objective(self, trial, params=None):
        raise NotImplementedError


def test_multiclass_model_evaluation_supports_macro_micro_average_precision():
    model = _DummyMulticlassModel()
    x_test = np.zeros((6, 2))
    y_test = np.array([0, 1, 2, 1, 0, 2])

    metrics_dict, curves_dict = model.model_evaluation(x_test, y_test)

    assert metrics_dict["average_precision_macro"] is not None
    assert metrics_dict["average_precision_micro"] is not None
    assert metrics_dict["roc_auc_macro"] is not None
    assert metrics_dict["roc_auc_micro"] is not None
    assert set(curves_dict["roc"]) == {"micro", "macro"}
    assert set(curves_dict["prc"]) == {"micro", "macro"}
