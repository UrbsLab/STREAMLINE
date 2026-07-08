from abc import ABC

from streamline.p6_modeling.utils.submodels import BinaryClassificationModel

try:
    # pip install wittgenstein
    from wittgenstein import RIPPER
except ImportError as e:
    raise ImportError(
        "RIPPERClassifier requires the 'wittgenstein' package. "
        "Install it with `pip install wittgenstein`."
    ) from e


class RIPPERClassifier(BinaryClassificationModel, ABC):
    model_name = "RIPPER"
    small_name = "RIPPER"
    color = "darkred"

    def __init__(self, cv_folds=3, scoring_metric='balanced_accuracy',
                 metric_direction='maximize', random_state=None, cv=None, n_jobs=None,
                 k=None, prune_size=None):
        """
        STREAMLINE wrapper for the RIPPER rule learner using the
        'wittgenstein' implementation.

        Parameters exposed for tuning mirror the core RIPPER knobs:

        - k:           loss ratio parameter (class imbalance / rule preference)
        - prune_size:  fraction of data used for pruning
        """
        super().__init__(RIPPER, "RIPPER", cv_folds, scoring_metric, metric_direction, random_state, cv)

        # Modest search space (categorical grid) similar to eLCS/XCS/ExSTraCS style.
        self.param_grid = {
            'k': [1.0, 2.0, 3.0],            # typical range for Ripper-style learners
            'prune_size': [0.25, 0.33, 0.5],  # 0.33 is a common default
        }

        # Optional overrides (same pattern as iterations/N/nu in your LCS models)
        if k is not None:
            self.param_grid['k'] = [k]
        if prune_size is not None:
            self.param_grid['prune_size'] = [prune_size]

        # Keep random_state consistent with other models
        self.param_grid['random_state'] = [random_state]

        self.small_name = "RIPPER"
        self.color = "darkred"
        self.n_jobs = n_jobs

    def objective(self, trial, params=None):
        """
        Optuna objective for RIPPER.

        RIPPER follows a sklearn-like API, so we only need to set
        constructor params; feature_names etc. are not required here.
        """
        self.params = {
            'k': trial.suggest_categorical('k', self.param_grid['k']),
            'prune_size': trial.suggest_categorical('prune_size', self.param_grid['prune_size']),
            'random_state': trial.suggest_categorical('random_state', self.param_grid['random_state']),
        }

        mean_cv_score = self.hyper_eval()
        return mean_cv_score
