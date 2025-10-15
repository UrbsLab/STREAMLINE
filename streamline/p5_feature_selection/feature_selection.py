from __future__ import annotations
import os, time
from streamline.p5_feature_selection.utils.fs_loader import load_strategy

class FeatureSelection:
    """
    Phase 5: loads a selection strategy from registry (default: 'default')
    and delegates the whole selection process.
    """

    def __init__(
        self,
        *,
        dataset_dir: str,                 # <output>/<experiment>/<dataset>
        n_splits: int,
        algorithms: "list[str] | str",
        class_label: str = "Class",
        instance_label: "str | None" = None,
        max_features_to_keep: int = 2000,
        filter_poor_features: bool = True,
        overwrite_cv: bool = False,
        # strategy selection
        selector_id: str = "default",
        selector_params: "dict | None" = None,
        # plotting / summary (forwarded to default strategy)
        export_scores: bool = True,
        top_features: int = 20,
        show_plots: bool = False,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = os.path.basename(dataset_dir.rstrip("/"))
        self.n_splits = int(n_splits)
        self.algorithms = self._csv_to_list(algorithms)
        self.class_label = class_label
        self.instance_label = instance_label
        self.max_features_to_keep = int(max_features_to_keep)
        self.filter_poor_features = bool(filter_poor_features)
        self.overwrite_cv = bool(overwrite_cv)
        self.selector_id = selector_id or "default"
        self.selector_params = selector_params or {}
        # convenience forwarding defaults
        self.selector_params.setdefault("export_scores", bool(export_scores))
        self.selector_params.setdefault("top_features", int(top_features))
        self.selector_params.setdefault("show_plots", bool(show_plots))
        self.job_start_time = time.time()

    def run(self):
        strat = load_strategy(self.selector_id, **self.selector_params)
        strat.select(
            dataset_dir=self.dataset_dir,
            dataset_name=self.dataset_name,
            n_splits=self.n_splits,
            algorithms=self.algorithms,
            class_label=self.class_label,
            instance_label=self.instance_label,
            max_features_to_keep=self.max_features_to_keep,
            filter_poor_features=self.filter_poor_features,
            overwrite_cv=self.overwrite_cv,
        )
        self._save_runtime(); self._complete_flag()

    # ---- helpers ----
    def _save_runtime(self):
        rt_dir = os.path.join(self.dataset_dir, "runtime"); os.makedirs(rt_dir, exist_ok=True)
        with open(os.path.join(rt_dir, "runtime_featureselection.txt"), "w") as f:
            f.write(str(time.time() - self.job_start_time))

    def _complete_flag(self):
        exp_dir = os.path.dirname(self.dataset_dir.rstrip("/"))
        os.makedirs(os.path.join(exp_dir, "jobsCompleted"), exist_ok=True)
        with open(os.path.join(exp_dir, f"jobsCompleted/job_featureselection_{self.dataset_name}.txt"), "w") as f:
            f.write("complete")

    @staticmethod
    def _csv_to_list(v):
        if v is None: return []
        if isinstance(v, list): return v
        if isinstance(v, str): return [x.strip() for x in v.split(",") if x.strip()]
        return list(v)
