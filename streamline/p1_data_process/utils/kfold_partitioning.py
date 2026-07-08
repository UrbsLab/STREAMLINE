import os
from typing import Optional, Tuple, List

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold


class KFoldPartitioner:
    """
    K-fold CV partitioner that operates on a pandas.DataFrame only.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset including the outcome column (and optional match column).
    experiment_path : str
        Directory where CV splits are saved.
    dataset_name : str
        Name used in output filenames and directories.
    outcome_label : str, default="Class"
        Column name for the outcome/target.
    match_label : str, optional
        Grouping column name used only when partition_method="Group".
    partition_method : {"Random", "Stratified", "Group"}, default="Stratified"
        The CV splitting strategy.
    n_splits : int, default=10
        Number of folds.
    random_state : int, optional
        RNG seed for reproducibility.
    """

    SUPPORTED_METHODS = ("Random", "Stratified", "Group")

    def __init__(
        self,
        data: pd.DataFrame,
        experiment_path: str,
        dataset_name: str,
        outcome_label: str = "Class",
        match_label: Optional[str] = None,
        partition_method: str = "Stratified",
        n_splits: int = 10,
        random_state: Optional[int] = None,

    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame.")
        if not outcome_label:
            raise ValueError("`outcome_label` is required.")
        if partition_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown partition method '{partition_method}'. Choose from {self.SUPPORTED_METHODS}.")

        # Column checks
        if outcome_label not in data.columns:
            raise ValueError(f"Outcome column '{outcome_label}' not found in data.")
        if partition_method == "Group":
            if match_label is None:
                raise ValueError("partition_method='Group' requires `match_label`.")
            if match_label not in data.columns:
                raise ValueError(f"Match column '{match_label}' not found in data.")

        self.data = data
        self.outcome_label = outcome_label
        self.match_label = match_label
        self.name = dataset_name

        self.partition_method = partition_method
        self.experiment_path = experiment_path
        self.n_splits = int(n_splits)
        self.random_state = random_state

        self.train_dfs: Optional[List[pd.DataFrame]] = None
        self.test_dfs: Optional[List[pd.DataFrame]] = None
        self.cv = None  # sklearn splitter instance


    # -------------------------
    # Main Run Function
    # -------------------------

    def run(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Convenience wrapper to generate and save splits; returns (train_dfs, test_dfs)."""
        train_dfs, test_dfs = self.cv_partitioner(return_dfs=True, save_dfs=True)
        return train_dfs, test_dfs


    # -------------------------
    # Helpers
    # -------------------------
    def feature_only_data(self) -> pd.DataFrame:
        """Return features-only DataFrame (drops outcome and optional match columns if present)."""
        drop_cols = [self.outcome_label]
        if self.match_label is not None and self.match_label in self.data.columns:
            drop_cols.append(self.match_label)
        return self.data.drop(columns=[c for c in drop_cols if c in self.data.columns], errors="ignore")

    def make_splitter(self):
        if self.partition_method == "Random":
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        if self.partition_method == "Stratified":
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        if self.partition_method == "Group":
            return StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # unreachable given validation
        raise RuntimeError("Unexpected partition method")

    # -------------------------
    # Public API
    # -------------------------
    def cv_partitioner(
        self,
        return_dfs: bool = True,
        save_dfs: bool = True,
        partition_method: Optional[str] = None,
    ) -> Tuple[Optional[List[pd.DataFrame]], Optional[List[pd.DataFrame]]]:
        """
        Create CV splits.

        Parameters
        ----------
        return_dfs : bool, default=True
            If True, store and return (train_dfs, test_dfs); otherwise returns (None, None).
        save_dfs : bool, default=True
            If True, write CSVs to {experiment_path}/{dataset_name}/CVDatasets.
        partition_method : str, optional
            Override the initialized method (must still be one of SUPPORTED_METHODS).

        Returns
        -------
        (train_dfs, test_dfs) or (None, None)
        """
        if partition_method:
            if partition_method not in self.SUPPORTED_METHODS:
                raise ValueError(f"Unknown partition method '{partition_method}'.")
            if partition_method == "Group" and self.match_label is None:
                raise ValueError("partition_method='Group' requires `match_label`.")
            self.partition_method = partition_method

        self.cv = self.make_splitter()

        x = self.feature_only_data()
        y = self.data[self.outcome_label]
        groups = self.data[self.match_label] if (self.partition_method == "Group") else None

        train_dfs: List[pd.DataFrame] = []
        test_dfs: List[pd.DataFrame] = []

        if return_dfs:
            if self.partition_method == "Group":
                for tr_idx, te_idx in self.cv.split(x, y, groups):
                    train_dfs.append(self.data.iloc[tr_idx, :])
                    test_dfs.append(self.data.iloc[te_idx, :])
            else:
                for tr_idx, te_idx in self.cv.split(x, y):
                    train_dfs.append(self.data.iloc[tr_idx, :])
                    test_dfs.append(self.data.iloc[te_idx, :])

            self.train_dfs = train_dfs
            self.test_dfs = test_dfs

        if save_dfs:
            self.save_datasets(self.experiment_path, self.train_dfs, self.test_dfs)

        return (self.train_dfs, self.test_dfs) if return_dfs else (None, None)

    def save_datasets(
        self,
        experiment_path: Optional[str] = None,
        train_dfs: Optional[List[pd.DataFrame]] = None,
        test_dfs: Optional[List[pd.DataFrame]] = None,
    ) -> None:
        """Save train/test folds as CSV files."""
        experiment_path = experiment_path or self.experiment_path

        if train_dfs is None or test_dfs is None:
            if self.train_dfs is None or self.test_dfs is None:
                if self.cv is None:
                    # Build splits on the fly
                    self.cv_partitioner(return_dfs=True, save_dfs=False)
                train_dfs, test_dfs = self.train_dfs, self.test_dfs
            else:
                train_dfs, test_dfs = self.train_dfs, self.test_dfs

        out_dir = os.path.join(experiment_path, self.name, "CVDatasets")
        os.makedirs(out_dir, exist_ok=True)

        for i, df in enumerate(train_dfs or []):
            df.to_csv(os.path.join(out_dir, f"{self.name}_CV_{i}_Train.csv"), index=False)

        for i, df in enumerate(test_dfs or []):
            df.to_csv(os.path.join(out_dir, f"{self.name}_CV_{i}_Test.csv"), index=False)
