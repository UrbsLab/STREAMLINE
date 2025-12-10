# dataprocess.py  (your refactored DataProcess file)

import copy
import csv
import os
import time
import pickle
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from streamline.p1_data_process.utils.kfold_partitioning import KFoldPartitioner
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, f_oneway, kruskal, spearmanr
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import warnings

from streamline.p1_data_process.utils.validators import find_cv_pairs, validate_cv_pair
from streamline.p1_data_process.utils.features_meta import build_feature_meta, save_feature_meta

warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

sns.set_theme()


class DataProcess:
    """
    STREAMLINE Phase-1 (DataFrame-only). Heavy plots are disabled by default;
    produce CSV artifacts first, then use render_plots_from_artifacts.py to plot later.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        experiment_path: str,
        outcome_label: str,
        match_label: "str | None" = None,
        instance_label: "str | None" = None,
        ignore_features=None,
        categorical_features=None,
        quantitative_features=None,
        exclude_eda_output=None,
        categorical_cutoff: int = 10,
        sig_cutoff: float = 0.05,
        featureeng_missingness: float = 0.5,
        cleaning_missingness: float = 0.5,
        correlation_removal_threshold: float = 1.0,
        partition_method: str = "Stratified",
        n_splits: int = 10,
        one_hot_encoding: bool = True,
        cv_provided: bool = False,
        cv_input_path: "str | None" = None,
        random_state: "int | None" = None,
        show_plots: bool = False,
        dataset_name: str = "dataset",

        # NEW: plotting flags (all default False)
        enable_plots: bool = False,
        plot_missingness: bool = False,
        plot_class_counts: bool = False,
        plot_correlation: bool = False,
        correlation_plot_max_features: int = 200,
        plot_univariate: bool = False,
        univariate_top_k: int = 20,
        plot_anomalies: bool = False,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame.")
        if outcome_label not in data.columns:
            raise ValueError(f"Outcome column '{outcome_label}' not found in data.")

        self.data = data.copy()
        self.experiment_path = experiment_path
        self.name = dataset_name

        # labels
        self.outcome_label = outcome_label
        self.match_label = match_label if (match_label in self.data.columns) else None
        self.instance_label = instance_label if (instance_label in self.data.columns) else None

        # outcome type
        n_unique = self.data[self.outcome_label].nunique()
        if n_unique == 2:
            self.outcome_type = "Binary"
        elif 2 < n_unique <= categorical_cutoff:
            self.outcome_type = "Multiclass"
        else:
            self.outcome_type = "Continuous"

        # keep explorations (CSV-producing analyses)
        explorations_list = ["Describe", "Univariate Analysis", "Feature Correlation"]
        if exclude_eda_output is not None:
            known_exclude_options = ['describe_csv', 'correlation']
            for x in exclude_eda_output:
                if x not in known_exclude_options:
                    logging.warning("Unknown EDA exclusion option %s", x)
            if 'describe_csv' in exclude_eda_output and "Describe" in explorations_list:
                explorations_list.remove("Describe")
            if 'correlation' in exclude_eda_output and "Feature Correlation" in explorations_list:
                explorations_list.remove("Feature Correlation")
            # we no longer treat *_plots here; plotting is controlled only by `plots` below

        self.explorations = explorations_list

        # ignore features
        if ignore_features is None:
            self.ignore_features = []
        elif isinstance(ignore_features, str):
            self.ignore_features = list(pd.read_csv(ignore_features, sep=',').iloc[:, 0])
        elif isinstance(ignore_features, list):
            self.ignore_features = ignore_features
        else:
            raise ValueError("`ignore_features` must be None, path to CSV, or list of strings.")

        # user-specified categorical/quantitative
        if categorical_features is None:
            self.specified_categorical = None
        elif isinstance(categorical_features, str) and categorical_features != '':
            self.specified_categorical = list(pd.read_csv(categorical_features, sep=',').iloc[:, 0])
        elif isinstance(categorical_features, list):
            self.specified_categorical = list(categorical_features)
        elif categorical_features == '':
            self.specified_categorical = None
        else:
            raise ValueError("`categorical_features` must be None, path, list, or ''.")

        if quantitative_features is None:
            self.specified_quantitative = None
        elif isinstance(quantitative_features, str) and quantitative_features != '':
            self.specified_quantitative = list(pd.read_csv(quantitative_features, sep=',').iloc[:, 0])
        elif isinstance(quantitative_features, list):
            self.specified_quantitative = list(quantitative_features)
        elif quantitative_features == '':
            self.specified_quantitative = None
        else:
            raise ValueError("`quantitative_features` must be None, path, list, or ''.")

        # state
        self.quantitative_features: list[str] = []
        self.categorical_features: list[str] = []
        self.engineered_features: list[str] = []
        self.one_hot_features: list[str] = []
        self.categorical_cutoff = int(categorical_cutoff)
        self.featureeng_missingness = float(featureeng_missingness)
        self.cleaning_missingness = float(cleaning_missingness)
        self.correlation_removal_threshold = correlation_removal_threshold
        self.sig_cutoff = float(sig_cutoff)

        # NEW: plotting controls
        self.enable_plots = bool(enable_plots)
        self.plot_missingness = bool(plot_missingness)
        self.plot_class_counts = bool(plot_class_counts)
        self.plot_correlation = bool(plot_correlation)
        self.correlation_plot_max_features = int(correlation_plot_max_features)
        self.plot_univariate = bool(plot_univariate)
        self.univariate_top_k = int(univariate_top_k)
        self.plot_anomalies = bool(plot_anomalies)

        self.show_plots = bool(show_plots)  # interactive display toggle

        # CV config
        self.cv_partitioner = None
        self.partition_method = partition_method
        self.n_splits = int(n_splits)
        self.one_hot_encoding = bool(one_hot_encoding)
        self.cv_provided = bool(cv_provided)
        self.cv_input_path = cv_input_path

        self.random_state = random_state
        self.job_start_time = None  # runtime

    # ----------------------------
    # Main flow
    # ----------------------------
    def run(self, top_features=20):
        self.job_start_time = time.time()

        if self.cv_provided:
            self.run_process(top_features)
            self.import_user_cv()
        else:
            self.run_process(top_features)
            self.cv_partitioner = KFoldPartitioner(
                data=self.data,
                partition_method=self.partition_method,
                experiment_path=self.experiment_path,
                n_splits=self.n_splits,
                random_state=self.random_state,
                outcome_label=self.outcome_label,
                match_label=self.match_label,
                dataset_name=self.name,
            )
            self.cv_partitioner.run()

        self.save_runtime()

    def run_process(self, top_features=20):
        random.seed(self.random_state); np.random.seed(self.random_state)
        self.make_log_folders()

        if self.match_label is None or self.match_label not in self.data.columns:
            self.match_label = None
            self.partition_method = 'Stratified'
            logging.warning("Specified 'match_label' not found; defaulting to Stratified CV.")

        self.identify_feature_types()

        logging.info("Running Initial EDA:")
        self.initial_eda(initial='initial/')

        self.data_manipulation_steps(top_features)

        # self.anomaly_detection()

        self.second_eda(top_features)

    def data_manipulation_steps(self, top_features=20):
        self.set_original_headers()

        # Transition table
        if self.outcome_type == "Binary":
            cols = ['Instances','Total Features','Categorical Features','Quantitative Features',
                    'Missing Values','Missing Percent','Class 0','Class 1']
        elif self.outcome_type == "Multiclass":
            n_class = len(self.counts_summary(save=False)) - 6
            cols = ['Instances','Total Features','Categorical Features','Quantitative Features',
                    'Missing Values','Missing Percent'] + [f'Class {i}' for i in range(n_class)]
        else:
            cols = ['Instances','Total Features','Categorical Features','Quantitative Features',
                    'Missing Values','Missing Percent']
        transition_df = pd.DataFrame(columns=cols)

        transition_df.loc["Original"] = self.counts_summary(save=False)

        self.label_encoder()
        self.drop_ignored_rowcols()
        transition_df.loc["C1"] = self.counts_summary(save=False)

        self.feature_engineering()
        transition_df.loc["E1"] = self.counts_summary(save=False)

        self.drop_invariant()
        self.feature_removal()
        transition_df.loc["C2"] = self.counts_summary(save=False)

        self.instance_removal()
        transition_df.loc["C3"] = self.counts_summary(save=False)

        self.categorical_feature_encoding_pandas()
        transition_df.loc["E2"] = self.counts_summary(save=False)

        if (self.correlation_removal_threshold is not None
                and self.correlation_removal_threshold <= 1
                and "Feature Correlation" in self.explorations):
            self.drop_highly_correlated_features()
        transition_df.loc["C4"] = self.counts_summary(save=False)

        self.set_processed_headers()
        transition_df.to_csv(os.path.join(self.experiment_path, self.name, 'exploratory', 'DataProcessSummary.csv'),
                             index=True)

        with open(os.path.join(self.experiment_path, self.name, 'exploratory', 'categorical_features.pickle'), 'wb') as f:
            pickle.dump(self.categorical_features, f)
        with open(os.path.join(self.experiment_path, self.name, 'exploratory', 'post_processed_features.pickle'), 'wb') as f:
            pickle.dump(list(self.data.columns), f)

    # ----------------------------
    # Inlined helpers
    # ----------------------------
    def feature_only_data(self):
        drop_cols = [self.outcome_label]
        if self.instance_label and self.instance_label in self.data.columns:
            drop_cols.append(self.instance_label)
        if self.match_label and self.match_label in self.data.columns:
            drop_cols.append(self.match_label)
        return self.data.drop(columns=[c for c in drop_cols if c in self.data.columns], errors="ignore")

    def non_feature_data(self):
        cols = [self.outcome_label]
        if self.instance_label and self.instance_label in self.data.columns:
            cols.append(self.instance_label)
        if self.match_label and self.match_label in self.data.columns:
            cols.append(self.match_label)
        return self.data[cols]

    def get_outcome(self):
        return self.data[self.outcome_label]

    def clean_data(self, ignore_features: "list[str] | None"):
        self.data = self.data.dropna(axis=0, how='any', subset=[self.outcome_label]).reset_index(drop=True)
        try:
            if self.outcome_type in ("Binary", "Multiclass"):
                self.data[self.outcome_label] = self.data[self.outcome_label].astype('int8')
        except Exception:
            pass
        if ignore_features:
            self.data = self.data.drop(ignore_features, axis=1, errors='ignore')

    def get_headers(self):
        headers = list(self.data.columns)
        for c in [self.outcome_label, self.match_label, self.instance_label]:
            if c and c in headers:
                headers.remove(c)
        return headers

    def set_original_headers(self, phase: str = "exploratory", initial: str = "initial/"):
        path = os.path.join(self.experiment_path, self.name, phase)
        os.makedirs(path, exist_ok=True)

        headers = self.get_headers()

        # Represent headers as a single-row DataFrame to preserve original layout
        df = pd.DataFrame([headers])

        out_path = os.path.join(path, f"{initial}OriginalFeatureNames.csv")
        df.to_csv(out_path, index=False, header=False)

        return headers


    def set_processed_headers(self, phase: str = "exploratory", initial: str = ""):
        path = os.path.join(self.experiment_path, self.name, phase)
        os.makedirs(path, exist_ok=True)

        headers = self.get_headers()

        df = pd.DataFrame([headers])

        out_path = os.path.join(path, f"{initial}ProcessedFeatureNames.csv")
        df.to_csv(out_path, index=False, header=False)

        return headers

    def describe_data(self, initial=''):
        path = os.path.join(self.experiment_path, self.name, 'exploratory')
        os.makedirs(path, exist_ok=True)
        self.data.describe().to_csv(os.path.join(path, initial + 'DescribeDataset.csv'))
        self.data.dtypes.to_csv(os.path.join(path, initial + 'DtypesDataset.csv'),
                                header=['DataType'], index_label='Variable')
        self.data.nunique().to_csv(os.path.join(path, initial + 'NumUniqueDataset.csv'),
                                   header=['Count'], index_label='Variable')

    def missingness_counts(self, initial='', save=True):
        missing_count = self.data.isnull().sum()
        total_missing = int(missing_count.sum())
        if save:
            path = os.path.join(self.experiment_path, self.name, 'exploratory')
            os.makedirs(path, exist_ok=True)
            missing_count.to_csv(os.path.join(path, initial + 'DataMissingness.csv'),
                                 header=['Count'], index_label='Variable')
        return total_missing

    def missing_count_plot(self, initial=''):
        """PNG only if enabled."""
        if not (self.enable_plots and self.plot_missingness):
            return
        path = os.path.join(self.experiment_path, self.name, 'exploratory')
        os.makedirs(path, exist_ok=True)
        missing_count = self.data.isnull().sum()
        plt.hist(missing_count, bins=100)
        plt.xlabel("Missing Value Counts")
        plt.ylabel("Frequency")
        plt.title("Histogram of Missing Value Counts in Dataset")
        plt.savefig(os.path.join(path, initial + 'DataMissingnessHistogram.png'), bbox_inches='tight')
        if self.show_plots:
            plt.show()
        plt.close('all')

    def feature_correlation(self, x_data: "pd.DataFrame | None" = None, initial=''):
        """Always writes CSV; PNG heatmap only if enabled and within feature limit."""
        if x_data is None:
            x_data = self.feature_only_data()
        path = os.path.join(self.experiment_path, self.name, 'exploratory')
        os.makedirs(path, exist_ok=True)

        corr = x_data.corr(method='pearson', numeric_only=True)
        corr.to_csv(os.path.join(path, initial + 'FeatureCorrelations.csv'))

        # plot if allowed and not too wide
        if self.enable_plots and self.plot_correlation and x_data.shape[1] <= self.correlation_plot_max_features:
            import numpy as np
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            num_features = len(x_data.columns)
            sns.set_style("white")
            fig_size = (max(6, num_features // 2), max(6, num_features // 2))
            plt.subplots(figsize=fig_size)
            sns.heatmap(corr, mask=mask, vmax=1, vmin=-1, square=True, cmap='RdBu', cbar_kws={"shrink": .75})
            plt.savefig(os.path.join(path, initial + 'FeatureCorrelations.png'), bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close('all')
            sns.set_theme()


    def import_user_cv(self):
        if not self.cv_input_path:
            raise Exception("cv_input_path must be provided when cv_provided=True")

        ds_root = os.path.join(self.experiment_path, self.name)
        cv_src = os.path.join(self.cv_input_path, "CVDatasets")
        cv_dst = os.path.join(ds_root, "CVDatasets")

        if not os.path.isdir(cv_src):
            raise Exception(f"Expected CVDatasets/ under: {self.cv_input_path}")

        os.makedirs(cv_dst, exist_ok=True)

        pairs = find_cv_pairs(cv_src)
        if not pairs:
            raise Exception(f"No complete Train/Test fold pairs found under {cv_src}")

        outcome = self.outcome_label
        instance = self.instance_label

        for fold, files in pairs.items():
            df_tr = pd.read_csv(files["Train"], na_values="NA")
            df_te = pd.read_csv(files["Test"], na_values="NA")
            validate_cv_pair(df_tr, df_te, outcome, instance)

            out_train = os.path.join(cv_dst, f"{self.name}_CV_{fold}_Train.csv")
            out_test = os.path.join(cv_dst, f"{self.name}_CV_{fold}_Test.csv")
            df_tr.to_csv(out_train, index=False)
            df_te.to_csv(out_test, index=False)

            idx_path = os.path.join(ds_root, f"cv_index_cv{fold}.csv")
            pd.DataFrame({"split": "Train", "index": df_tr.index}).to_csv(idx_path, index=False, mode="w")
            pd.DataFrame({"split": "Test", "index": df_te.index}).to_csv(idx_path, index=False, mode="a", header=False)

        # try feature_meta
        try:
            feature_meta = {
                "dataset_name": self.name,
                "outcome_label": self.outcome_label,
                "match_label": self.match_label,
                "instance_label": self.instance_label,
                "categorical_features": list(self.categorical_features),
                "quantitative_features": list(self.quantitative_features),
                "one_hot": self.one_hot_encoding,
                "one_hot_features": list(self.one_hot_features),
                "engineered_features": list(self.engineered_features),
                "columns": list(self.data.columns),
            }
            save_feature_meta(self.experiment_path, self.name, feature_meta)
        except Exception as e:
            logging.warning(f"Feature meta not saved via util; ({e})")

        with open(os.path.join(ds_root, "phase_done.json"), "w") as f:
            f.write('{"phase":"p1_data","mode":"import_user_cv"}')

    def make_log_folders(self):
        base = os.path.join(self.experiment_path, self.name)
        os.makedirs(os.path.join(base), exist_ok=True)
        os.makedirs(os.path.join(base, 'exploratory'), exist_ok=True)
        os.makedirs(os.path.join(base, 'exploratory', 'anomaly_detection'), exist_ok=True)
        os.makedirs(os.path.join(base, 'exploratory', 'initial'), exist_ok=True)

    def identify_feature_types(self, x_data: "pd.DataFrame | None" = None):
        logging.info("Validating and Identifying Feature Types...")

        if self.specified_categorical is not None:
            self.specified_categorical = [s.strip() for s in self.specified_categorical]
        if self.specified_quantitative is not None:
            self.specified_quantitative = [s.strip() for s in self.specified_quantitative]

        if self.specified_quantitative is not None and self.specified_categorical is not None:
            dup = list(set(self.specified_categorical) & set(self.specified_quantitative))
            if dup:
                raise Exception("Features specified as both categorical and quantitative: " + str(dup))
            logging.warning("Both cat/quant lists provided; binaries treated as categorical; remaining auto-assigned.")

        if self.specified_quantitative is None and self.specified_categorical is None:
            logging.warning("No user lists; auto-assign based on cutoff and dtype.")

        if x_data is None:
            x_data = self.feature_only_data()

        headers = list(x_data.columns)
        if self.specified_categorical is not None:
            self.specified_categorical = [f for f in self.specified_categorical if f in headers]
        if self.specified_quantitative is not None:
            self.specified_quantitative = [f for f in self.specified_quantitative if f in headers]

        # binaries => categorical
        binary_categoricals_dict = {}
        for col in headers:
            unique_vals = x_data[col].dropna().unique().tolist()
            if len(unique_vals) == 2:
                if str(x_data[col].dtype) != 'object':
                    binary_categoricals_dict[col] = unique_vals
                self.categorical_features.append(col)
                if self.specified_quantitative and col in self.specified_quantitative:
                    self.specified_quantitative.remove(col)

        with open(os.path.join(self.experiment_path, self.name, 'exploratory', 'binary_categorical_dict.pickle'), 'wb') as f:
            pickle.dump(binary_categoricals_dict, f)

        if self.specified_categorical is not None and self.specified_quantitative is None:
            logging.warning("Only categorical list provided; others quantitative unless binary.")
            self.categorical_features = list(set(self.categorical_features + self.specified_categorical))
            self.quantitative_features = list(set(self.get_headers()) - set(self.categorical_features))

        if self.specified_quantitative is not None and self.specified_categorical is None:
            logging.warning("Only quantitative list provided; others categorical.")
            self.quantitative_features = list(self.specified_quantitative)
            self.categorical_features = list(set(self.get_headers()) - set(self.quantitative_features))

        if self.specified_quantitative is not None and self.specified_categorical is not None:
            self.quantitative_features = list(set(self.specified_quantitative))
            self.categorical_features = list(set(self.categorical_features + self.specified_categorical))

        # auto assign remaining
        for col in headers:
            if col not in self.categorical_features and col not in self.quantitative_features:
                if x_data[col].nunique() <= self.categorical_cutoff or not pd.api.types.is_numeric_dtype(x_data[col]):
                    self.categorical_features.append(col)
                else:
                    self.quantitative_features.append(col)

        init_dir = os.path.join(self.experiment_path, self.name, 'exploratory', 'initial')
        os.makedirs(init_dir, exist_ok=True)
        with open(os.path.join(init_dir, 'initial_categorical_features.pickle'), 'wb') as f:
            pickle.dump(self.categorical_features, f)
        with open(os.path.join(init_dir, 'initial_quantitative_features.pickle'), 'wb') as f:
            pickle.dump(self.quantitative_features, f)
        with open(os.path.join(init_dir, 'initial_categorical_features.csv'), 'w', newline="") as f:
            csv.writer(f).writerow(self.categorical_features)
        with open(os.path.join(init_dir, 'initial_quantitative_features.csv'), 'w', newline="") as f:
            csv.writer(f).writerow(self.quantitative_features)

    def counts_summary(self, total_missing=None, save=True, replicate=False):
        f_count = self.data.shape[1] - 1
        if self.instance_label is not None and self.instance_label in self.data.columns:
            f_count -= 1
        if self.match_label is not None and self.match_label in self.data.columns:
            f_count -= 1

        if total_missing is None:
            total_missing = self.missingness_counts(save=False)
        percent_missing = int(total_missing) / float(self.data.shape[0] * max(1, f_count))

        summary = [
            ['instances', self.data.shape[0]],
            ['features', f_count],
            ['categorical_features', len(self.categorical_features)],
            ['quantitative_features', len(self.quantitative_features)],
            ['missing_values', total_missing],
            ['missing_percent', round(percent_missing, 5)]
        ]
        summary_df = pd.DataFrame(summary, columns=['Variable', 'Count'])
        class_counts = self.data[self.outcome_label].value_counts()

        if save:
            out_dir = os.path.join(self.experiment_path, self.name, 'exploratory')
            os.makedirs(out_dir, exist_ok=True)
            summary_df.to_csv(os.path.join(out_dir, 'DataCounts.csv'), index=False)

            if self.outcome_type in ("Binary", "Multiclass"):
                df_value_counts = pd.DataFrame(class_counts).reset_index()
                df_value_counts.columns = ['Class', 'Instances']
                class_counts.to_csv(os.path.join(out_dir, 'ClassCounts.csv'), header=['Count'], index_label='Class')
            else:
                df_value_counts = pd.DataFrame(class_counts).reset_index()
                df_value_counts.columns = ['Top Occurring Values', 'Counts']
                class_counts.to_csv(os.path.join(out_dir, 'ClassCounts.csv'), header=['Count'], index_label='Label')
                logging.info("Skewness: %s", str(skew(self.data[self.outcome_label])))
                logging.info("Kurtosis: %s", str(kurtosis(self.data[self.outcome_label])))

            if not replicate:
                logging.info("Categorical: %s", self.categorical_features)
                logging.info("Quantitative: %s", self.quantitative_features)

            # PNG only if enabled
            if self.enable_plots and self.plot_class_counts:
                if self.outcome_type in ("Binary", "Multiclass"):
                    class_counts.plot(kind='bar')
                    plt.ylabel('Count')
                    plt.title('Class Counts')
                else:
                    plt.figure()
                    plt.hist(self.data[self.outcome_label], bins=100)
                    plt.ylabel('Count'); plt.xlabel('Label'); plt.title('Label Counts')
                plt.savefig(os.path.join(out_dir, 'ClassCountsBarPlot.png'), bbox_inches='tight')
                if self.show_plots: plt.show()
                plt.close('all')

        if self.outcome_type == "Binary":
            return list(summary_df['Count']) + [class_counts.get(0, 0), class_counts.get(1, 0)]
        elif self.outcome_type == "Multiclass":
            class_counts_list = [class_counts[i] for i in class_counts.index]
            return list(summary_df['Count']) + class_counts_list
        else:
            return list(summary_df['Count'])

    def label_encoder(self):
        string_cols = [feat for feat, typ in self.data.dtypes.to_dict().items()
                       if str(typ) == 'object' and (self.instance_label is None or feat != self.instance_label)]

        ord_label = pd.DataFrame(columns=['Category', 'Encoding'])
        if len(string_cols) > 0:
            logging.info("Ordinal encoding textual features...")
            for feat in string_cols:
                if feat in self.quantitative_features \
                        and not (feat == self.outcome_label or (self.match_label and feat == self.match_label)):
                    raise Exception("Text feature specified as quantitative; please encode it before running.")
                if feat not in self.categorical_features \
                        and not (feat == self.outcome_label or (self.match_label and feat == self.match_label)):
                    self.categorical_features.append(feat)

                if feat == self.outcome_label:
                    self.data[feat], labels = pd.factorize(self.data[feat])
                    ord_label.loc[feat] = [list(labels), list(range(len(labels)))]
                elif self.data[feat].nunique() <= 2:
                    self.data[feat], labels = pd.factorize(self.data[feat])
                    ord_label.loc[feat] = [list(labels), list(range(len(labels)))]

            out_dir = os.path.join(self.experiment_path, self.name, 'exploratory')
            os.makedirs(out_dir, exist_ok=True)
            ord_label.to_csv(os.path.join(out_dir, 'Numerical_Encoding_Map.csv'))
            with open(os.path.join(out_dir, 'ordinal_encoding.pickle'), 'wb') as f:
                pickle.dump(ord_label, f)

    def drop_ignored_rowcols(self, ignored_features=None):
        if ignored_features is None:
            ignored_features = self.ignore_features
        for feat in ignored_features:
            if feat in self.categorical_features: self.categorical_features.remove(feat)
            if feat in self.quantitative_features: self.quantitative_features.remove(feat)
        self.clean_data(ignored_features)

    def drop_invariant(self):
        try:
            invariant_columns = list(self.data.columns[self.data.nunique(dropna=True) <= 1])
        except Exception:
            invariant_columns = []
        if invariant_columns:
            for feat in list(invariant_columns):
                for lst in (self.categorical_features, self.quantitative_features,
                            self.engineered_features, self.one_hot_features):
                    if feat in lst:
                        lst.remove(feat)
        self.data.drop(invariant_columns, axis=1, inplace=True)

    def feature_engineering(self):
        missingness = self.data.isnull().sum() / len(self.data)
        high_missing = list(missingness[missingness > self.featureeng_missingness].index)
        self.engineered_features = ['Miss_' + f for f in high_missing]
        for feat in high_missing:
            newf = 'Miss_' + feat
            self.data[newf] = self.data[feat].isnull().astype(int)
            self.categorical_features.append(newf)
        out_dir = os.path.join(self.experiment_path, self.name, 'exploratory')
        os.makedirs(out_dir, exist_ok=True)
        if high_missing:
            with open(os.path.join(out_dir, 'engineered_features.pickle'), 'wb') as f:
                pickle.dump(high_missing, f)
            with open(os.path.join(out_dir, 'Missingness_Engineered_Features.csv'), 'w') as f:
                f.write("\n".join(self.engineered_features))

    def feature_removal(self):
        original = self.get_headers()
        thresh = int(self.data.shape[0] * self.cleaning_missingness) - 1
        self.data.dropna(thresh=thresh, axis=1, inplace=True)
        new_features = self.get_headers()
        removed = [c for c in original if c not in new_features]
        for feat in removed:
            for lst in (self.categorical_features, self.engineered_features,
                        self.one_hot_features, self.quantitative_features):
                if feat in lst:
                    lst.remove(feat)
        out_dir = os.path.join(self.experiment_path, self.name, 'exploratory')
        if removed:
            with open(os.path.join(out_dir, 'removed_features.pickle'), 'wb') as f:
                pickle.dump(removed, f)
            with open(os.path.join(out_dir, 'Missingness_Feature_Cleaning.csv'), 'w') as f:
                f.write("\n".join(removed))

    def instance_removal(self):
        f_count = self.data.shape[1] - 1
        if self.instance_label is not None and self.instance_label in self.data.columns:
            f_count -= 1
        if self.match_label is not None and self.match_label in self.data.columns:
            f_count -= 1
        self.data = self.data[self.data.isnull().sum(axis=1) < int(self.cleaning_missingness * max(1, f_count))]

    def categorical_feature_encoding_pandas(self):
        non_binary_categorical = []
        for feat in list(self.categorical_features):
            if feat in self.data.columns and self.data[feat].nunique() > 2:
                non_binary_categorical.append(feat)

        if len(non_binary_categorical) > 0 and self.one_hot_encoding:
            one_hot_df = pd.get_dummies(self.data[non_binary_categorical], columns=non_binary_categorical)
            self.one_hot_features = list(one_hot_df.columns)
            self.data.drop(non_binary_categorical, axis=1, inplace=True)
            self.data = pd.concat([self.data, one_hot_df], axis=1)
            for feat in non_binary_categorical:
                if feat in self.categorical_features:
                    self.categorical_features.remove(feat)
            self.categorical_features += self.one_hot_features

            out_dir = os.path.join(self.experiment_path, self.name, 'exploratory')
            with open(os.path.join(out_dir, 'one_hot_feature.pickle'), 'wb') as f:
                pickle.dump(self.one_hot_features, f)

    def drop_highly_correlated_features(self):
        df_corr_org = self.feature_only_data().corr()
        df_corr = df_corr_org.stack().reset_index()
        df_corr.columns = ['Removed_Feature', 'Correlated_Feature', 'Correlation']
        mask_dups = (df_corr[['Removed_Feature', 'Correlated_Feature']].apply(frozenset, axis=1).duplicated()) | \
                    (df_corr['Removed_Feature'] == df_corr['Correlated_Feature'])
        df_corr = df_corr[~mask_dups].sort_values(by='Correlation', key=abs, ascending=False)
        df_corr = df_corr[abs(df_corr['Correlation']) >= float(self.correlation_removal_threshold)]
        features_to_drop = [f for f in df_corr['Removed_Feature'] if f in self.data.columns]

        if features_to_drop:
            for feat in features_to_drop:
                for lst in (self.categorical_features, self.engineered_features,
                            self.one_hot_features, self.quantitative_features):
                    if feat in lst:
                        lst.remove(feat)
            self.data.drop(columns=features_to_drop, inplace=True, errors='ignore')

            out_dir = os.path.join(self.experiment_path, self.name, 'exploratory')
            with open(os.path.join(out_dir, 'correlated_features.pickle'), 'wb') as f:
                pickle.dump(features_to_drop, f)

            all_features = set(self.get_headers())
            features_kept = list(all_features - set(features_to_drop))
            with open(os.path.join(out_dir, 'correlation_feature_cleaning.csv'), 'w', newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Retained Feature', 'Deleted Features'])
                for feat in features_kept:
                    if feat in df_corr_org.columns:
                        corr_feat = list(df_corr_org[abs(df_corr_org[feat]) >= float(self.correlation_removal_threshold)].index)
                        if feat in corr_feat: corr_feat.remove(feat)
                        if corr_feat:
                            writer.writerow([feat] + corr_feat)

    def initial_eda(self, initial='initial/'):
        logging.info(self.experiment_path)
        if "Describe" in self.explorations:
            self.describe_data(initial=initial)
            _ = self.missingness_counts(initial=initial)
            self.missing_count_plot(initial=initial)  # will only plot if flags allow
        if "Feature Correlation" in self.explorations:
            self.feature_correlation(initial=initial)  # CSV always; PNG gated

    def second_eda(self, top_features=20):
        logging.info("Running Post-Processing EDA...")

        if "Describe" in self.explorations:
            self.describe_data()
            total_missing = self.missingness_counts()
            self.counts_summary(total_missing, save=True, replicate=False)
            self.missing_count_plot()  # gated

        if "Feature Correlation" in self.explorations:
            x_data = self.feature_only_data()
            self.feature_correlation(x_data)  # CSV always; PNG gated

        if "Univariate Analysis" in self.explorations:
            sorted_p_list = self.univariate_analysis(top_features)
            if self.enable_plots and self.plot_univariate:
                self.univariate_plots(sorted_p_list[: self.univariate_top_k])

        pd.DataFrame(self.categorical_features, columns=['Feature']).to_csv(
            os.path.join(self.experiment_path, self.name, 'exploratory', 'processed_categorical_features.csv'),
            index=False
        )
        pd.DataFrame(self.quantitative_features, columns=['Feature']).to_csv(
            os.path.join(self.experiment_path, self.name, 'exploratory', 'processed_quantitative_features.csv'),
            index=False
        )

    def univariate_analysis(self, top_features=20):
        try:
            out_dir = os.path.join(self.experiment_path, self.name, 'exploratory', 'univariate_analyses')
            os.makedirs(out_dir, exist_ok=True)

            p_value_dict = {}
            for column in self.data.columns:
                if column != self.outcome_label and column != self.instance_label:
                    p_value_dict[column] = self.test_selector(column)

            dict_items = list(p_value_dict.items())
            sorted_p_list = sorted(dict_items, key=lambda item: float(item[1][0]))
            sorted_p_list = [(item[0], float(item[1][0])) for item in sorted_p_list]

            pval_df = pd.DataFrame.from_dict(p_value_dict, orient='index')
            pval_df.to_csv(os.path.join(out_dir, 'Univariate_Significance.csv'),
                           index_label='Feature', header=['p-value', 'Test-statistic', 'Test-name'], na_rep='NaN')

        except Exception as e:
            sorted_p_list = []
            logging.warning('Univariate analysis failed (scipy compat). Consider scipy>=1.8.0.')
            for column in self.data.columns:
                if column != self.outcome_label and column != self.instance_label:
                    sorted_p_list.append([column, 'None'])

        return sorted_p_list
    
    def test_selector(self, feature_name):
        outcome = self.outcome_label
        p_val, test_stat, test_name = None, None, None

        try:
            # Work on rows where both vars are present
            df = self.data[[feature_name, outcome]].dropna()
            if df.empty:
                raise ValueError(f"No non-NaN data for {feature_name} vs {outcome}.")

            is_feature_cat = feature_name in getattr(self, "categorical_features", set()) \
                            or not is_numeric_dtype(df[feature_name])
            # NOTE: trust caller's self.outcome_type, but still infer basic properties
            outcome_unique = df[outcome].unique()

            if self.outcome_type in ("Binary", "Multiclass"):
                if is_feature_cat:
                    # Categorical x Categorical -> Chi-square (fallback to Fisher for 2x2 with small expected)
                    table = pd.crosstab(df[feature_name], df[outcome])
                    if table.shape[0] < 2 or table.shape[1] < 2:
                        raise ValueError("Contingency table must be at least 2x2.")
                    # Check expected counts
                    chi2, p, dof, expected = chi2_contingency(table)
                    if table.shape == (2, 2) and (expected < 5).any():
                        # Use Fisher’s exact for small counts
                        # Flatten to [[a,b],[c,d]]
                        a, b = table.iloc[0, 0], table.iloc[0, 1]
                        c, d = table.iloc[1, 0], table.iloc[1, 1]
                        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
                        p_val, test_stat, test_name = p, odds, "Fisher's Exact Test (2x2)"
                    else:
                        p_val, test_stat, test_name = p, chi2, "Chi-Square Test"
                else:
                    # Numeric feature vs categorical outcome
                    if len(outcome_unique) == 2:
                        # Don’t assume classes are 0/1; take the two labels
                        cls_a, cls_b = outcome_unique[:2]
                        x = df.loc[df[outcome] == cls_a, feature_name].astype(float)
                        y = df.loc[df[outcome] == cls_b, feature_name].astype(float)
                        # Guard for zero-length or constant vectors (MWU will error or be meaningless)
                        if len(x) == 0 or len(y) == 0:
                            raise ValueError("Empty group for Mann-Whitney U.")
                        if x.nunique() <= 1 and y.nunique() <= 1 and float(x.iloc[0]) == float(y.iloc[0]):
                            # Completely constant and equal → non-differentiable
                            p_val, test_stat, test_name = 1.0, 0.0, "Mann-Whitney U Test"
                        else:
                            u, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
                            p_val, test_stat, test_name = p, u, "Mann-Whitney U Test"
                    else:
                        # >2 classes: one-way ANOVA (fallback to Kruskal if any group size < 2)
                        groups = [df.loc[df[outcome] == cat, feature_name].astype(float) for cat in outcome_unique]
                        # Remove empty groups after dropna
                        groups = [g for g in groups if len(g) > 0]
                        if len(groups) < 2:
                            raise ValueError("Need at least two non-empty groups for ANOVA.")
                        if any(len(g) < 2 for g in groups):
                            H, p = kruskal(*groups)
                            p_val, test_stat, test_name = p, H, "Kruskal-Wallis H Test"
                        else:
                            F, p = f_oneway(*groups)
                            p_val, test_stat, test_name = p, F, "One-way ANOVA"
            elif self.outcome_type == "Continuous":
                if is_feature_cat:
                    # Compare continuous outcome across feature categories
                    cats = df[feature_name].unique()
                    groups = [df.loc[df[feature_name] == cat, outcome].astype(float) for cat in cats]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) < 2:
                        raise ValueError("Need at least two non-empty groups for ANOVA.")
                    if any(len(g) < 2 for g in groups):
                        H, p = kruskal(*groups)
                        p_val, test_stat, test_name = p, H, "Kruskal-Wallis H Test"
                    else:
                        F, p = f_oneway(*groups)
                        p_val, test_stat, test_name = p, F, "One-way ANOVA"
                else:
                    # Numeric vs numeric: Spearman default (robust to monotone nonlinearity)
                    res = spearmanr(df[feature_name].astype(float), df[outcome].astype(float), nan_policy="omit")
                    p_val, test_stat, test_name = float(res.pvalue), float(res.statistic), "Spearman Correlation"
            else:
                raise ValueError(f"Unknown outcome_type: {self.outcome_type}")

        except Exception as e:
            logging.error("Stat test failure for %s vs %s: %s", feature_name, outcome, e)
            raise Exception("Stat test error (likely data/assumption issue).")

        return p_val, test_stat, test_name

    def univariate_plots(self, sorted_p_list=None, top_features=20):
        if sorted_p_list is None:
            sorted_p_list = self.univariate_analysis(top_features)
        # Only called if gated; generate PNGs here.
        out_dir = os.path.join(self.experiment_path, self.name, 'exploratory', 'univariate_analyses')
        os.makedirs(out_dir, exist_ok=True)

        for name, pval in sorted_p_list:
            if pval == 'None' or pval > self.sig_cutoff:
                continue
            if self.outcome_type == "Binary":
                if name in self.categorical_features:
                    table = pd.crosstab(self.data[name], self.data[self.outcome_label])
                    table.plot(kind='bar'); plt.ylabel('Contingency Table Count')
                    fname = f'Barplot_{name.replace(" ", "").replace("*", "").replace("/", "")}.png'
                else:
                    self.data.boxplot(column=name, by=self.outcome_label); plt.ylabel(name); plt.title('')
                    fname = f'Boxplot_{name.replace(" ", "").replace("*", "").replace("/", "")}.png'
            elif self.outcome_type == "Continuous":
                if name in self.categorical_features:
                    self.data.boxplot(column=self.outcome_label, by=name); plt.ylabel(self.outcome_label); plt.title('')
                    fname = f'Boxplot_{name.replace(" ", "").replace("*", "").replace("/", "")}.png'
                else:
                    self.data.plot(x=name, y=self.outcome_label, kind='scatter')
                    fname = f'Scatter_{name.replace(" ", "").replace("*", "").replace("/", "")}.png'
            plt.savefig(os.path.join(out_dir, fname), bbox_inches="tight", format='png')
            if self.show_plots: plt.show()
            plt.close('all')

    # ----------------------------
    # Optional: Anomaly detection (CSV-first, plots gated)
    # ----------------------------
    def anomaly_detection(self, use_normalized_data=True, num_instances_to_show=50):
        logging.info('Running anomaly detection.')
        imputed_data = copy.deepcopy(self.data)

        imputer = IterativeImputer(random_state=self.random_state)
        imputed_data[self.quantitative_features] = imputer.fit_transform(imputed_data[self.quantitative_features])

        if use_normalized_data:
            scaler = StandardScaler()
            normalized_imputed_data = scaler.fit_transform(imputed_data[self.quantitative_features])
            normalized_imputed_data = pd.DataFrame(normalized_imputed_data, columns=self.quantitative_features)
        else:
            normalized_imputed_data = imputed_data[self.quantitative_features]

        plot_dir = os.path.join(self.experiment_path, self.name, 'exploratory', 'anomaly_detection')
        os.makedirs(plot_dir, exist_ok=True)

        imputed_isolation_forest = IsolationForest(contamination='auto', random_state=self.random_state)
        imputed_isolation_forest.fit(normalized_imputed_data)
        iso_scores = imputed_isolation_forest.decision_function(normalized_imputed_data)

        lof = LocalOutlierFactor(n_neighbors=15, novelty=True, contamination='auto')
        lof_scores = lof.fit(normalized_imputed_data).negative_outlier_factor_

        ee = EllipticEnvelope(contamination=0.05, support_fraction=0.75, random_state=self.random_state)
        ee.fit(normalized_imputed_data)
        ee_scores = ee.decision_function(normalized_imputed_data)

        scores_df = pd.DataFrame({
            'Isolation Forest': iso_scores,
            'Local Outlier Factor': lof_scores,
            'Elliptic Envelope': ee_scores
        })
        scores_df.to_csv(os.path.join(plot_dir, 'imputed_anomaly_scores.csv'), index=False)

        imputed_data.index = range(len(imputed_data))
        imputed_data.to_csv(os.path.join(plot_dir, 'raw_unnormalized_scores.csv'), index=False)

        ranked = scores_df.rank(axis=0, ascending=False)
        ranks_df = ranked.copy()
        ranks_df.index = range(len(ranks_df))
        ranks_df['Avg_Rank'] = ranks_df.mean(axis=1)
        ranks_df = ranks_df.sort_values(by='Avg_Rank')
        ranks_path = os.path.join(plot_dir, 'rankings.csv')
        ranks_df.to_csv(ranks_path, index=False)

        # PNGs only if enabled
        if self.enable_plots and self.plot_anomalies:
            plt.figure(figsize=(8, 6)); plt.hist(iso_scores, bins=30, edgecolor='black', range=(-1, 1))
            plt.title('Histogram of Isolation Forest Anomaly Scores'); plt.xlabel('Anomaly Score'); plt.ylabel('Frequency')
            plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'isolation_forest_histogram.png')); 
            if self.show_plots: plt.show(); plt.close()

            plt.figure(figsize=(8, 6)); plt.hist(lof_scores, bins=30, edgecolor='black')
            plt.title('Histogram of Local Outlier Factor Anomaly Scores'); plt.xlabel('Anomaly Score'); plt.ylabel('Frequency')
            plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'local_outlier_factor_histogram.png'))
            if self.show_plots: plt.show(); plt.close()

            plt.figure(figsize=(8, 6)); plt.hist(ee_scores, bins=30, edgecolor='black')
            plt.title('Histogram of Elliptic Envelope Anomaly Scores'); plt.xlabel('Anomaly Score'); plt.ylabel('Frequency')
            plt.tight_layout(); plt.savefig(os.path.join(plot_dir, 'elliptic_envelope_histogram.png'))
            if self.show_plots: plt.show(); plt.close()

            # Heatmaps + boxplots can be regenerated later via the renderer; skip here to keep runs light.

        logging.info("Anomaly detection completed.")

    # ----------------------------
    # Runtime
    # ----------------------------

    def save_runtime(self):
        runtime = str(time.time() - self.job_start_time)
        logging.log(0, "PHASE 1 Completed: Runtime=" + str(runtime))
        run_dir = os.path.join(self.experiment_path, self.name, 'runtime')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'runtime_exploratory.txt'), 'w') as f:
            f.write(runtime)

    def start(self, top_features=20):
        self.run(top_features)

    def join(self):
        pass
