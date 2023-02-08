import sys
import argparse


def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    # Arguments with no defaults - Global Args
    parser.add_argument('--data-path', dest='dataset_path', type=str, help='path to directory containing datasets')
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str,
                        help='name of experiment output folder (no spaces)')
    # Arguments with defaults available (but critical to check)
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets',
                        default="Class")
    parser.add_argument('--inst-label', dest='instance_label', type=str,
                        help='instance label of all datasets (if present)', default="None")

    # Phase 1
    parser.add_argument('--fi', dest='ignore_features_path', type=str,
                        help='path to .csv file with feature labels to be ignored in analysis '
                             '(e.g. ./droppedFeatures.csv))',
                        default="")
    parser.add_argument('--cf', dest='categorical_feature_path', type=str,
                        help='path to .csv file with feature labels specified to '
                             'be treated as categorical where possible',
                        default="")
    # Arguments with defaults available (but less critical to check)
    parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions', default=10)
    parser.add_argument('--part', dest='partition_method', type=str,
                        help="'S', or 'R', or 'M', for stratified, random, or matched, respectively", default="S")
    parser.add_argument('--match-label', dest='match_label', type=str,
                        help='only applies when M selected for partition-method; '
                             'indicates column with matched instance ids',
                        default="None")
    parser.add_argument('--cat-cutoff', dest='categorical_cutoff', type=int,
                        help='number of unique values after which a variable is '
                             'considered to be quantitative vs categorical',
                        default=10)
    parser.add_argument('--sig', dest='sig_cutoff', type=float, help='significance cutoff used throughout pipeline',
                        default=0.05)
    parser.add_argument('--export-fc', dest='export_feature_correlations', type=bool,
                        help='run and export feature correlation analysis (yields correlation heatmap)', default=True)
    parser.add_argument('--export-up', dest='export_univariate_plots', type=bool,
                        help='export univariate analysis plots (note: univariate analysis still output by default)',
                        default=True)
    parser.add_argument('--rand-state', dest='random_state', type=int,
                        help='"Dont Panic" - sets a specific random seed for reproducible results', default=42)
    # Logistical arguments
    parser.add_argument('--run-parallel', dest='run_parallel', type=bool,
                        help='if run parallel on LSF compatible computing cluster', default=True)
    parser.add_argument('--res-mem', dest='reserved_memory', type=int,
                        help='reserved memory for the job (in Gigabytes)', default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int,
                        help='maximum memory before the job is automatically terminated', default=15)

    # Defaults available - Phase 2
    parser.add_argument('--scale', dest='scale_data', type=bool,
                        help='perform data scaling (required for SVM, and to use '
                             'Logistic regression with non-uniform feature importance estimation)',
                        default=True)
    parser.add_argument('--impute', dest='impute_data', type=bool,
                        help='perform missing value data imputation '
                             '(required for most ML algorithms if missing data is present)',
                        default=True)
    parser.add_argument('--multi-impute', dest='multi_impute', type=bool,
                        help='applies multivariate imputation to '
                             'quantitative features, otherwise uses median imputation',
                        default=True)
    parser.add_argument('--over-cv', dest='overwrite_cv', type=bool,
                        help='overwrites earlier cv datasets with new scaled/imputed ones', default=True)

    # Defaults available - Phase 3
    parser.add_argument('--do-mi', dest='do_mutual_info', type=bool, help='do mutual information analysis',
                        default=True)
    parser.add_argument('--do-ms', dest='do_multisurf', type=bool, help='do multiSURF analysis', default=True)
    parser.add_argument('--use-turf', dest='use_TURF', type=bool,
                        help='use TURF wrapper around MultiSURF to improve feature '
                             'interaction detection in large feature spaces '
                             '(only recommended if you have reason to believe at '
                             'least half of your features are non-informative)',
                        default=False)
    parser.add_argument('--turf-pct', dest='TURF_pct', type=float,
                        help='proportion of instances removed in an iteration (also dictates number of iterations)',
                        default=0.5)
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        help='number of cores dedicated to running algorithm; '
                             'setting to -1 will use all available cores',
                        default=1)
    parser.add_argument('--inst-sub', dest='instance_subset', type=int, help='sample subset size to use with multiSURF',
                        default=2000)

    # Defaults available - Phase 4
    parser.add_argument('--max-feat', dest='max_features_to_keep', type=int,
                        help='max features to keep (only applies if filter_poor_features is True)', default=2000)
    parser.add_argument('--filter-feat', dest='filter_poor_features', type=bool,
                        help='filter out the worst performing features prior to modeling', default=True)
    parser.add_argument('--top-features', dest='top_features', type=int,
                        help='number of top features to illustrate in figures', default=40)
    parser.add_argument('--export-scores', dest='export_scores', type=bool,
                        help='export figure summarizing average feature importance scores over cv partitions',
                        default=True)
    parser.add_argument('--over-cv', dest='overwrite_cv', type=bool,
                        help='overwrites working cv datasets with new feature subset datasets', default=True)

    # Defaults available - Phase 5
    # Sets default run all or none to make algorithm selection from command line simpler
    parser.add_argument('--do-all', dest='do_all', type=bool,
                        help='run all modeling algorithms by default (when set False, individual algorithms are '
                             'activated individually)',
                        default=True)
    # ML modeling algorithms: Defaults available
    parser.add_argument('--do-NB', dest='do_NB', type=bool, help='run naive bayes modeling', default=False)
    parser.add_argument('--do-LR', dest='do_LR', type=bool, help='run logistic regression modeling', default=False)
    parser.add_argument('--do-DT', dest='do_DT', type=bool, help='run decision tree modeling', default=False)
    parser.add_argument('--do-RF', dest='do_RF', type=bool, help='run random forest modeling', default=False)
    parser.add_argument('--do-GB', dest='do_GB', type=bool, help='run gradient boosting modeling', default=False)
    parser.add_argument('--do-XGB', dest='do_XGB', type=bool, help='run XGBoost modeling', default=False)
    parser.add_argument('--do-LGB', dest='do_LGB', type=bool, help='run LGBoost modeling', default=False)
    parser.add_argument('--do-CGB', dest='do_CGB', type=bool, help='run CatBoost modeling', default=False)
    parser.add_argument('--do-SVM', dest='do_SVM', type=bool, help='run support vector machine modeling', default=False)
    parser.add_argument('--do-ANN', dest='do_ANN', type=bool, help='run artificial neural network modeling',
                        default=False)
    parser.add_argument('--do-KNN', dest='do_KNN', type=bool, help='run k-nearest neighbors classifier modeling',
                        default=False)
    parser.add_argument('--do-GP', dest='do_GP', type=bool, help='run genetic programming symbolic classifier modeling',
                        default=False)
    # Experimental ML modeling algorithms (rule-based ML algorithms that are in develompent by our research group)
    parser.add_argument('--do-eLCS', dest='do_eLCS', type=bool,
                        help='run eLCS modeling (a basic supervised-learning learning classifier system)',
                        default=False)
    parser.add_argument('--do-XCS', dest='do_XCS', type=bool,
                        help='run XCS modeling (a supervised-learning-only implementation of the best studied '
                             'learning classifier system)',
                        default=False)
    parser.add_argument('--do-ExSTraCS', dest='do_ExSTraCS', type=bool,
                        help='run ExSTraCS modeling (a learning classifier system designed for biomedical data mining)',
                        default=False)

    parser.add_argument('--exclude', dest='exclude', type=str,
                        help='comma seperated list of algorithms to exclude',
                        default='eLCS,XCS')

    # Other Analysis Parameters - Defaults available
    parser.add_argument('--metric', dest='primary_metric', type=str,
                        help='primary scikit-learn specified scoring metric used for hyperparameter optimization and '
                             'permutation-based model feature importance evaluation',
                        default='balanced_accuracy')
    parser.add_argument('--subsample', dest='training_subsample', type=int,
                        help='for long running algos (XGB,SVM,ANN,KN), option to subsample training set (0 for no '
                             'subsample)',
                        default=0)
    parser.add_argument('--use-uniformFI', dest='use_uniform_FI', type=str,
                        help='overrides use of any available feature importance estimate methods from models, '
                             'instead using permutation_importance uniformly',
                        default='True')
    # Hyperparameter sweep options - Defaults available
    parser.add_argument('--n-trials', dest='n_trials', type=str,
                        help='# of bayesian hyperparameter optimization trials using optuna (specify an integer or '
                             'None)',
                        default=200)
    parser.add_argument('--timeout', dest='timeout', type=str,
                        help='seconds until hyperparameter sweep stops running new trials (Note: it may run longer to '
                             'finish last trial started) If set to None, STREAMLINE is completely replicable, '
                             'but will take longer to run',
                        default=900)  # 900 sec = 15 minutes default
    parser.add_argument('--export-hyper-sweep', dest='export_hyper_sweep_plots', type=str,
                        help='export optuna-generated hyperparameter sweep plots', default='False')
    # LCS specific parameters - Defaults available
    parser.add_argument('--do-LCS-sweep', dest='do_lcs_sweep', type=str,
                        help='do LCS hyper-param tuning or use below params', default='False')
    parser.add_argument('--nu', dest='nu', type=int,
                        help='fixed LCS nu param (recommended range 1-10), set to larger value for data with less or '
                             'no noise',
                        default=1)
    parser.add_argument('--iter', dest='iterations', type=int, help='fixed LCS # learning iterations param',
                        default=200000)

    # Defaults available - Phase 6
    parser.add_argument('--plot-ROC', dest='plot_ROC', type=str,
                        help='Plot ROC curves individually for each algorithm including all CV results and averages',
                        default='True')
    parser.add_argument('--plot-PRC', dest='plot_PRC', type=str,
                        help='Plot PRC curves individually for each algorithm including all CV results and averages',
                        default='True')
    parser.add_argument('--plot-box', dest='plot_metric_boxplots', type=str,
                        help='Plot box plot summaries comparing algorithms for each metric', default='True')
    parser.add_argument('--plot-FI_box', dest='plot_FI_box', type=str,
                        help='Plot feature importance boxplots and histograms for each algorithm', default='True')
    parser.add_argument('--metric-weight', dest='metric_weight', type=str,
                        help='ML model metric used as weight in composite FI plots (only supports balanced_accuracy '
                             'or roc_auc as options) Recommend setting the same as primary_metric if possible.',
                        default='balanced_accuracy')
    parser.add_argument('--top-features', dest='top_model_features', type=int,
                        help='number of top features to illustrate in figures', default=40)


if __name__ == '__main__':
    sys.exit(main(sys.argv))