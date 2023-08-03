import os
import pickle
import argparse
from streamline.modeling.utils import SUPPORTED_MODELS_SMALL


def comma_sep_choices(choices):
    """
    Return a function that splits and checks comma-separated values.
    """

    def splitarg(arg):
        if arg == 'None':
            return None
        elif ',' not in arg:
            return [arg, ]
        else:
            values = arg.split(',')
        for value in values:
            if value not in choices:
                raise argparse.ArgumentTypeError(
                    'invalid choice: {!r} (choose from {})'
                    .format(value, ', '.join(map(repr, choices))))
        return values

    return splitarg


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')


def save_config(output_path, experiment_name, config_dict):
    if not os.path.exists(config_dict['output_path']):
        os.mkdir(str(config_dict['output_path']))
    with open(output_path + '/' + experiment_name + '_params.pickle', 'wb') as file:
        pickle.dump(config_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_config(output_path, experiment_name, config=None):
    if config is None:
        config = dict()
    try:
        with open(output_path + '/' + experiment_name + '_params.pickle', 'rb') as file:
            config_file = pickle.load(file)
            config.update(config_file)
    except FileNotFoundError:
        pass
    return config


def update_dict_from_parser(argv, parser, params_dict=None):
    if not params_dict:
        params_dict = dict()
    args, unknown = parser.parse_known_args(argv[1:])
    params_dict.update(vars(args))
    return params_dict


def parse_general(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Arguments with no defaults - Global Args
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str,
                        help='name of experiment output folder (no spaces)')
    # Arguments with defaults available (but critical to check)
    parser.add_argument('--verbose', dest='verbose', type=str2bool, nargs='?', default=False,
                        help='give output to command line')
    return update_dict_from_parser(argv, parser, params_dict)


def parse_eda(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', dest='dataset_path', type=str, help='path to directory containing datasets')
    parser.add_argument('--inst-label', dest='instance_label', type=str,
                        help='instance label of all datasets (if present)', default="")
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets',
                        default="Class")
    parser.add_argument('--match-label', dest='match_label', type=str,
                        help='only applies when Group selected for partition-method; '
                             'indicates column with matched instance ids',
                        default='')
    # Arguments with defaults available (but less critical to check)
    parser.add_argument('--fi', dest='ignore_features_path', type=str,
                        help='path to .csv file with feature labels to be ignored in analysis '
                             '(e.g. ./droppedFeatures.csv))',
                        default="")
    parser.add_argument('--cf', dest='categorical_feature_path', type=str,
                        help='path to .csv file with feature labels specified to '
                             'be treated as categorical where possible',
                        default="")
    parser.add_argument('--qf', dest='quantitative_feature_path', type=str,
                        help='path to .csv file with feature labels specified to '
                             'be treated as categorical where possible',
                        default="")

    parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions', default=10)
    parser.add_argument('--part', dest='partition_method', type=str,
                        help="Stratified, Random, or Group Stratification", default="Stratified")
    parser.add_argument('--cat-cutoff', dest='categorical_cutoff', type=int,
                        help='number of unique values after which a variable is '
                             'considered to be quantitative vs categorical',
                        default=10)
    parser.add_argument('--top-uni-features', dest='top_uni_features', type=int,
                        help='number of top features to illustrate in figures', default=40)
    parser.add_argument('--sig', dest='sig_cutoff', type=float, help='significance cutoff used throughout pipeline',
                        default=0.05)
    parser.add_argument('--feat_miss', dest='featureeng_missingness', type=float,
                        help='feature missingness cutoff used throughout pipeline',
                        default=0.5)
    parser.add_argument('--clean_miss', dest='cleaning_missingness', type=float,
                        help='cleaning missingness cutoff used throughout pipeline',
                        default=0.5)
    parser.add_argument('--corr_thresh', dest='correlation_removal_threshold', type=float,
                        help='correlation removal threshold',
                        default=0.8)
    # parser.add_argument('--export-fc', dest='export_feature_correlations', type=str2bool, nargs='?',
    #                     help='run and export feature correlation analysis (yields correlation heatmap)', default=True)
    # parser.add_argument('--export-up', dest='export_univariate_plots', type=str2bool, nargs='?',
    #                     help='export univariate analysis plots (note: univariate analysis still output by default)',
    #                     default=True)
    parser.add_argument('--exclude-eda-output', dest='exclude_eda_output',
                        type=comma_sep_choices(['describe_csv', 'univariate_plots', 'correlation_plots']),
                        help='comma seperated list of eda outputs to exclude',
                        default='None')

    parser.add_argument('--rand-state', dest='random_state', type=int,
                        help='"Dont Panic" - sets a specific random seed for reproducible results', default=42)
    return update_dict_from_parser(argv, parser, params_dict)


def parse_dataprep(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Defaults available - Phase 2
    parser.add_argument('--scale', dest='scale_data', type=str2bool, nargs='?',
                        help='perform data scaling (required for SVM, and to use '
                             'Logistic regression with non-uniform feature importance estimation)', const=True,
                        default=True)
    parser.add_argument('--impute', dest='impute_data', type=str2bool, nargs='?',
                        help='perform missing value data imputation '
                             '(required for most ML algorithms if missing data is present)', const=True,
                        default=True)
    parser.add_argument('--multi-impute', dest='multi_impute', type=str2bool, nargs='?',
                        help='applies multivariate imputation to '
                             'quantitative features, otherwise uses median imputation', const=True,
                        default=True)
    parser.add_argument('--over-cv', dest='overwrite_cv', type=str2bool, nargs='?', const=False,
                        help='overwrites earlier cv datasets with new scaled/imputed ones', default=False)

    return update_dict_from_parser(argv, parser, params_dict)


def parse_feat_imp(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Defaults available - Phase 3
    parser.add_argument('--do-mi', dest='do_mutual_info', type=str2bool, nargs='?',
                        help='do mutual information analysis', const=True,
                        default=True)
    parser.add_argument('--do-ms', dest='do_multisurf', type=str2bool, nargs='?', const=True,
                        help='do multiSURF analysis', default=True)
    parser.add_argument('--use-turf', dest='use_turf', type=str2bool, nargs='?', const=True,
                        help='use TURF wrapper around MultiSURF to improve feature '
                             'interaction detection in large feature spaces '
                             '(only recommended if you have reason to believe at '
                             'least half of your features are non-informative)',
                        default=False)
    parser.add_argument('--turf-pct', dest='turf_pct', type=float,
                        help='proportion of instances removed in an iteration (also dictates number of iterations)',
                        default=0.5)
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        help='number of cores dedicated to running algorithm; '
                             'setting to -1 will use all available cores',
                        default=1)
    parser.add_argument('--inst-sub', dest='instance_subset', type=int, help='sample subset size to use with multiSURF',
                        default=2000)
    return update_dict_from_parser(argv, parser, params_dict)


def parse_feat_sel(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Defaults available - Phase 4
    parser.add_argument('--max-feat', dest='max_features_to_keep', type=int,
                        help='max features to keep (only applies if filter_poor_features is True)', default=2000)
    parser.add_argument('--filter-feat', dest='filter_poor_features', type=str2bool, nargs='?', const=True,
                        help='filter out the worst performing features prior to modeling', default=True)
    parser.add_argument('--top-fi-features', dest='top_fi_features', type=int,
                        help='number of top features to illustrate in figures', default=40)
    parser.add_argument('--export-scores', dest='export_scores', type=str2bool, nargs='?',
                        help='export figure summarizing average feature importance scores over cv partitions',
                        default=True)
    parser.add_argument('--over-cv-feat', dest='overwrite_cv_feat', type=str2bool, nargs='?', const=True,
                        help='overwrites working cv datasets with new feature subset datasets', default=True)
    return update_dict_from_parser(argv, parser, params_dict)


def parse_model(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Defaults available - Phase 5
    # Sets default run all or none to make algorithm selection from command line simpler
    parser.add_argument('--do-all', dest='do_all', type=str2bool, nargs='?',
                        help='run all modeling algorithms by default (when set False, individual algorithms are '
                             'activated individually)',
                        default=False)

    parser.add_argument('--algorithms', dest='algorithms',
                        type=comma_sep_choices(SUPPORTED_MODELS_SMALL),
                        help='comma seperated list of algorithms to exclude',
                        default='LR,DT,NB')

    parser.add_argument('--model-resubmit', dest='model_resubmit',
                        type=str2bool, nargs='?',
                        help='flag to resubmit models instead',
                        default=False)

    parser.add_argument('--exclude', dest='exclude',
                        type=comma_sep_choices(SUPPORTED_MODELS_SMALL),
                        help='comma seperated list of algorithms to exclude',
                        default='eLCS,XCS')

    # Other Analysis Parameters - Defaults available
    parser.add_argument('--metric', dest='primary_metric', type=str,
                        help='primary scikit-learn specified scoring metric used for hyper parameter optimization and '
                             'permutation-based model feature importance evaluation',
                        default='balanced_accuracy')
    parser.add_argument('--metric-direction', dest='metric_direction', type=str,
                        help='optimization direction on primary metric, maximize or minimize, default maximize',
                        default='maximize')
    parser.add_argument('--subsample', dest='training_subsample', type=int,
                        help='for long running algos (XGB,SVM,ANN,KN), option to subsample training set (0 for no '
                             'subsample)',
                        default=0)
    parser.add_argument('--use-uniformFI', dest='use_uniform_fi', type=str,
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
    parser.add_argument('--nu', dest='lcs_nu', type=int,
                        help='fixed LCS nu param (recommended range 1-10), set to larger value for data with less or '
                             'no noise',
                        default=1)
    parser.add_argument('--iter', dest='lcs_iterations', type=int, help='fixed LCS # learning iterations param',
                        default=200000)
    parser.add_argument('--N', dest='lcs_n', type=int, help='fixed LCS rule population maximum size param',
                        default=2000)
    parser.add_argument('--lcs-timeout', dest='lcs_timeout', type=int, help='seconds until hyper parameter sweep stops '
                                                                            'for LCS algorithms', default=1200)
    return update_dict_from_parser(argv, parser, params_dict)


def parse_stats(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Defaults available - Phase 6
    # parser.add_argument('--plot-ROC', dest='plot_roc', type=str,
    #                     help='Plot ROC curves individually for each algorithm including all CV results and averages',
    #                     default='True')
    # parser.add_argument('--plot-PRC', dest='plot_prc', type=str,
    #                     help='Plot PRC curves individually for each algorithm including all CV results and averages',
    #                     default='True')
    # parser.add_argument('--plot-box', dest='plot_metric_boxplots', type=str,
    #                     help='Plot box plot summaries comparing algorithms for each metric', default='True')
    # parser.add_argument('--plot-FI_box', dest='plot_fi_box', type=str,
    #                     help='Plot feature importance boxplots and histograms for each algorithm', default='True')
    parser.add_argument('--exclude-plots', dest='exclude_plots',
                        type=comma_sep_choices(['plot_ROC', 'plot_PRC', 'plot_FI_box', 'plot_metric_boxplots']),
                        help='comma seperated list of plots to exclude '
                             'possible options plot_ROC, plot_PRC, plot_FI_box, plot_metric_boxplots',
                        default='None')
    parser.add_argument('--metric-weight', dest='metric_weight', type=str,
                        help='ML model metric used as weight in composite FI plots (only supports balanced_accuracy '
                             'or roc_auc as options) Recommend setting the same as primary_metric if possible.',
                        default='balanced_accuracy')
    parser.add_argument('--top-model-features', dest='top_model_fi_features', type=int,
                        help='number of top features to illustrate in figures', default=40)
    return update_dict_from_parser(argv, parser, params_dict)


def parse_replicate(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Phase 9/11
    parser.add_argument('--rep-path', dest='rep_data_path', type=str,
                        help='path to directory containing replication or hold-out testing datasets (must have at '
                             'least all features with same labels as in original training dataset)', default="")

    parser.add_argument('--dataset', dest='dataset_for_rep', type=str,
                        help='path to directory containing replication or hold-out testing datasets (must have at '
                             'least all features with same labels as in original training dataset)', default="")
    # Defaults available
    parser.add_argument('--rep-export-fc', dest='rep_export_feature_correlations', type=str2bool, nargs='?',
                        help='run and export feature correlation analysis (yields correlation heatmap)', default=True)
    # parser.add_argument('--rep-plot-ROC', dest='rep_plot_roc', type=str2bool, nargs='?',
    #                     help='Plot ROC curves individually for each algorithm including all CV results and averages',
    #                     default=True)
    # parser.add_argument('--rep-plot-PRC', dest='rep_plot_prc', type=str2bool, nargs='?',
    #                     help='Plot PRC curves individually for each algorithm including all CV results and averages',
    #                     default=True)
    # parser.add_argument('--rep-plot-box', dest='rep_plot_metric_boxplots', type=str2bool, nargs='?',
    #                     help='Plot box plot summaries comparing algorithms for each metric', default=True)
    parser.add_argument('--exclude-rep-plots', dest='exclude_rep_plots',
                        type=comma_sep_choices(['plot_ROC', 'plot_PRC',
                                                'plot_metric_boxplots', 'feature_correlations']),
                        help='comma seperated list of plots to exclude '
                             'possible options plot_ROC, plot_PRC, plot_FI_box, plot_metric_boxplots',
                        default='None')
    return update_dict_from_parser(argv, parser, params_dict)


def parse_cleanup(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--del-time', dest='del_time', type=str2bool, nargs='?',
                        help='flag to run cleanup', default=True)
    parser.add_argument('--del-old-cv', dest='del_old_cv', type=str2bool, nargs='?',
                        help='flag to run cleanup', default=True)
    return update_dict_from_parser(argv, parser, params_dict)


def parse_logistic(argv, params_dict=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Logistical arguments
    parser.add_argument('--run-parallel', dest='run_parallel', type=str,
                        help='if run parallel on through multiprocessing', default=False)
    parser.add_argument('--run-cluster', dest='run_cluster', type=str,
                        help='if run parallel through SLURM process', default="SLURM")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int,
                        help='reserved memory for the job (in Gigabytes)', default=4)
    parser.add_argument('--queue', dest='queue', type=str,
                        help='default partition queue', default="defq")
    return update_dict_from_parser(argv, parser, params_dict)


def parser_function_all(argv, params_dict=None):
    params_dict = parse_general(argv, params_dict)
    params_dict = parse_eda(argv, params_dict)
    params_dict = parse_dataprep(argv, params_dict)
    params_dict = parse_feat_imp(argv, params_dict)
    params_dict = parse_feat_sel(argv, params_dict)
    params_dict = parse_model(argv, params_dict)
    params_dict = parse_stats(argv, params_dict)
    params_dict = parse_logistic(argv, params_dict)
    return params_dict


PARSER_LIST = [parse_eda,
               parse_dataprep,
               parse_feat_imp,
               parse_feat_sel,
               parse_model,
               parse_stats,
               None,
               None,
               parse_replicate,
               None,
               parse_cleanup]
