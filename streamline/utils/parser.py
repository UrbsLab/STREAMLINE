import argparse
import configparser
from streamline.utils.parser_helpers import str2bool, save_config, load_config
from streamline.utils.parser_helpers import parse_general
from streamline.utils.parser_helpers import parse_logistic
from streamline.utils.parser_helpers import parser_function_all
from streamline.utils.parser_helpers import PARSER_LIST


def process_params(params):
    if params['run_cluster'] not in [False, "False"]:
        params['run_parallel'] = True

    if params['do_till_report']:
        params["do_eda"] = True
        params["do_dataprep"] = True
        params["do_feat_imp"] = True
        params["do_feat_sel"] = True
        params["do_model"] = True
        params["do_stats"] = True
        params["do_compare_dataset"] = True
        params["do_report"] = True

    if params['do_feat_imp'] or params['do_feat_sel'] \
            or params['do_report'] or params['do_rep_report']:
        if 'feat_algorithms' not in params:
            feat_algorithms = list()
            if params['do_mutual_info']:
                feat_algorithms.append("MI")
            if params['do_multisurf']:
                feat_algorithms.append("MS")
            params['feat_algorithms'] = feat_algorithms

    if params['do_model'] or params['do_stats'] or params["do_compare_dataset"] \
            or params['do_report'] or params['do_replicate'] or params['do_rep_report']:
        if params['do_all']:
            params['algorithms'] = None

    if params['algorithms'] == 'All':
        params['algorithms'] = None
    if params['ignore_features_path'] == '':
        params['ignore_features_path'] = None
    if params['categorical_feature_path'] == '':
        params['categorical_feature_path'] = None
    if params['match_label'] == '':
        params['match_label'] = None
    if params['instance_label'] == '':
        params['instance_label'] = None
    if params['run_cluster'] == "False":
        params['run_cluster'] = False
    if params['run_parallel'] == "False":
        params['run_parallel'] = False
    if params['run_parallel'] == "True":
        params['run_parallel'] = True

    return params


def single_parse(mode_params, argv, config_dict=None):
    if config_dict is None:
        config_dict = dict()
    config_dict = parse_general(argv, config_dict)
    keys = ['do_eda',
            'do_dataprep',
            'do_feat_imp',
            'do_feat_sel',
            'do_model',
            'do_stats',
            'do_compare_dataset',
            'do_report',
            'do_replicate',
            'do_rep_report',
            'do_cleanup', ]
    for i in range(len(keys)):
        if mode_params[keys[i]]:
            if i == 0:
                config_dict = PARSER_LIST[i](argv, config_dict)
                save_config(config_dict['output_path'],
                            config_dict['experiment_name'],
                            config_dict)
            if i not in [6, 7, 9]:
                config_dict = load_config(config_dict['output_path'],
                                          config_dict['experiment_name'])
                config_dict = PARSER_LIST[i](argv, config_dict)
                save_config(config_dict['output_path'],
                            config_dict['experiment_name'],
                            config_dict)
            else:
                config_dict = load_config(config_dict['output_path'],
                                          config_dict['experiment_name'])
            config_dict = parse_logistic(argv, config_dict)
    return config_dict


def parser_function(argv):
    parser = argparse.ArgumentParser(description="STREAMLINE: \n"
                                                 "Simple Transparent End-To-End Automated Machine "
                                                 "Learning Pipeline for Supervised Learning in Tabular "
                                                 "Binary Classification Data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c',
                        dest='config', type=str, default="",
                        help='flag to load config file')
    parser.add_argument('--verbose', dest='verbose', type=str2bool, nargs='?', const=True, default=False,
                        help='give output to command line')
    parser.add_argument('--do-till-report', '--dtr', dest='do_till_report', type=str2bool, nargs='?', const=True,
                        help='flag to do all phases', default=False)
    parser.add_argument('--do-eda', dest='do_eda', type=str2bool, nargs='?', const=True,
                        help='flag to eda', default=False)
    parser.add_argument('--do-dataprep', dest='do_dataprep', type=str2bool, nargs='?', const=True,
                        help='flag to data preprocessing', default=False)
    parser.add_argument('--do-feat-imp', dest='do_feat_imp', type=str2bool, nargs='?', const=True,
                        help='flag to feature importance', default=False)
    parser.add_argument('--do-feat-sel', dest='do_feat_sel', type=str2bool, nargs='?', const=True,
                        help='flag to feature selection', default=False)
    parser.add_argument('--do-model', dest='do_model', type=str2bool, nargs='?', const=True,
                        help='flag to run models', default=False)
    parser.add_argument('--do-stats', dest='do_stats', type=str2bool, nargs='?', const=True,
                        help='flag to run statistics', default=False)
    parser.add_argument('--do-compare-dataset', dest='do_compare_dataset', type=str2bool, nargs='?', const=True,
                        help='flag to run compare dataset dataset', default=False)
    parser.add_argument('--do-report', dest='do_report', type=str2bool, nargs='?', const=True,
                        help='flag to run report dataset', default=False)
    parser.add_argument('--do-replicate', dest='do_replicate', type=str2bool, nargs='?', const=True,
                        help='flag to run replication dataset', default=False)
    parser.add_argument('--do-rep-report', dest='do_rep_report', type=str2bool, nargs='?', const=True,
                        help='flag to run replication report', default=False)
    parser.add_argument('--do-cleanup', dest='do_cleanup', type=str2bool, nargs='?', const=True,
                        help='flag to run cleanup', default=False)
    args, unknown = parser.parse_known_args(argv[1:])
    mode_params = vars(args)
    if len(mode_params) == 0 or ('verbose' in mode_params and len(mode_params) == 1):
        return Exception("Improper Phase Declaration")

    config_dict = dict()

    if mode_params['config'] != "":
        config_file = mode_params['config']
        config = configparser.ConfigParser()
        config.read(config_file)
        for s in config.sections():
            config_dict.update({k: eval(v) for k, v in config.items(s)})
    elif mode_params['do_till_report']:
        print("Running till Report Generation Stage")
        config = parser_function_all(argv)
        config_dict.update(config)
        config_dict.update(mode_params)
    else:
        config_dict = single_parse(mode_params, argv, config_dict)
        config_dict.update(mode_params)

    config_dict = process_params(config_dict)

    return config_dict
