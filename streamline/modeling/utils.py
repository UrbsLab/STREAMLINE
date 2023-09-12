import os
import pickle
import logging
import pandas as pd
import multiprocessing
from streamline.modeling.load_models import load_class_from_folder

num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

SUPPORTED_MODELS_OBJ = load_class_from_folder()

SUPPORTED_MODELS = [m.model_name for m in SUPPORTED_MODELS_OBJ]

# logging.warning(SUPPORTED_MODELS)


SUPPORTED_MODELS_SMALL = [m.small_name for m in SUPPORTED_MODELS_OBJ]

COLOR_LIST = [m.color for m in SUPPORTED_MODELS_OBJ]

MODEL_DICT = dict(zip(SUPPORTED_MODELS + SUPPORTED_MODELS_SMALL,
                      SUPPORTED_MODELS_OBJ + SUPPORTED_MODELS_OBJ))

LABELS = dict(zip(SUPPORTED_MODELS + SUPPORTED_MODELS_SMALL,
                  SUPPORTED_MODELS + SUPPORTED_MODELS))

ABBREVIATION = dict(zip(SUPPORTED_MODELS, SUPPORTED_MODELS_SMALL))

COLORS = dict(zip(SUPPORTED_MODELS, COLOR_LIST))


def is_supported_model(string):
    try:
        return LABELS[string]
    except KeyError:
        raise Exception("Unknown Model")


def model_str_to_obj(string):
    assert is_supported_model(string)
    return MODEL_DICT[string]


def get_fi_for_ExSTraCS(output_path, experiment_name, dataset_name, class_label, instance_label,
                        cv, filter_poor_features):
    """
    For ExSTraCS, gets the MultiSURF (or MI if MS not available) FI scores for the feature subset being analyzed
    here in modeling
    """
    scores = []  # to be filled in, in fitted dataset order.
    full_path = output_path + '/' + experiment_name + '/' + dataset_name
    # If MultiSURF was done previously
    if os.path.exists(full_path + "/feature_selection/multisurf/pickledForPhase4/"):
        algorithm_label = 'multisurf'
    elif os.path.exists(full_path + "/feature_selection/mutual_information/pickledForPhase4/"):
        # If MI was done previously and MS wasn't:
        algorithm_label = 'mutual_information'
    else:
        scores = None
        return scores

    if filter_poor_features:
        # obtain feature importance scores for feature subset analyzed (in correct training dataset order)
        # Load current data ordered_feature_names
        header = pd.read_csv(
            full_path + '/CVDatasets/' + dataset_name + '_CV_' + str(cv) + '_Test.csv').columns.values.tolist()
        if instance_label is not None:
            header.remove(instance_label)
        header.remove(class_label)
        # Load original dataset multisurf scores
        score_info = full_path + "/feature_selection/" + algorithm_label + "/pickledForPhase4/" + str(cv) + '.pickle'
        file = open(score_info, 'rb')
        raw_data = pickle.load(file)
        file.close()
        score_dict = raw_data[1]
        # Generate filtered multisurf score list with same order as working datasets
        for each in header:
            scores.append(score_dict[each])
    else:  # obtain feature importance scores for all features (i.e. no feature selection was conducted)
        # Load original dataset multisurf scores
        score_info = full_path + "/feature_selection/" + algorithm_label + "/pickledForPhase4/" + str(cv) + '.pickle'
        file = open(score_info, 'rb')
        raw_data = pickle.load(file)
        file.close()
        scores = raw_data[0]
    return scores
