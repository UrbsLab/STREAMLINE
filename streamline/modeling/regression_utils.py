import os
import logging
import multiprocessing
from streamline.modeling.load_models import load_class_from_folder

num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

SUPPORTED_REGRESSION_MODELS_OBJ = load_class_from_folder(model_type="Classification")

SUPPORTED_REGRESSION_MODELS = [m.model_name for m in SUPPORTED_REGRESSION_MODELS_OBJ]

# logging.warning(SUPPORTED_REGRESSION_MODELS)


SUPPORTED_REGRESSION_MODELS_SMALL = [m.small_name for m in SUPPORTED_REGRESSION_MODELS_OBJ]

REGRESSION_COLOR_LIST = [m.color for m in SUPPORTED_REGRESSION_MODELS_OBJ]

REGRESSION_MODEL_DICT = dict(zip(SUPPORTED_REGRESSION_MODELS + SUPPORTED_REGRESSION_MODELS_SMALL,
                                 SUPPORTED_REGRESSION_MODELS_OBJ + SUPPORTED_REGRESSION_MODELS_OBJ))

REGRESSION_LABELS = dict(zip(SUPPORTED_REGRESSION_MODELS + SUPPORTED_REGRESSION_MODELS_SMALL,
                             SUPPORTED_REGRESSION_MODELS + SUPPORTED_REGRESSION_MODELS))

REGRESSION_ABBREVIATION = dict(zip(SUPPORTED_REGRESSION_MODELS, SUPPORTED_REGRESSION_MODELS_SMALL))

REGRESSION_COLORS = dict(zip(SUPPORTED_REGRESSION_MODELS, REGRESSION_COLOR_LIST))


def is_supported_model(string):
    try:
        return REGRESSION_LABELS[string]
    except KeyError:
        raise Exception("Unknown Model")


def model_str_to_obj(string):
    assert is_supported_model(string)
    return REGRESSION_MODEL_DICT[string]
