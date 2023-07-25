import os
import logging
import multiprocessing
from streamline.modeling.load_models import load_class_from_folder

num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

SUPPORTED_CLASSIFICATION_MODELS_OBJ = load_class_from_folder(model_type="MulticlassClassification")

SUPPORTED_CLASSIFICATION_MODELS = [m.model_name for m in SUPPORTED_CLASSIFICATION_MODELS_OBJ]

# logging.warning(SUPPORTED_CLASSIFICATION_MODELS)


SUPPORTED_CLASSIFICATION_MODELS_SMALL = [m.small_name for m in SUPPORTED_CLASSIFICATION_MODELS_OBJ]

CLASSIFICATION_COLOR_LIST = [m.color for m in SUPPORTED_CLASSIFICATION_MODELS_OBJ]

CLASSIFICATION_MODEL_DICT = dict(zip(SUPPORTED_CLASSIFICATION_MODELS + SUPPORTED_CLASSIFICATION_MODELS_SMALL,
                                 SUPPORTED_CLASSIFICATION_MODELS_OBJ + SUPPORTED_CLASSIFICATION_MODELS_OBJ))

CLASSIFICATION_LABELS = dict(zip(SUPPORTED_CLASSIFICATION_MODELS + SUPPORTED_CLASSIFICATION_MODELS_SMALL,
                             SUPPORTED_CLASSIFICATION_MODELS + SUPPORTED_CLASSIFICATION_MODELS))

CLASSIFICATION_ABBREVIATION = dict(zip(SUPPORTED_CLASSIFICATION_MODELS, SUPPORTED_CLASSIFICATION_MODELS_SMALL))

CLASSIFICATION_COLORS = dict(zip(SUPPORTED_CLASSIFICATION_MODELS, CLASSIFICATION_COLOR_LIST))


def is_supported_model(string):
    try:
        return CLASSIFICATION_LABELS[string]
    except KeyError:
        raise Exception("Unknown Model")


def model_str_to_obj(string):
    assert is_supported_model(string)
    return CLASSIFICATION_MODEL_DICT[string]
