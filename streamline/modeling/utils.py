import logging

from streamline.modeling.load_models import load_class_from_folder


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
