import os
from streamline.featurefns.load_algorithms import load_class_from_folder

SUPPORTED_ALGORITHM_OBJ = load_class_from_folder()

SUPPORTED_ALGORITHM = [m.model_name for m in SUPPORTED_ALGORITHM_OBJ]

SUPPORTED_ALGORITHM_SMALL = [m.small_name for m in SUPPORTED_ALGORITHM_OBJ]

ALGORITHM_DICT = dict(zip(SUPPORTED_ALGORITHM + SUPPORTED_ALGORITHM_SMALL,
                      SUPPORTED_ALGORITHM_OBJ + SUPPORTED_ALGORITHM_OBJ))

LABELS = dict(zip(SUPPORTED_ALGORITHM + SUPPORTED_ALGORITHM_SMALL,
                  SUPPORTED_ALGORITHM + SUPPORTED_ALGORITHM))


def is_supported_algorithm(string):
    try:
        return LABELS[string]
    except KeyError:
        raise Exception("Unknown Model")


def algorithm_str_to_obj(string):
    assert is_supported_algorithm(string)
    return ALGORITHM_DICT[string]
