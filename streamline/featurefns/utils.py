import os
from streamline.featurefns.load_algorithms import load_class_from_folder

# Load all algorithm classes from the specified folder
SUPPORTED_ALGORITHM_OBJ = load_class_from_folder()

# Extract the model names from the loaded algorithm objects
SUPPORTED_ALGORITHM = [m.model_name for m in SUPPORTED_ALGORITHM_OBJ]

# Extract the small names (short names) from the loaded algorithm objects
SUPPORTED_ALGORITHM_SMALL = [m.small_name for m in SUPPORTED_ALGORITHM_OBJ]

# Create a dictionary mapping both model names and small names to their respective algorithm objects
ALGORITHM_DICT = dict(zip(SUPPORTED_ALGORITHM + SUPPORTED_ALGORITHM_SMALL,
                          SUPPORTED_ALGORITHM_OBJ + SUPPORTED_ALGORITHM_OBJ))

# Create a dictionary mapping both model names and small names to themselves (for label checking)
LABELS = dict(zip(SUPPORTED_ALGORITHM + SUPPORTED_ALGORITHM_SMALL,
                  SUPPORTED_ALGORITHM + SUPPORTED_ALGORITHM_SMALL))


def is_supported_algorithm(string):
    """
    Check if the given string corresponds to a supported algorithm.
    If it is supported, return the label.
    If not, raise an exception.
    """
    try:
        return LABELS[string]
    except KeyError:
        raise Exception("Unknown Model")


def algorithm_str_to_obj(string):
    """
    Convert an algorithm name (or small name) to its corresponding algorithm object.
    This function asserts that the given string is a supported algorithm.
    """
    assert is_supported_algorithm(string)  # Ensure the string is a supported algorithm
    return ALGORITHM_DICT[string]  # Return the corresponding algorithm object
