import os
import logging
import multiprocessing
from streamline.modeling.load_models import load_class_from_folder

# Determine the number of CPU cores to use, checking for SLURM_CPUS_PER_TASK environment variable or using total CPU count
num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

# Load all supported multiclass classification models
SUPPORTED_CLASSIFICATION_MODELS_OBJ = load_class_from_folder(model_type="MulticlassClassification")

# Extract the model names from the loaded models
SUPPORTED_CLASSIFICATION_MODELS = [m.model_name for m in SUPPORTED_CLASSIFICATION_MODELS_OBJ]

# Extract the small names (abbreviations) from the loaded models
SUPPORTED_CLASSIFICATION_MODELS_SMALL = [m.small_name for m in SUPPORTED_CLASSIFICATION_MODELS_OBJ]

# Extract the color associated with each model
CLASSIFICATION_COLOR_LIST = [m.color for m in SUPPORTED_CLASSIFICATION_MODELS_OBJ]

# Create a dictionary mapping both full and small names to the model objects
CLASSIFICATION_MODEL_DICT = dict(zip(SUPPORTED_CLASSIFICATION_MODELS + SUPPORTED_CLASSIFICATION_MODELS_SMALL,
                                     SUPPORTED_CLASSIFICATION_MODELS_OBJ + SUPPORTED_CLASSIFICATION_MODELS_OBJ))

# Create a dictionary mapping both full and small names to the full model names
CLASSIFICATION_LABELS = dict(zip(SUPPORTED_CLASSIFICATION_MODELS + SUPPORTED_CLASSIFICATION_MODELS_SMALL,
                                 SUPPORTED_CLASSIFICATION_MODELS + SUPPORTED_CLASSIFICATION_MODELS))

# Create a dictionary mapping full model names to their abbreviations
CLASSIFICATION_ABBREVIATION = dict(zip(SUPPORTED_CLASSIFICATION_MODELS, SUPPORTED_CLASSIFICATION_MODELS_SMALL))

# Create a dictionary mapping full model names to their associated colors
CLASSIFICATION_COLORS = dict(zip(SUPPORTED_CLASSIFICATION_MODELS, CLASSIFICATION_COLOR_LIST))

# Function to check if a given string corresponds to a supported model
def is_supported_model(string):
    try:
        # Return the corresponding model name if it exists in the CLASSIFICATION_LABELS dictionary
        return CLASSIFICATION_LABELS[string]
    except KeyError:
        # Raise an exception if the model is unknown
        raise Exception("Unknown Model")

# Function to convert a model name string to the corresponding model object
def model_str_to_obj(string):
    # Ensure the model is supported
    assert is_supported_model(string)
    # Return the corresponding model object from the CLASSIFICATION_MODEL_DICT dictionary
    return CLASSIFICATION_MODEL_DICT[string]
