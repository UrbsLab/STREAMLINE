import logging
import os
from pathlib import Path

def load_class_from_folder(model_type="BinaryClassification"):
    """
    Load classes from a specified folder based on the model type.
    
    Args:
    model_type (str): Type of model to load (BinaryClassification, MulticlassClassification, Regression).
    
    Returns:
    list: Sorted list of loaded classes based on their model_name attribute.
    """
    folder_path, package_path = None, None

    # Determine the folder path and package path based on the model type
    if model_type == "BinaryClassification":
        folder_path = os.path.join(Path(__file__).parent.parent, 'models/binary_classification')
        package_path = 'streamline.models.binary_classification'
    elif model_type == "MulticlassClassification":
        folder_path = os.path.join(Path(__file__).parent.parent, 'models/multiclass_classification')
        package_path = 'streamline.models.multiclass_classification'
    elif model_type == "Regression":
        folder_path = os.path.join(Path(__file__).parent.parent, 'models/regression')
        package_path = 'streamline.models.regression'

    classes = list()
    
    # Iterate over Python files in the specified folder, excluding '__init__.py'
    for py in [f[:-3] for f in os.listdir(folder_path) if f.endswith('.py') and f != '__init__.py']:
        # Dynamically import the module
        mod = __import__('.'.join([package_path, py]), fromlist=[py])
        # Get all class objects defined in the module
        classes_list = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        
        for cls in classes_list:
            # Filter classes to include only those related to 'streamline' and exclude 'basemodel' or 'submodels'
            if ('streamline' in str(cls)) and not ('basemodel' in str(cls) or 'submodels' in str(cls)):
                classes.append(cls)
    
    # Return the sorted list of classes based on their model_name attribute
    return sorted(classes, key=lambda x: x.model_name)
