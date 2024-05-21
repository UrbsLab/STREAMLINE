import os
import logging
from pathlib import Path

def load_class_from_folder(path=None):
    """
    Load classes from a specified folder. If no path is provided, 
    defaults to the 'featurefns/algorithms' directory in the parent folder.

    Args:
    path (str): The path to the folder from which to load the classes.

    Returns:
    list: Sorted list of loaded classes based on their model_name attribute.
    """
    if path is None:
        # Default path to 'featurefns/algorithms' directory in the parent folder of the current file
        path = os.path.join(Path(__file__).parent.parent, 'featurefns/algorithms')

    classes = list()
    
    # Iterate over Python files in the specified directory, excluding '__init__.py'
    for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
        # Dynamically import the module
        mod = __import__('.'.join(['streamline.featurefns.algorithms', py]), fromlist=[py])
        
        # Retrieve all class objects defined in the module
        classes_list = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        
        # Filter classes: include those in 'streamline' but exclude 'FeatureAlgorithm'
        for cls in classes_list:
            if ('streamline' in str(cls)) and not ('FeatureAlgorithm' in str(cls)):
                classes.append(cls)
    
    # Optional logging for debugging purposes
    # logging.warning(classes)
    
    # Return the classes sorted by their model_name attribute
    return sorted(classes, key=lambda x: x.model_name)
