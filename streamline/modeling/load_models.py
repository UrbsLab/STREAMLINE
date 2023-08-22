import os
from pathlib import Path


def load_class_from_folder(model_type="BinaryClassification"):
    folder_path, package_path = None, None
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
    for py in [f[:-3] for f in os.listdir(folder_path) if f.endswith('.py') and f != '__init__.py']:

        mod = __import__('.'.join([package_path, py]), fromlist=[py])
        classes_list = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for cls in classes_list:
            if ('streamline' in str(cls)) and not ('basemodel' in str(cls) or 'submodels' in str(cls)):
                classes.append(cls)
    # logging.warning(classes)
    return sorted(classes, key=lambda x: x.model_name)
