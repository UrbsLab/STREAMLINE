import os
from pathlib import Path


def load_class_from_folder(path=None):
    if path is None:
        path = os.path.join(Path(__file__).parent.parent, 'models/')

    classes = list()
    for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
        mod = __import__('.'.join(['streamline.models', py]), fromlist=[py])
        classes_list = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for cls in classes_list:
            if ('streamline' in str(cls)) and not ('basemodel' in str(cls)):
                classes.append(cls)
    # logging.warning(classes)
    return classes
