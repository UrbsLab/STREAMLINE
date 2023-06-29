import glob
from os.path import dirname, basename, isfile, join
from pathlib import Path

name = __name__
modules = glob.glob(join(dirname(__file__), "*.py"))
modules = [str(Path(path)) for path in modules]
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
