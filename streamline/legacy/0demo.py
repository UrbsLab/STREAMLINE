import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(str(Path(SCRIPT_DIR).parent.parent))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.modeling.utils import is_supported_model
print(is_supported_model("LR"))
