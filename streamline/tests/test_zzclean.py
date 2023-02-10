import os
import pytest
import shutil
DEBUG = False

# pytest.skip("Tested Already", allow_module_level=True)


def test_stub():
    if not DEBUG:
        if os.path.exists('/tests/'):
            shutil.rmtree('./tests/')
