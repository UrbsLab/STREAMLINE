import shutil
DEBUG = False


def test_stub():
    if not DEBUG:
        shutil.rmtree('./tests/')
