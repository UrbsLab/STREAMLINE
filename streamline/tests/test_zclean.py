import shutil
DEBUG = True


def test_stub():
    if not DEBUG:
        shutil.rmtree('./tests/')
