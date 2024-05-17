import sys
import warnings
import logging
import optuna
from streamline.runner import STREAMLINERunner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


if __name__ == '__main__':
    stl_runner = STREAMLINERunner()
    stl_runner.process_argv(sys.argv)
    logger = stl_runner.set_logger()
    # logging.warn(stl_runner.params)
    if stl_runner.checker:
        stl_runner.check_progress()
    else:
        stl_runner.run()
    logging.warning("FINISHED!!")
