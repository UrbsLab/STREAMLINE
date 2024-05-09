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
    stl_runner.set_logger()
    # logging.warn(stl_runner.params)
    stl_runner.run()
    logging.warn("FINISHED!!")

