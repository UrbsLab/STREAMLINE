import os

from streamline.utils.job import Job


class ModelJob(Job):
    def __init__(self):
        self.name = None
        self.output_path = None
        self.experiment_name = None
        self.metric = None

        # Argument checks
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 5 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")
