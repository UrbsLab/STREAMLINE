from streamline.utils.cleanup import Cleaner


class CleanRunner:
    def __init__(self, output_path, experiment_name, del_time=True, del_old_cv=True):
        self.clean = Cleaner(output_path, experiment_name, del_time, del_old_cv)

    def run(self, run_parallel=None):
        self.clean.run()
