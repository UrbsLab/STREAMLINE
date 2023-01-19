import os
import time
import logging


class Job:
    def __init__(self):
        self.cluster = None
        self.job_start_time = time.time()

    def run_local(self):
        pass

    def run_cluster(self):
        pass
