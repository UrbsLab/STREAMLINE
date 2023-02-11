import os
import glob
import pickle
from joblib import Parallel, delayed

from streamline.dataprep.data_process import DataProcessing
from streamline.utils.runners import runner_fn


class DataProcessRunner:
    """
    Runner class for Data Processing Jobs of CV Splits
    """
    def __init__(self, output_path, experiment_name, scale_data=True, impute_data=True,
                 multi_impute=True, overwrite_cv=True, class_label="Class", instance_label=None, random_state=None):
        """

        Args:
            output_path:
            experiment_name:
            scale_data:
            impute_data:
            multi_impute:
            overwrite_cv:
        """
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.multi_impute = multi_impute
        self.overwrite_cv = overwrite_cv
        self.class_label = class_label
        self.instance_label = instance_label
        self.random_state = random_state

        # Argument checks-------------------------------------------------------------
        if not os.path.exists(self.output_path):
            raise Exception("Output path must exist (from phase 1) before phase 2 can begin")
        if not os.path.exists(self.output_path + '/' + self.experiment_name):
            raise Exception("Experiment must exist (from phase 1) before phase 2 can begin")

        self.save_metadata()

    def run(self, run_parallel=True):
        job_counter = 0
        job_list = []
        dataset_paths = os.listdir(self.output_path + "/" + self.experiment_name)
        remove_list = ['metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'jobsCompleted',
                       'logs', 'jobs', 'DatasetComparisons']
        for text in remove_list:
            if text in dataset_paths:
                dataset_paths.remove(text)

        for dataset_directory_path in dataset_paths:
            full_path = self.output_path + "/" + self.experiment_name + "/" + dataset_directory_path

            # Create folder to store scaling and imputing files
            if not os.path.exists(full_path + '/scale_impute/'):
                os.makedirs(full_path + '/scale_impute/')

            for cv_train_path in glob.glob(full_path + "/CVDatasets/*Train.csv"):
                job_counter += 1
                cv_test_path = cv_train_path.replace("Train.csv", "Test.csv")
                if run_parallel:
                    job_obj = DataProcessing(cv_train_path, cv_test_path,
                                             self.output_path + "/" + self.experiment_name,
                                             self.scale_data, self.impute_data, self.multi_impute, self.overwrite_cv,
                                             self.class_label, self.instance_label, self.random_state)
                    # p = multiprocessing.Process(target=runner_fn, args=(job_obj, ))
                    job_list.append(job_obj)
                else:
                    job_obj = DataProcessing(cv_train_path, cv_test_path,
                                             self.output_path + "/" + self.experiment_name,
                                             self.scale_data, self.impute_data, self.multi_impute, self.overwrite_cv,
                                             self.class_label, self.instance_label, self.random_state)
                    job_obj.run()
        if run_parallel:
            Parallel()(delayed(runner_fn)(job_obj) for job_obj in job_list)

    def save_metadata(self):
        file = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'rb')
        metadata = pickle.load(file)
        file.close()
        metadata['Use Data Scaling'] = self.scale_data
        metadata['Use Data Imputation'] = self.impute_data
        metadata['Use Multivariate Imputation'] = self.multi_impute
        # Pickle the metadata for future use
        pickle_out = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb')
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
