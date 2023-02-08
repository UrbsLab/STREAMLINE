import sys
import os
import shutil
import glob
import argparse


class Cleaner:
    """
    Phase 11 of STREAMLINE (Optional)- This 'Main' script runs Phase 11 which deletes all
    temporary files in pipeline output folder.
    This script is not necessary to run, but serves as a convenience to reduce
    space and clutter following a pipeline run.
    """

    def __init__(self, output_path, experiment_name, del_time=True, del_old_cv=True):
        """
        Cleaner Class
        Args:
            output_path: path to output directory
            experiment_name: name of experiment output folder (no spaces)
            del_time: delete individual run-time files (but save summary), default=True
            del_old_cv: delete any of the older versions of CV training and \
                        testing datasets not overwritten (preserves final training and testing datasets, default=True
        """

        self.output_path = output_path
        self.experiment_name = experiment_name
        self.experiment_path = self.output_path + '/' + self.experiment_name
        self.del_time = del_time
        self.del_old_cv = del_old_cv

        if not os.path.exists(self.output_path):
            raise Exception("Provided output_path does not exist")
        if not os.path.exists(self.experiment_path):
            raise Exception("Provided experiment name in given output_path does not exist")

    def run(self):
        # Get dataset paths for all completed dataset analyses in experiment folder
        datasets = os.listdir(self.experiment_path)
        remove_ist = ['metadata.pickle', 'metadata.csv', 'algInfo.pickle', 'jobsCompleted', 'logs', 'jobs',
                      'DatasetComparisons', 'UsefulNotebooks', self.experiment_name + '_ML_Pipeline_Report.pdf']
        for text in remove_ist:
            if text in datasets:
                datasets.remove(text)

        # Delete log folder/files
        self.rm_tree(self.experiment_path + '/' + 'logs')
        # Delete job folder/files
        self.rm_tree(self.experiment_path + '/' + 'jobs')
        # Delete jobscompleted folder/files
        self.rm_tree(self.experiment_path + '/' + 'jobsCompleted')

        # Remake folders (empty) incase user wants to rerun scripts like pdf report from command line
        os.mkdir(self.experiment_path + '/jobsCompleted')
        os.mkdir(self.experiment_path + '/jobs')
        os.mkdir(self.experiment_path + '/logs')

        # Delete target files within each dataset subfolder
        for dataset in datasets:
            # Delete individual runtime files (save runtime summary generated in phase 6)
            if self.del_time:
                self.rm_tree(self.experiment_path + '/' + dataset + '/' + 'runtime')

            # Delete temporary feature importance pickle files
            # (only needed for phase 4 and then saved as summary files in phase 6)
            self.rm_tree(self.experiment_path + '/' + dataset + '/feature_selection/mutualinformation/pickledForPhase4')
            self.rm_tree(self.experiment_path + '/' + dataset + '/feature_selection/multisurf/pickledForPhase4')

            # Delete older training and testing CV datasets (does not delete any
            # final versions used for training). Older cv datasets might have been
            # kept to see what they look like prior to preprocessing and feature selection.
            if self.del_old_cv:
                # Delete CV files generated after preprocessing but before feature selection
                files = glob.glob(self.experiment_path + '/' + dataset + '/CVDatasets/*CVOnly*')
                for f in files:
                    self.rm_tree(f, False)
                # Delete CV files generated after CV partitioning but before preprocessing
                files = glob.glob(self.experiment_path + '/' + dataset + '/CVDatasets/*CVPre*')
                for f in files:
                    self.rm_tree(f, False)

    @staticmethod
    def rm_tree(path, folder=True):
        try:
            if folder:
                if os.path.exists(path):
                    shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    # No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str,
                        help='name of experiment output folder (no spaces)')
    parser.add_argument('--del-time', dest='del_time', type=str,
                        help='delete individual run-time files (but save summary)', default="True")
    parser.add_argument('--del-oldCV', dest='del_old_cv', type=str,
                        help='delete any of the older versions of CV training and testing datasets not overwritten ('
                             'preserves final training and testing datasets)',
                        default="True")

    options = parser.parse_args(sys.argv[1:])
    cleaner = Cleaner(options.output_path, options.experiment_name,
                      options.del_time, options.del_old_cv)
