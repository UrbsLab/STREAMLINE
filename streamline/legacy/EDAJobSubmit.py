import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

from streamline.utils.dataset import Dataset
from streamline.dataprep.data_process import DataProcess

class dotdict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

def save_metadata(self):
        metadata = dict()
        metadata['Data Path'] = self.data_path
        metadata['Output Path'] = self.output_path
        metadata['Experiment Name'] = self.experiment_name
        metadata['Outcome Label'] = self.outcome_label
        metadata['Outcome Type'] = self.outcome_type
        metadata['Instance Label'] = self.instance_label
        metadata['Match Label'] = self.match_label
        metadata['Ignored Features'] = self.ignore_features
        metadata['Specified Categorical Features'] = self.categorical_features
        metadata['Specified Quantitative Features'] = self.quantitative_features
        metadata['CV Partitions'] = self.n_splits
        metadata['Partition Method'] = self.partition_method
        metadata['Categorical Cutoff'] = self.categorical_cutoff
        metadata['Statistical Significance Cutoff'] = self.sig_cutoff
        metadata['Engineering Missingness Cutoff'] = self.featureeng_missingness
        metadata['Cleaning Missingness Cutoff'] = self.cleaning_missingness
        metadata['Correlation Removal Threshold'] = self.correlation_removal_threshold
        metadata['List of Exploratory Analysis Ran'] = self.exploration_list
        metadata['List of Exploratory Plots Saved'] = self.plot_list
        metadata['Random Seed'] = self.random_state
        metadata['Run From Notebook'] = self.show_plots
        # Pickle the metadata for future use
        pickle_out = open(self.output_path + '/' + self.experiment_name + '/' + "metadata.pickle", 'wb')
        pickle.dump(metadata, pickle_out)
        pickle_out.close()


def run_cluster(argv):
    param_path = argv[1]
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    globals().update(params)
    try:
        dataset = Dataset(dataset_path, outcome_label, match_label, instance_label, outcome_type)
    except Exception:
        dataset = (dataset_path, outcome_label, match_label, instance_label, outcome_type)
        save_metadata(dotdict(params))

    eda_obj = DataProcess(dataset, output_path + '/' + experiment_name,
                          ignore_features,
                          categorical_features, quantitative_features, exclude_eda_output,
                          categorical_cutoff, sig_cutoff, featureeng_missingness,
                          cleaning_missingness, correlation_removal_threshold, partition_method, n_splits,
                          random_state)
    eda_obj.run(top_features)
    


if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
