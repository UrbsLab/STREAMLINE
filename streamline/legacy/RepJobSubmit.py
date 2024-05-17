import os
import sys
import pickle
from pathlib import Path

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory of the script to the system path
# This allows importing modules from two levels up
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

# Import the ReplicateJob class from the streamline.postanalysis.model_replicate module
from streamline.postanalysis.model_replicate import ReplicateJob

# Main function to run the replication job
def run_cluster(argv):
    # Get the path to the parameter file from the command line arguments
    param_path = argv[1]
    # Open the parameter file in binary read mode
    with open(param_path, "rb") as input_file:
        # Load the parameters from the file using pickle
        params = pickle.load(input_file)
    # Update the global variables with the parameters from the file
    globals().update(params)

    # Commented-out code for loading additional information from pickle files
    # Uncomment if needed for further customization
    # file = open(experiment_path + '/' + "algInfo.pickle", 'rb')
    # alg_info = pickle.load(file)
    # file.close()
    # temp_algo = []
    # for key in alg_info:
    #     if alg_info[key][0]:
    #         temp_algo.append(key)
    # algorithms = temp_algo
    # file = open(experiment_path + '/' + "metadata.pickle", 'rb')
    # metadata = pickle.load(file)
    # file.close()
    # ignore_features = metadata['Ignored Features']

    # Create an instance of the ReplicateJob class with the loaded parameters
    job_obj = ReplicateJob(dataset_filename, dataset_for_rep, full_path, outcome_label, outcome_type, instance_label,
                           match_label, ignore_features, cv_partitions,
                           exclude_plots,
                           categorical_cutoff, sig_cutoff, scale_data, impute_data,
                           multi_impute, show_plots, scoring_metric, random_state)
    # Run the replication job
    job_obj.run()

# If the script is executed directly, run the run_cluster function with command line arguments
if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
