import os
import sys
import pickle
from pathlib import Path

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory of the script to the system path
# This allows importing modules from two levels up
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

# Import the ScaleAndImpute class from the streamline.dataprep.scale_and_impute module
from streamline.dataprep.scale_and_impute import ScaleAndImpute


def run_cluster(argv):
    # Get the path to the parameter file from the command line arguments
    param_path = argv[1]
    # Open the parameter file in binary read mode
    with open(param_path, "rb") as input_file:
        # Load the parameters from the file using pickle
        params = pickle.load(input_file)
    # Update the global variables with the parameters from the file
    globals().update(params)
    # Construct the full output path for the experiment
    full_path = output_path + "/" + experiment_name

    # Create an instance of the ScaleAndImpute class with the loaded parameters
    job_obj = ScaleAndImpute(cv_train_path, cv_test_path,
                             full_path,
                             scale_data, impute_data, multi_impute, overwrite_cv,
                             outcome_label, instance_label, random_state)
    # Run the scaling and imputation process
    job_obj.run()


if __name__ == "__main__":
    # Execute the run_cluster function with command line arguments
    # and exit the program with the return value of run_cluster
    sys.exit(run_cluster(sys.argv))
