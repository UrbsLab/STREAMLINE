import os
import sys
import pickle
from pathlib import Path

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory of the script to the system path
# This allows importing modules from two levels up
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

# Import the FeatureImportance class from the streamline.featurefns.importance module
from streamline.featurefns.importance import FeatureImportance

# Main function to run the feature importance analysis
def run_cluster(argv):
    # Get the path to the parameter file from the command line arguments
    param_path = argv[1]
    # Open the parameter file in binary read mode
    with open(param_path, "rb") as input_file:
        # Load the parameters from the file using pickle
        params = pickle.load(input_file)
    # Update the global variables with the parameters from the file
    globals().update(params)

    # Create an instance of the FeatureImportance class with the loaded parameters
    job_obj = FeatureImportance(cv_train_path, experiment_path, outcome_label,
                                instance_label, instance_subset, algorithm,
                                use_turf, turf_pct, random_state, n_jobs)
    # Run the feature importance analysis
    job_obj.run()

# If the script is executed directly, run the run_cluster function with command line arguments
if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
