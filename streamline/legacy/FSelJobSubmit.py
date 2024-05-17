import os
import sys
import pickle
from pathlib import Path

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory of the script to the system path
# This allows importing modules from two levels up
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

# Import the FeatureSelection class from the streamline.featurefns.selection module
from streamline.featurefns.selection import FeatureSelection

# Main function to run the feature selection process
def run_cluster(argv):
    # Get the path to the parameter file from the command line arguments
    param_path = argv[1]
    # Open the parameter file in binary read mode
    with open(param_path, "rb") as input_file:
        # Load the parameters from the file using pickle
        params = pickle.load(input_file)
    # Update the global variables with the parameters from the file
    globals().update(params)

    # Create an instance of the FeatureSelection class with the loaded parameters
    job_obj = FeatureSelection(full_path, n_datasets, algorithms,
                               outcome_label, instance_label, export_scores,
                               top_features, max_features_to_keep,
                               filter_poor_features, overwrite_cv)
    # Run the feature selection process
    job_obj.run()

# If the script is executed directly, run the run_cluster function with command line arguments
if __name__ == "__main__":
    # Exit the script with the status code returned by run_cluster
    sys.exit(run_cluster(sys.argv))
