import os
import sys
import pickle
from pathlib import Path

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory of the script to the system path
# This allows importing modules from two levels up
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

# Import the StatsJob class from the streamline.postanalysis.statistics module
from streamline.postanalysis.statistics import StatsJob

# Main function to run the statistics job
def run_cluster(argv):
    # Get the path to the parameter file from the command line arguments
    param_path = argv[1]
    # Open the parameter file in binary read mode
    with open(param_path, "rb") as input_file:
        # Load the parameters from the file using pickle
        params = pickle.load(input_file)
    # Update the global variables with the parameters from the file
    globals().update(params)

    # Create an instance of the StatsJob class with the loaded parameters
    job_obj = StatsJob(full_path, outcome_label, outcome_type, instance_label, scoring_metric,
                       len_cv, top_features, sig_cutoff, metric_weight, scale_data,
                       exclude_plots,
                       show_plots)
    # Run the statistics job
    job_obj.run()

# If the script is executed directly, run the run_cluster function with command line arguments
if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
