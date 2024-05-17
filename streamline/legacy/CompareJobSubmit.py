import os
import sys
import pickle
from pathlib import Path

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory to the system path to allow importing modules from there
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

# Import the CompareJob class from the specified module
from streamline.postanalysis.dataset_compare import CompareJob

def run_cluster(argv):
    # The first argument is expected to be the path to a parameters file
    param_path = argv[1]
    
    # Load the parameters from the specified file using pickle
    with open(param_path, "rb") as input_file:
        params = pickle.load(input_file)
    
    # Update the global namespace with the loaded parameters
    globals().update(params)

    # Instantiate the CompareJob class with the loaded parameters
    job_obj = CompareJob(output_path, experiment_name, experiment_path,
                         outcome_label, outcome_type, instance_label, sig_cutoff, show_plots)
    # Run the job
    job_obj.run()

# If the script is run as the main module, execute the run_cluster function
if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
