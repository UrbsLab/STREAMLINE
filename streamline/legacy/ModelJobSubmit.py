import os
import pickle
import sys
from pathlib import Path

# Determine the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the grandparent directory of the script to the system path
# This allows importing modules from two levels up
sys.path.append(str(Path(SCRIPT_DIR).parent.parent))

# Import necessary classes and functions from the streamline package
from streamline.modeling.modeljob import ModelJob
from streamline.modeling.utils import get_fi_for_ExSTraCS

# Main function to run the model training and evaluation
def run_cluster(argv):
    # Get the path to the parameter file from the command line arguments
    param_path = argv[1]
    # Open the parameter file in binary read mode
    with open(param_path, "rb") as input_file:
        # Load the parameters from the file using pickle
        params = pickle.load(input_file)
    # Update the global variables with the parameters from the file
    globals().update(params)
    print(params)
    print(vars())

    # Commented code for conditional imports based on outcome type with GlobalImport class
    # Uncomment if needed for different outcome types
    # if outcome_type == "Binary":
    #     with GlobalImport() as gi:
    #         from streamline.modeling.classification_utils import model_str_to_obj
    #         gi()
    # elif outcome_type == "Continuous":
    #     if scoring_metric == 'balanced_accuracy':
    #         scoring_metric = 'explained_variance'
    #     with GlobalImport() as gi:
    #         from streamline.modeling.regression_utils import model_str_to_obj
    #         gi()
    # elif outcome_type == "Multiclass":
    #     # logging.info("Using Multiclass Classification Models")
    #     with GlobalImport() as gi:
    #         from streamline.modeling.multiclass_utils import model_str_to_obj
    #         gi()
    # else:
    #     raise Exception("Unknown Outcome Type:" + str(outcome_type))

    # Load metadata from a previously saved pickle file
    file = open(output_path + '/' + experiment_name + '/' + "metadata.pickle", 'rb')
    metadata = pickle.load(file)
    filter_poor_features = metadata['Filter Poor Features']
    outcome_type = metadata['Outcome Type']
    file.close()
    
    dataset_directory_path = full_path.split('/')[-1]

    # Import the appropriate model function based on the outcome type
    if outcome_type == "Binary":
        from streamline.modeling.classification_utils import model_str_to_obj
    elif outcome_type == "Multiclass":
        from streamline.modeling.multiclass_utils import model_str_to_obj
    elif outcome_type == "Continuous":
        from streamline.modeling.regression_utils import model_str_to_obj
    else:
        raise Exception("Unknown Outcome Type:" + str(outcome_type))

    # Create an instance of the ModelJob class with the loaded parameters
    job_obj = ModelJob(full_path, output_path, experiment_name, cv_count, outcome_label,
                       instance_label, scoring_metric, metric_direction, n_trials,
                       timeout, training_subsample, uniform_fi, save_plots, random_state)

    # Initialize the model based on the specified algorithm
    if algorithm not in ['eLCS', 'XCS', 'ExSTraCS']:
        # Standard model initialization
        model = model_str_to_obj(algorithm)(cv_folds=3,
                                            scoring_metric=scoring_metric,
                                            metric_direction=metric_direction,
                                            random_state=random_state,
                                            cv=None, n_jobs=n_jobs)
    else:
        # Special handling for LCS algorithms
        if algorithm == 'ExSTraCS':
            # Get expert knowledge for ExSTraCS
            expert_knowledge = get_fi_for_ExSTraCS(output_path, experiment_name,
                                                   dataset_directory_path,
                                                   outcome_label, instance_label, cv_count,
                                                   filter_poor_features)
            if do_lcs_sweep:
                # Initialize ExSTraCS with LCS sweep
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs,
                                                    expert_knowledge=expert_knowledge)
            else:
                # Initialize ExSTraCS with specific parameters
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs,
                                                    iterations=lcs_iterations,
                                                    N=lcs_n, nu=lcs_nu,
                                                    expert_knowledge=expert_knowledge)
        else:
            # Initialize other LCS models
            if do_lcs_sweep:
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs)
            else:
                model = model_str_to_obj(algorithm)(cv_folds=3,
                                                    scoring_metric=scoring_metric,
                                                    metric_direction=metric_direction,
                                                    random_state=random_state,
                                                    cv=None, n_jobs=n_jobs,
                                                    iterations=lcs_iterations,
                                                    N=lcs_n, nu=lcs_nu)

    # Run the model job with the initialized model
    job_obj.run(model)

# If the script is executed directly, run the run_cluster function with command line arguments
if __name__ == "__main__":
    sys.exit(run_cluster(sys.argv))
