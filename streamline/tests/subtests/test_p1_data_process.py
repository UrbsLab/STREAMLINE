import os
import time
import pytest
import logging

from streamline.p1_data_process.p1_runner import P1Runner

pytest.skip("Tested Already", allow_module_level=True)

output_path = "./tests/"
dataset_path, experiment_name = "./data/UCIBinaryClassification/", "demo"

# NEW: choose execution mode
#   False      -> serial
#   "Local"    -> local Dask parallel
#   "BashSLURM"/"BashLSF" -> submit bash jobs
#   "<DaskClusterName>"   -> use get_cluster(...) for remote cluster
run_cluster = "Local"


def test_classification():
    start = time.time()
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    eda = P1Runner(
        dataset_path,
        output_path,
        experiment_name,
        exclude_eda_output=['correlation'],
        outcome_label="Class",
        instance_label="InstanceID",
        n_splits=3,
        ignore_features=None,
        categorical_features="./data/UCIFeatureTypes/hcc_survival_categorical_features.csv",
        quantitative_features="./data/UCIFeatureTypes/hcc_survival_quantitative_features.csv",
        correlation_removal_threshold=1,
        run_cluster=run_cluster,  # NEW
        force=1,
    )
    eda.run()
    del eda

    logging.warning("Ran Pipeline in " + str(time.time() - start))

    # dpr = ImputationRunner(
    #     output_path, experiment_name,
    #     outcome_label="Class",
    #     instance_label="InstanceID",
    #     run_cluster=run_cluster,  # NEW
    # )
    # dpr.run()
    # del dpr

    # f_imp = FeatureImportanceRunner(
    #     output_path, experiment_name,
    #     outcome_label="Class",
    #     instance_label="InstanceID",
    #     algorithms=algorithms,
    #     run_cluster=run_cluster,  # NEW
    # )
    # f_imp.run()
    # del f_imp

    # f_sel = FeatureSelectionRunner(
    #     output_path, experiment_name,
    #     outcome_label="Class",
    #     instance_label="InstanceID",
    #     algorithms=algorithms,
    #     run_cluster=run_cluster,  # NEW
    # )
    # f_sel.run()
    # del f_sel
