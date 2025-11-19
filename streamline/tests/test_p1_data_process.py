import os
import time
import pytest
import logging

from streamline.p1_data_process.p1_runner import P1Runner

pytest.skip("Tested Already", allow_module_level=True)

output_path = "./tests/"
dataset_path, experiment_name = "./data/DemoData/", "demo"

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
        categorical_features=[
            'Gender', 'Symptoms ', 'Alcohol', 'Hepatitis B Surface Antigen',
            'Hepatitis B e Antigen', 'Hepatitis B Core Antibody',
            'Hepatitis C Virus Antibody', 'Cirrhosis',
            'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity',
            'Hemochromatosis', 'Arterial Hypertension',
            'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus',
            'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly',
            'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis',
            'Radiological Hallmark', 'Sim_Cat_2', 'Sim_Cat_3', 'Sim_Cat_4',
            'Sim_Text_Cat_2', 'Sim_Text_Cat_3', 'Sim_Text_Cat_4'
        ],
        quantitative_features=[
            'Sim_Miss_0.6', 'Alkaline phosphatase (U/L)',
            'Aspartate transaminase (U/L)', 'International Normalised Ratio*',
            'Performance Status*', 'Sim_Cor_-1.0_B', 'Alanine transaminase (U/L)',
            'Platelets', 'Direct Bilirubin (mg/dL)', 'Encephalopathy degree*',
            'Sim_Cor_0.9_A', 'Albumin (mg/dL)', 'Number of Nodules',
            'Sim_Cor_1.0_A', 'Haemoglobin (g/dL)',
            'Major dimension of nodule (cm)', 'Leukocytes(G/L)',
            'Total Proteins (g/dL)', 'Sim_Miss_0.7', 'Ascites degree*',
            'Creatinine (mg/dL)', 'Iron', 'Sim_Cor_0.9_B',
            'Grams of Alcohol per day', 'Sim_Cor_-1.0_A',
            'Oxygen Saturation (%)', 'Gamma glutamyl transferase (U/L)',
            'Total Bilirubin(mg/dL)', 'Ferritin (ng/mL)',
            'Packs of cigarets per year', 'Mean Corpuscular Volume',
            'Sim_Cor_1.0_B', 'Alpha-Fetoprotein (ng/mL)'
        ],
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
