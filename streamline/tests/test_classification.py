import os
import time
import optuna
import pytest
import logging
from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.imputation_runner import ImputationRunner
from streamline.runners.feature_runner import FeatureImportanceRunner
from streamline.runners.feature_runner import FeatureSelectionRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner
from streamline.runners.compare_runner import CompareRunner
from streamline.runners.report_runner import ReportRunner
from streamline.runners.replicate_runner import ReplicationRunner

# pytest.skip("Tested Already", allow_module_level=True)

algorithms, run_parallel, output_path = ["MI", "MS"], False, "./tests/"
dataset_path, rep_data_path, experiment_name = "./data/DemoData/", "./data/DemoRepData/", "demo",
model_algorithms = ["LR", "NB", "DT"]


def test_classification():
    start = time.time()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    eda = DataProcessRunner(dataset_path, output_path, experiment_name,
                            exploration_list=None,
                            plot_list=None,
                            outcome_label="Class", instance_label="InstanceID", n_splits=3, ignore_features=None,
                            categorical_features=['Gender', 'Symptoms ', 'Alcohol', 'Hepatitis B Surface Antigen',
                                                  'Hepatitis B e Antigen', 'Hepatitis B Core Antibody',
                                                  'Hepatitis C Virus Antibody', 'Cirrhosis',
                                                  'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity',
                                                  'Hemochromatosis', 'Arterial Hypertension',
                                                  'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus',
                                                  'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly',
                                                  'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metastasis',
                                                  'Radiological Hallmark',
                                                  'Sim_Cat_2', 'Sim_Cat_3', 'Sim_Cat_4', 'Sim_Text_Cat_2',
                                                  'Sim_Text_Cat_3', 'Sim_Text_Cat_4'],
                            quantitative_features=['Sim_Miss_0.6', 'Alkaline phosphatase (U/L)',
                                                   'Aspartate transaminase (U/L)',
                                                   'International Normalised Ratio*', 'Performance Status*',
                                                   'Sim_Cor_-1.0_B',
                                                   'Alanine transaminase (U/L)', 'Platelets',
                                                   'Direct Bilirubin (mg/dL)', 'Encephalopathy degree*',
                                                   'Sim_Cor_0.9_A', 'Albumin (mg/dL)', 'Number of Nodules',
                                                   'Sim_Cor_1.0_A', 'Haemoglobin (g/dL)',
                                                   'Major dimension of nodule (cm)', 'Leukocytes(G/L)',
                                                   'Total Proteins (g/dL)', 'Sim_Miss_0.7',
                                                   'Ascites degree*', 'Creatinine (mg/dL)', 'Iron', 'Sim_Cor_0.9_B',
                                                   'Grams of Alcohol per day',
                                                   'Sim_Cor_-1.0_A', 'Oxygen Saturation (%)',
                                                   'Gamma glutamyl transferase (U/L)', 'Total Bilirubin(mg/dL)',
                                                   'Ferritin (ng/mL)', 'Packs of cigarets per year',
                                                   'Mean Corpuscular Volume', 'Sim_Cor_1.0_B',
                                                   'Alpha-Fetoprotein (ng/mL)'],
                            correlation_removal_threshold=1)
    eda.run(run_parallel=run_parallel)
    del eda

    dpr = ImputationRunner(output_path, experiment_name,
                           outcome_label="Class", instance_label="InstanceID")
    dpr.run(run_parallel=run_parallel)
    del dpr

    f_imp = FeatureImportanceRunner(output_path, experiment_name,
                                    outcome_label="Class", instance_label="InstanceID",
                                    algorithms=algorithms)
    f_imp.run(run_parallel=run_parallel)
    del f_imp

    f_sel = FeatureSelectionRunner(output_path, experiment_name,
                                   outcome_label="Class", instance_label="InstanceID",
                                   algorithms=algorithms)
    f_sel.run(run_parallel=run_parallel)
    del f_sel

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    runner = ModelExperimentRunner(output_path, experiment_name, model_algorithms,
                                   outcome_label="Class", instance_label="InstanceID")
    runner.run(run_parallel=run_parallel)
    del runner

    stats = StatsRunner(output_path, experiment_name,
                        outcome_label="Class", instance_label="InstanceID")
    stats.run(run_parallel=run_parallel)
    del stats

    compare = CompareRunner(output_path, experiment_name,
                            outcome_label="Class", instance_label="InstanceID")
    compare.run(run_parallel=run_parallel)
    del compare

    report = ReportRunner(output_path, experiment_name)
    report.run(run_parallel=run_parallel)
    del report

    repl = ReplicationRunner('./data/DemoRepData', dataset_path + 'hcc-data_example_custom.csv',
                             output_path, experiment_name)
    repl.run(run_parallel=run_parallel)

    report = ReportRunner(output_path, experiment_name,
                          training=False, rep_data_path=rep_data_path,
                          dataset_for_rep=dataset_path + 'hcc-data_example_custom.csv')
    report.run(run_parallel)

    logging.warning("Ran Pipeline in " + str(time.time() - start))
