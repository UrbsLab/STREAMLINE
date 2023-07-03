import pytest
import shutil
from streamline.utils.dataset import Dataset
from streamline.dataprep.data_process import DataProcess

pytest.skip("Tested Already", allow_module_level=True)


@pytest.mark.parametrize(
    ("dataset_path", "outcome_label", "match_label", "instance_label", "exception"),
    [
        ("", None, None, None, Exception),
        ("./hcc-data_example.csv", "something", None, None, FileNotFoundError),
        ("./hcc-data_example.csv", "something", None, None, Exception),
        ("./hcc-data_example.csv", "something", "otherthing", None, Exception),
        ("./hcc-data_example.csv", "something", None, "otherthing", Exception),
    ],
)
def test_invalid_datapath(dataset_path, outcome_label, match_label, instance_label, exception):
    with pytest.raises(exception):
        Dataset(dataset_path, outcome_label, match_label, instance_label)


@pytest.mark.parametrize(
    ("dataset_path", "outcome_label", "match_label", "instance_label"),
    [
        ("./DemoData/hcc-data_example.csv", "Class", None, None),
    ],
)
def test_valid_dataset(dataset_path, outcome_label, match_label, instance_label):
    dataset = Dataset(dataset_path, outcome_label, match_label, instance_label)
    drop_list = [outcome_label, ]
    if match_label:
        drop_list.append(match_label)
    if instance_label:
        drop_list.append(instance_label)
    assert (outcome_label in dataset.data.columns)
    assert (dataset.feature_only_data().equals(dataset.data.drop(drop_list, axis=1)))
    assert (dataset.get_outcome().equals(dataset.data[dataset.outcome_label]))
    dataset.clean_data(None)
    dataset.set_headers('./tests/')
    shutil.rmtree('./tests/')


@pytest.mark.parametrize(
    ("dataset", "experiment_path", "exception"),
    [
        ("", "./test/", Exception),
        ("../sdsad.txt", "./test/", Exception),
    ],
)
def test_invalid_eda(dataset, experiment_path, exception):
    with pytest.raises(exception):
        DataProcess(dataset, experiment_path)


def test_invalid_eda_2():
    dataset, experiment_path = "./DemoData/hcc-data_example.csv", "./tests/"
    explorations = ["sdasd"]
    plots = ["dsfsdf"]
    with pytest.raises(Exception):
        DataProcess(dataset, experiment_path, explorations=explorations)
    with pytest.raises(Exception):
        DataProcess(dataset, experiment_path, plots=plots)


def test_valid_eda():
    dataset = Dataset("./DemoData/hcc-data_example.csv", "Class", None, "InstanceID")
    eda = DataProcess(dataset, "./tests/")
    eda.make_log_folders()
    assert (eda.dataset.data.equals(dataset.data))
    eda.drop_ignored_rowcols()
    assert (eda.dataset.data.equals(dataset.data))
    categorical_variables = eda.identify_feature_types()

    test_cv = ['Gender', 'Symptoms ', 'Alcohol', 'Hepatitis B Surface Antigen',
               'Hepatitis B e Antigen', 'Hepatitis B Core Antibody', 'Hepatitis C Virus Antibody',
               'Cirrhosis', 'Endemic Countries', 'Smoking', 'Diabetes', 'Obesity', 'Hemochromatosis',
               'Arterial Hypertension', 'Chronic Renal Insufficiency', 'Human Immunodeficiency Virus',
               'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly', 'Portal Hypertension',
               'Portal Vein Thrombosis', 'Liver Metastasis', 'Radiological Hallmark', 'Performance Status*',
               'Encephalopathy degree*', 'Ascites degree*', 'Number of Nodules']

    assert (categorical_variables == test_cv)
    eda.dataset.describe_data("./tests/")
    eda.dataset.missingness_counts("./tests/")
    eda.dataset.missing_count_plot("./tests/")
    eda.counts_summary()
    eda.univariate_analysis()
    eda.univariate_plots()
    shutil.rmtree('./tests/')


def test_valid_eda_general():
    dataset = Dataset("./DemoData/hcc-data_example.csv", "Class", None, "InstanceID")
    eda = DataProcess(dataset, "./tests/")
    eda.run()
    shutil.rmtree('./tests/')
