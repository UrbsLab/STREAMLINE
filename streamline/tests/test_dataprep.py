import pytest
import shutil
from streamline.utils.dataset import Dataset
from streamline.dataprep.exploratory_analysis import ExploratoryDataAnalysis


@pytest.mark.parametrize(
    ("dataset_path", "class_label", "match_label", "instance_label", "exception"),
    [
        ("", None, None, None, Exception),
        ("./demodata.csv", "something", None, None, FileNotFoundError),
        ("./DemoData.csv", "something", None, None, Exception),
        ("./DemoData.csv", "something", "otherthing", None, Exception),
        ("./DemoData.csv", "something", None, "otherthing", Exception),
    ],
)
def test_invalid_datapath(dataset_path, class_label, match_label, instance_label, exception):
    with pytest.raises(exception):
        Dataset(dataset_path, class_label, match_label, instance_label)


@pytest.mark.parametrize(
    ("dataset_path", "class_label", "match_label", "instance_label"),
    [
        ("./DemoData/demodata.csv", "Class", None, None),
    ],
)
def test_valid_dataset(dataset_path, class_label, match_label, instance_label):
    dataset = Dataset(dataset_path, class_label, match_label, instance_label)
    drop_list = [class_label, ]
    if match_label:
        drop_list.append(match_label)
    if instance_label:
        drop_list.append(instance_label)
    assert (class_label in dataset.data.columns)
    assert (dataset.feature_only_data().equals(dataset.data.drop(drop_list, axis=1)))
    assert (dataset.get_outcome().equals(dataset.data[dataset.class_label]))
    dataset.clean_data(None)
    dataset.set_headers('./tests/')
    # shutil.rmtree('./tests/')


@pytest.mark.parametrize(
    ("dataset", "experiment_path", "exception"),
    [
        ("", "./test/", Exception),
        ("../sdsad.txt", "./test/", Exception),
    ],
)
def test_invalid_eda(dataset, experiment_path, exception):
    with pytest.raises(exception):
        ExploratoryDataAnalysis(dataset, experiment_path)


def test_invalid_eda_2():
    dataset, experiment_path = "./DemoData/demodata.csv", "./tests/"
    explorations = ["sdasd"]
    plots = ["dsfsdf"]
    with pytest.raises(Exception):
        ExploratoryDataAnalysis(dataset, experiment_path, explorations=explorations)
    with pytest.raises(Exception):
        ExploratoryDataAnalysis(dataset, experiment_path, plots=plots)


def test_valid_eda():
    dataset = Dataset("./DemoData/demodata.csv", "Class", None, "InstanceID")
    eda = ExploratoryDataAnalysis(dataset, "./tests/")
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
    eda.describe_data()
    eda.missingness_counts()
    eda.missing_count_plot()
    eda.counts_summary()
    eda.feature_correlation_plot()
    eda.univariate_analysis()
    eda.univariate_plots()
    # shutil.rmtree('./tests/')


def test_valid_eda_general():
    dataset = Dataset("./DemoData/demodata.csv", "Class", None, "InstanceID")
    eda = ExploratoryDataAnalysis(dataset, "./tests/")
    eda.run()
    # shutil.rmtree('./tests/')
