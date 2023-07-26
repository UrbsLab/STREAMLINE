# Datasets

## Input Data Requirements
Here we specify the formatting requirements for datasets when running STREAMLINE. 
1. Dataset files are in comma-separated or tab-delimited format with the extension `.txt`, `.csv`, or `.tsv`.
2. Data columns should represent variables, and rows should represent instances (i.e. samples).
3. Any missing values in the dataset should be left blank (i.e. NaN) or indicated with the text 'NA'.
    * Do not leave placeholder values for missing values such as 99, -99, or text other than 'NA'.
4. Dataset files should include a header that gives column lables.
5. Data columns should only include the following (column order does not matter):
    * Outcome/Class Label (i.e. the dependant variable) - column indicated by `class_label`
    * Instance Label (i.e. unique identifiers for each instance/row in the dataset) \[Optional] - column indicated by `instance_label`
    * Match Label (i.e. an instance group identifier used to keep instances of the same group together during k-fold CV using the group stratification option) column indicated by `match_label`
    * Features (i.e. independant variables) - all other columns in dataset are assumed to be features (except those excluded using `ignore_features_path`)
6. The outcome/class column includes only two possible values (i.e. a binary outcome) [Note: STREAMLINE will soon be expanded to allow for multi-class and quantiative outcomes]
7. If multiple target datasets are being analyzed they must each have the same `class_label` (e.g. 'Class'), and (if present), the same `instance_label` (e.g. 'ID') and `match_label` (e.g. 'Match_ID'). The same is true for any 'replication datasets' (if present) when using Phase 8. 

### Additional Notes
**Specifying Feature Types:** Users are strongly encouraged to specify which features should be treated as categorical or quantitative using `quantitative_feature_path` or `categorical_feature_path`, or setting the `categorical_cutoff` to a value that will correctly assign feature types automatically (i.e. if all features in the dataset have 3 possible values and `categorical_cutoff` is set to 4, all features would be treated as categorical). If multiple target datasets are being analyzed, then `quantitative_feature_path` or `categorical_feature_path` should include the names of all features to be treated as categorical vs. quanatative across all datasets. 

**Text-valued Features:** STREAMLINE allows datasets to be loaded that have text-valued (i.e. non-numeric) entries. However be aware of how the pipeline will treat different columns:
* Outcome Label - if text, will be numerically encoded (0 or 1) based on alphabetical order 
* Instance Label - is commonly non-numeric and is not changed by STREAMLINE
* Match Label - is commonly non-numeric and is not changed by STREAMLINE
* Features - if text, will be numerically encoded based on alphabetical order and will automatically be treated as categorical features (overriding any user specification with `quantitative_feature_path` or `categorical_feature_path`)

**Outcome Labels:** STREAMLINE assumes that instances with class 0 are *negative* and those with class 1 are *positive* with respect to true positive, true negative, false positive, false negative metrics. This also impacts PRC plots, which focus on the classification of *positives*. Often, accurate prediction of the *positive* class is of greater interest than the *negative* class. Additionally, it is common in datasets for there to be a larger number of *negative* instances than *postitive* ones. 

### Pre-STREAMLINE Data Processing
STREAMLINE seeks to automate many of the common elements of a machine learning data science pipeline, but every dataset and analysis can have it's own unique needs and challenges. While not required by STREAMLINE to run, we recommmend users consider if any of the following steps apply and should be conducted prior to running as they can impact model interpretation and conclusions. 

**Text-valued Ordinal Features:** Some features values in data may be encoded as text entries, but have a natural quantitative ordering to them (i.e. ordinal features), but the distances between these values is not known. Since STREAMLINE will automatically treat any text-valued features as categorical, users that wish to have these features treated quanatitatively should first numerically encode these features in the data with values that seem most appropriate given the feature and relevant domain experience. 

For example, a feature could be the answer to a survey question with the values {strongly disagree, weakly disagree, neutral, weakly agree, or strongly agree}. These might intuitively be encoded by the user as {0,1,2,3,4}, however another user may have reason to encode these values as {0,3,5,7,10}.

**Class Label Encoding:** As indicated above, STREAMLINE will automatically numerically encode a text-based outcome value based on alphabetical order. However, this can break the assumption being made by STREAMLINE that class 0 is *negative* and class 1 is *positive* based on the given dataset and outcome values.

For example, if the outcome values were 'negative' and 'positive', STREAMLINE would automatically, and correctly numerically encode them as 0 and 1, respectively. However if the outcome values were 'well' and 'sick', these would be encoded as 1 and 0, respectively.

To avoid missinterpretation, we recommend users numerically encode class labels ahead of time based on their own needs for *negative* and *positive* lables. 

**Outliers:** Of note, outlier values within a feature or anomolous instances do not always reflect poor data quality, but may just reflect extreame true observations. As this aspect of data processing can not be reliably automated, we recommend users examine their data for possible outliers and remove them based on their own judgement prior to running STREAMLINE. This could involve replacing impossible feature value (e.g. age of 235 in a human) with a missing value, or removing entire instances from the dataset.

**Feature Extraction:** STREAMLINE does not currently tackle any feature extraction aspects of data science. If your raw data is in an unstructured or non-tabular data format (e.g. images, video, time-series data, natural language text), we recommend looking into relevant approaches to create structured features from these data sources prior to running STREAMLINE.

**Feature Engineering:** Traditionally, good feature engineering requires domain knowledge about the target problem or dataset. STREAMLINE automates a couple basic feature engineering elements, however we encourage users to consider other strategies to engineer features (in a manner that does not look at the outcome label). A simple example of this might be taking features representing start and end dates of a drug treatment and engineering a feature that indicates the time duration. 

**Feature Transformation:** While STREAMLINE uses a standard scalar to transform features in this pipeline, many other transformations are possible (for various reasons). Users that wish to apply these alternative transformations should do so before running STREAMLINE and then optionally turn of the standard scaling with `scale_data`.


## Demonstration Data
For demonstration and quick code-testing purposes, the STREAMLINE repository includes two small 'target datasets' that would be used in Phases 1-7, as well as a small 'replication dataset' that would be used in Phase 8. These datasets can be found in `./data/DemoData` and `./data/DemoRepData`, respectively.

New users can easily run STREAMLINE on these datasets in whatever run-mode desired. Instructions for running STREAMLINE are given using this demo data as an example for each run-mode (see [here](running.md)). Details on each of the demonstration datasets are given below. 

### Real-World HCC Dataset
The first demo dataset (`hcc-data_example.csv`) is an example of a real-world biomedical classification task. This is a [Hepatocellular Carcinoma (HCC)](https://archive.ics.uci.edu/dataset/423/hcc+survival) dataset taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). It includes 165 instances, 49 features, and a binary class label. It also includes a mix of categorical and quantitative features (however all categorical features are binary), about 10% missing values, and class imbalance, i.e. 63 deceased (class = 1), and 102 surived (class 0).

### Custom Extension of HCC Dataset
The second demo dataset (`hcc-data_example_custom.csv`) is similar to the first, but we have made a number of modifications to it in order to test the data cleaning and feature engineering functionalities of STREAMLINE. 

Modifcations include the following:
1. Removal of covariate features (i.e. 'Gender' and 'Age at diagnosis')
2. Addition of two simulated instances with a missing class label (to test basic data cleaning)
3. Addition of two simulated instances with a high proportion of missing values (to test instance missingness data cleaning)
4. Addition of three simulated, numerically encoded categorical features with 2, 3, or 4 unique values, respectively (to test one-hot-encoding)
5. Addition of three simulated, text-valued categorical features with 2, 3, or 4 unique values, respectively (to test one-hot-encoding of text-based features)
6. Addition of three simulated quantiative features with high missingness (to test feature missingness data cleaning)
7. Addition of three pairs of correlated quanatiative features (6 features added in total), with correlations of -1.0, 0.9, and 1.0, respectively (to test high correlation data cleaning)

These simulated features and instances have been clearly identified in the feature names and instances IDs of this dataset. 

### Simulated Replication Dataset
The last demo dataset (`hcc-data_example_custom_rep.csv`) was simulated as a mock replication dataset for `hcc-data_example_custom.csv`. To generate this dataset we first took `hcc-data_example_custom.csv` and for 30% of instances randomly generated realistic looking new values for each feature and class outcome (effectively adding noise to this data). Furthermore we simulated further instances that test the ability of STREAMLINE's one-hot-encoding to ignore new (as-of-yet unseen) categorical features values during STREAMLINE's replication phase. If this were to happen, the new value would be ignored (i.e. no new feature columns added).

Modifications included adding a simulated instance that includes a new (as-of-yet unseen) value for the following previously simulated features:
1. The binary text-valued categorical feature
2. The 3-value text-valued categorical feature
3. The binary numerically encoded categorical feature
4. The 3-value numerically encoded categorical feature


## Run Pipeline and Output Folders
* To quickly pre-view the pipeline (pre-run on included [demonstration datasets](#demonstration-data) without any installation whatsoever, open the following link:

[https://github.com/UrbsLab/STREAMLINE/blob/dev/STREAMLINE-Notebook.ipynb](https://github.com/UrbsLab/STREAMLINE/blob/dev/STREAMLINE-Notebook.ipynb)

Note, that with the above link, you can only view the pre-run STREAMLINE Jupyter Notebook and will not be able to run or permanently edit the code. This is an easy way to get a feel for what the pipeline is and does.

* To quickly pre-view the folder of output files generated when running STREAMLINE on the [demonstration datasets](#demonstration-data), open the following link:

[https://drive.google.com/drive/folders/1hMWOVJiOtQY_c767BylWTLObsKrANXRm?usp=sharing](https://drive.google.com/drive/folders/1hMWOVJiOtQY_c767BylWTLObsKrANXRm?usp=sharing)

## Figures Summary
Below is an example overview of the different figures generated by STREAMLINE for binary classification data.
![alttext](pictures/STREAMLINE_Figures.png)



