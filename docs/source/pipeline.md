#  Pipeline Design Details
This section is for users who want a more detailed understanding of (1) what STREAMLINE does, (2) what happens in durring each phase, (3) why it's designed the way it has been, (4) what user options are available to customize a run, and (5) what to expect when running a given phase. Phases 1-6 make up the core automated pipeline, with Phase 7 and beyond being run optionally based on user needs. Phases are organized to both encapsulate related pipeline elements, as well as to address practical computational needs. STREAMLINE includes reliable default run parameters so that it can easily be used 'as-is', but these parameters can be adjusted for further customization.

***
## Phase 1: Data Exploration & Processing
This phase (1) provides the user with key information about the dataset(s) they wish to analyze, via an initial exploratory data analysis (EDA) (2) numerically encodes any text-based feature values in the data, (3) applies basic data cleaning and feature engineering to process the data, (4) informs the user how the data has been changed by the data processing, via a secondary, more in-depth EDA, and then (5) partitions the data using k-fold cross validation. 

* Parallizability: Runs once for each target dataset to be analyzed
* Run Time: Typically fast, except when evaluating and visualizing feature correlation in datasets with a large number of features

### Initial EDA
Characterizes the orignal dataset as loaded by the user, including: data dimensions, feature type counts, missing value counts, class balance, other standard pandas data summaries (i.e. describe(), dtypes(), nunique()) and feature correlations (pearson). 

For precision, we strongly suggest users identify which features in their data should be treated as categorical vs. quanatiative using the `categorical_feature_path` and/or `quantitative_feature_path` run parameters. However, if not specified by the user, STREAMLINE will attempt to automatically determine feature types relying on the `categorical_cutoff` run parameter. Any features with fewer unique values than `categorical_cutoff` will be treated as categorical, and all others will be treated as quantitative. 

* Output: (1) CSV files for all above data characteristics, (2) bar plot of class balance, (3) histogram of missing values in data, (4) feature correlation heatmap

### Numerical Encoding of Text-based Features
Detects any features in the data with non-numeric values and applies [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) to make them numeric as required by scikit-learn machine learning packages.

* Output: None

### Basic Data Cleaning and Feature Engineering
Applies the following steps to the target data, keeping track of changes to all data counts along the way:
1. Remove any instances that are missing an outcome label (as these can not be used while conducting supervised learning)
2. Remove any features identified by the user in the `ignore_features_path` run parameter (a convenience for users that may wish to exclude one or more features from the analysis without changing the original dataset)
3. Engineer/add 'missingness' features. Any original feature with a missing value proportion greater than the `featureeng_missingness` run parameter will have a new feature added to the dataset that encodes missingness with 0 = not missing and 1 = missing. This allows the user to examine whether missingness is 'not at random', and is predictive of outcome itself.
4. Remove any features with a missingness greater than the `cleaning_missingness` run parameter. Afterwards, remove any instances in the data that may have a missingness greater than `cleaning_missingness`.
5. Engineer/add [one-hot-encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) for any categorical features in the data. This ensures that all categorical features are treated as such throughout all aspects of the pipeline. For example, a single categorical feature with 3 possible states will be encoded as 3 separate binary-valued features indicating whether an instance has that feature's state or not. Feature names are automatically updated by STREAMLINE to reflect this change.
6. Remove highly correlated features based on the `correlation_removal_threshold` run parameter. Randomly removes one feature of a highly correlated feature pair. While perfectly correlated features can be safely cleaned in this way, there is a chance of information loss when removing less correlated features.

* Output: CSV file summarizing changes to data counts during these cleaning and engineering steps.

### Processed Data EDA
Completes a more comprehensive EDA of the processed dataset including: everything examined in the initial EDA, as well as a univariate association analysis of all features using Chi-Square (for categorical features), or Mann-Whitney U-Test (for quantitative features). 

* Output: (1) CSV files for all above data characteristics, (2) bar plot of class balance, (3) histogram of missing values in data, (4) feature correlation heatmap, (5) a CSV file summarizing the univariate analyses including the test applied, test statistic, and p-value for each feature, (6) for any feature with a univariate analysis p-value less than the `sig_cutoff` run parameter (i.e. significant association with outcome), a bar-plot will be generated if it is categorical, and a box-plot will be generated if it is quanatiative.

### k-fold Cross Validation (CV) Partitioning
For k-fold CV, STREAMLINE uses 'Stratified' partitioning by default, which aims to maintain the same/similar class balance within the 'k' training and testing datasets. However using the `partition_method` run parameter, users can also select 'Random' or 'Group' partitioning. 

Of note, 'Group' partitioning requires the dataset to include a column identified by the `match_label` run parameter. This column includes a group membership identifier for each instance which indicates that any instance with the same group ID should be kept within the same partition during cross validation. This was originally intended for running STREAMLINE on epidemiological data that had been matched for one or more covariates (e.g. age, sex, race) in order to adjust for their effects during modeling.

* Output: CSV files for each training and testing dataset generated following partitioning. Note, by default STREAMLINE will overwrite these files as the working datasets undergo imputation, scaling and feature selection in subsequent phases. However, the user can keep copies of these intermediary CV datasets for review using the run parameter `overwrite_cv`.

## Phase 2: Imputation and Scaling
This phase conducts additional data preparation elements of the pipeline that occur after CV partitioning, i.e. missing value imputation and feature scaling. Both elements 
are 'trained' and applied separately to each individual training dataset. The respective testing datasets are not looked at when running imputation or feature scaling learning to avoid potential data leakage. However the learned imputation and scaling patterns are applied in the same way to the testing data as they were in the training data. Both imputation and scaling can optionally be turned off using the run parameters `impute_data` and `scale_data`, respectively for some specific use cases, however imputation must be on when missing data is present in order to run most scikit-learn modeling algorithms, and scaling should be on for certain modeling algorithms learn effectively (e.g. artificial neural networks), and for if the user wishes to infer feature importances directly from certain algorithm's internal estimators (e.g. logistic regression).

* Parallizability: Runs 'k' times for each target dataset being analyzed (where k is number of CV partitions)
* Run Time: Typically fast, with the exception of imputing larger datasets with many missing values

### Imputation
This phase first conducts imputation to replace any remaining missing values in the dataset with a 'value guess'. While missing value imputation could be resonably viewed as data manufacturing, it is a common practice and viewed here as a necessary 'evil' in order to run scikit-learn modeling algorithms downstream (which mostly require complete datasets). Imputation is completed prior to scaling so that 

Missing value imputation seeks to make a reasonable, educated guess as to the value of a given missing data entry. By default, STREAMLINE uses 'mode imputation' for all categorical values, and [multivariate imputation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) for all quantitative features. However, for larger datasets, multivariate imputation can be slow and require alot of memory. Therefore, the user can deactivate multiple imputation with the `multi_impute`
run parameter, and STREAMLINE will use median imputaion for quantitative features instead.

### Scaling
Second, this phase conducts feature scaling with [StandardScalar](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to transform features to have a mean at zero with unit variance. This is only necessary for certain modeling algorithms, but it should not hinder the performance of other algorithms. The primary drawback to scaling prior to modeling is that any future data applied to the model will need to be scaled in the same way prior to making predictions. Furthermore, for algorithms that have directly interpretable models (e.g. decision tree), the values specified by these models need to be un-scaled in order to understand the model in the context of the original data values. STREAMLINE includes a [Useful Notebook](more.md) that can generate direct model visualizations for decision tree and genetic programming models. This code automatically un-scales the values specified in these models so they retain their interpretability.

* Output: (1) Learned imputation and scaling strategies for each training dataset are saved as pickled objects allowing any replication or other future data to be identically processed prior to running it through the model. (2) If `overwrite_cv` is False, new imputed and scaled copies of the training and testing datasets are saved as CSV output files, otherwise the old dataset files are overwritten with these new ones to save space.

## Phase 3: Feature Importance Estimation
This phase applies feature importance estimation algorithms (i.e. [Mutual information (MI)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) and [MultiSURF](https://github.com/UrbsLab/scikit-rebate), found in the ReBATE software package) often used as filter-based feature selection algorithms. Both algorithms are run by default, however the user can deactivate either using the `do_mutual_info` and `do_multisurf` run parameters, respectively. MI scores features based on their univariate association with outcome, while MultiSURF scores features in a manner that is sensitive to both univariate and epistatic (i.e. multivariate feature interaction) associations.

For datasets with a larger number of features (i.e. > 10,000) we recommend turning on the TuRF wrapper algorithm with the `use_turf` run parameter, which has been shown to improve the sensitivity of MultiSURF to interactions, particularly in larger feature spaces. Users can increase the number of TuRF iterations (and performance) by decreasing the `turf_pct` run parameter from 0.5, to approaching 0. However, this will dramatically increase run time.

Overall, this phase is important not only for subsequent feature selection, but as an opportunity to evaluate feature importance estimates prior to modeling outside of the initial univariate analyses (conducted on the entire dataset). Further, comparing feature rankings between MI, and MultiSURF can highlight features that may have little or no univariate effects, but that are involved in epistatic interactions that are predictive of outcome.

* Parallizability: Runs 'k' times for each algorithm (MI and MultiSURF) and each target dataset being analyzed (where k is number of CV partitions)
* Run Time: Typically reasonably fast, but takes more time to run MultiSURF, in particular as the number of training instances approaches the default `instance_subset` run parameter of 2000 instances, or if this parameter set higher in larger datasets. This is because MultiSURF scales quadratically with the number of training instances.

* Output: CSV files of feature importance scores for both algorithms and each CV partition ranked from largest to smallest scores.

## Phase 4: Feature Selection
This phase uses the feature importance estimates learned in the prior phase to conduct feature selection using a 'collective' feature selection approach. By default, STREAMLINE will remove any features from the training data that scored 0 or less by both feature importance algorithms (i.e. features deamed uninformative). Users can optionally ensure retention of all features prior to modeling by setting the `filter_poor_features` run parameter to False. Users can also specify a maximum number of features to retain in each training dataset using the `max_features_to_keep` run parameter (which can help reduce overall pipeline runtime and make learning easier for modeling algorithms). If after removing 'uninformative features' there are still more features present than the user specified maximum, STREAMLINE will pick the unique top scoring features from one algorithm then the next until the maximum is reached and all other features are removed. Any features identified for removal from the training data are similarly removed from the testing data.

* Parallizability: Runs 'k' times for each target dataset being analyzed (where k is number of CV partitions)
* Run Time: Fast

* Output: (1) CSV files summarizing feature selection for a target dataset (i.e. how many features were identified as informative or uninformative within each CV partition) and (2) a barplot of average feature importance scores (across CV partitions). The user can specify the maximum number of top scoring features to be plotted using the `top_features` run parameter.

## Phase 5: Machine Learning (ML) Modeling
At the heart of STREAMLINE, this phase conducts machine learning modeling using the training data, model feature importance estimation (also with the training data), and model evaluation on testing data. 

* Parallizability: Runs 'k' times for each algorithm and each target dataset being analyzed (where k is number of CV partitions)
* Run Time: Slowest phase, but can be sped up by reducing the set of ML methods selected to run, or deactivating ML methods that run slowly on large datasets

### Model Selection
The first step is to decide which modeling algorithms to run. By default, STREAMLINE applies 14 of the 16 algorithms (excluding eLCS and XCS) it currently has built in. Users can specify a specific subset of algorithms to run using the `algorithms` run parameter, or alternatively indicate a list of algorithms to exclude from all available algorithms using the `exclude` run parameter. STREAMLINE is also set up so that more advanced users can add other scikit-learn compatible modeling algorithms to run within the pipeline (as explained in [Adding New Modeling Algorithms](models.md)). This allows STREAMLINE to be used as a rigourous framework to easily benchmark new modeling algorithms in comparison to other established algorithms.

Modeling algorithms vary in the implicit or explict assumptions they make, the manner in which they learn, how they represent a solution (as a model), how well they handle different patterns of association in the data, how long they take to run, and how complex and/or interpretable the resulting model can be. To reduce runtime in datasets with a large number of training instances, users can specify a limited random number of training instances to use in training algorithms that run slowly in such datasets using the `training_subsample` run parameter.

In the STREAMLINE demo, we run only the fastest/simplest three algorithms (Naive Bayes, Logistic Regression, and Decision Tree), however these algorithms all have known limitations in their ability to detect complex associations in data. We encourage users to utilize the full variety of 14 algorithms in their analyses to give STREAMLINE the best opportunity to identify a best performing model for the given task (which is effectively impossible to predict for a given dataset ahead of time). We recommend users utilize at least the following set of algorithms within STREAMLINE: Naive Bayes, Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, ANN, and ExSTraCS as we have found these to be a reliable set of algorithms with an array of complementary strengths and weaknesses on different problems. 

### Hyperparameter Optimization
Most machine learning algorithms have a variety of hyperparameter options that influence how the algorithm runs and performs on a given dataset. In designing STREAMLINE we sought to identify the full range of important hyperparameters for each algorithm, along with a broad range of possible settings and hard-coded these into the pipeline.  STREAMLINE adopts the [Optuna](https://optuna.org/) package to conduct automated Bayesian optimization of hyperparameters for most algorithms by default. The evaluation metric 'balanced accuracy' is used to optimize hyperparameters as it takes class imbalance into account and places equal weight on the accurate prediction of both class outcomes. However, users can select an alternative metric with the `primary_metric` run parameter and whether that metric needs to be maximized or minimized using the `metric_direction` run parameter. To conduct the hyperparameter sweep, Optuna splits a given training dataset further, applying 3-fold cross validation to generate further internal training and validation partitions with which to evaluate different hyperparameter combinations.

Users can also configure how Optuna operates in STREAMLINE with the `n_trials` and `timeout` run parameters which controls the target number of hyperparameter value combination trials to conduct, as well as how much total time to try and complete these trials before picking the best hypercombination found. To ensure reproducibility of the pipeline, note that `timeout` should be set to 'None' (however this can take much longer to complete depending on other pipeline settings).

Notable exceptions to most algorithms; Naive Bayes has no run parameters to optimize, and rule-based (i.e. LCS) ML algorithms including ExSTraCS, eLCS, and XCS can be computationally expensive thus STREAMLINE is set to use their default hyperparameter settings without a sweep unless the user updates the `do_lcs_sweep` and `lcs_timeout` run parameters.  While these LCS algorithms have many possible hyperparameters to manipulate they have largely stable performance when leaving most of their run parameters to default values. Exeptions to this include the key LCS hyperparameters (1) number of learning iterations, (2) maximum rule-population size, and (3) accuracy pressure (nu), which can be manually controled without a hyperparmater sweep by the run parameters `lcs_iterations`, `lcs_N`, and `lcs_nu`, respectively.

* Output: CSV files specifying the optimized hyperparameter settings found by Optuna for each partition and algorithm combination.

### Train 'Optimized' Model
Having selected the best hyperparameter combination identified for a given training dataset and algorithm, STREAMLINE now retrains each model on the entire training dataset using those respective hyperparameter settings. This yields a total of 'k' potentially 'optimized' models for each algorithm. 

* Output: All trained models are pickled as python objects that can be loaded and applied later.

### Model Feature Importance Estimation
Next, STREAMINE estimates and summarizes model feature importance scores for every algorithm run. This is distinct from the initial feature importance estimation phase, in that these estimates are specific to a given model as a useful part of model interpretation/explanation. By default, STREAMLINE employes [permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) for estimating feature importances scores in the same uniform manner across all algorithms. However, the user can deactive this by setting the `use_uniform_fi` run parameter to 'False'. This will direct STREAMLINE to report any available internal feature importance estimate for a given algorithm, while still utilizing permutation feature importance for algorithms with no such internal estimator. 

* Output: See below.

### Evaluate Performance
The last step in this phase is to evaluate all trained models using their respective testing datasets.  A total of 16 standard classification metrics calculated for each model including: balanced accuracy, standard accuracy, F1 Score, sensitivity (recall), specificity, precision (positive predictive value), true positives, true negatives, false postitives, false negatives, negative predictive value, likeliehood ratio positive, likeliehood ratio negative, area under the ROC, area under the PRC, and average precision of PRC. 

* Output: All feature importance scores and evaluation metrics are pickled as python objects for use in the next phase of the pipeline.

## Phase 6: Post-Analysis
This phase combines all modeling results to generate summary statistics files, generate results plots, and conduct non-parametric statistical significance analysis comparing ML performance across CV runs.



* Parallizability: Runs once for each target dataset being analyzed.
* Run Time: Moderately fast




### STREAMLINE Phases Described
The base code for STREAMLINE (located in the `streamline` folder) is organized into a series of script phases designed to best optimize the parallelization of a given analysis. 


* Phase 6: Statistics Summary
  * Combines all results to generate summary statistics files, generate results plots, and conduct non-parametric statistical significance analyses comparing ML model performance across CV runs
  * \[Runtime]: Moderately fast

* Phase 7: [Optional] Compare Datasets
  * NOTE: Only can be run if the STREAMLINE was run on more than dataset
  * Conducts non-parametric statistical significance analyses comparing separate original 'target' datasets analyzed by pipeline
  * \[Runtime]: Fast

* Phase 8: [Optional] Generate PDF Training Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, statistical comparisons, and runtime.
  * \[Runtime]: Moderately fast

* Phase 9: [Optional] Apply Models to Replication Data
  * Applies all previously trained models for a single 'target' dataset to one or more new 'replication' dataset(s) that has the same features from an original 'target' dataset
  * Conducts exploratory analysis on new 'replication' dataset(s)
  * Applies scaling, imputation, and feature selection (unique to each CV partition from model training) to new 'replication' dataset(s) in preparation for model application
  * Evaluates performance of all models the prepared 'replication' dataset(s)
  * Generates summary statistics files, results plots, and conducts non-parametric statistical significance analyses comparing ML model performance across replications CV data transformations
  * NOTE: feature importance evaluation and 'target' dataset statistical comparisons are irrelevant to this phase
  * \[Runtime]: Moderately fast

* Phase 10: [Optional] Generate PDF 'Apply Replication' Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, and statistics.
  * \[Runtime]: Moderately fast

* Phase 11: [Optional] File Cleanup
  * Deletes files that do not need to be kept following pipeline run.
  * \[Runtime]: Fast



* Modeling
 * Includes 3 rule-based machine learning algorithms: ExSTraCS, XCS, and eLCS (to run optionally). These 'learning classifier systems' have been demonstrated to be able to detect complex associations while providing human interpretable models in the form of IF:THEN rule-sets. The ExSTraCS algorithm was developed by our research group to specifically handle the challenges of scalability, noise, and detection of epistasis and genetic heterogeneity in biomedical data mining.  
 * Utilizes the 'optuna' package to conduct automated Bayesian hyperparameter optimization during modeling (and optionally outputs plots summarizing the sweep).
 * We have sought to specify a comprehensive range of relevant hyperparameter options for all included ML algorithms.
 * Some ML algorithms that have a build in strategy to gather model feature importance estimates (i.e. LR,DT,RF,XGB,LGB,GB,eLCS,XCS,ExSTraCS) These can be used in place of permutation feature importance estimates by setting the parameter `use_uniform_fi` to 'False'.
 * All other algorithms rely on estimating feature importance using permutation feature importance.
 * All models are evaluated, reporting 16 classification metrics: Accuracy, Balanced Accuracy, F1 Score, Sensitivity(Recall), Specificity, Precision (PPV), True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN), Negative Predictive Value (NPV), Likeliehood Ratio + (LR+), Likeliehood Ratio - (LR-), ROC AUC, PRC AUC, and PRC APS.
 * All models are saved as 'pickle' files so that they can be loaded and reapplied in the future.

* Post-Analysis
 * Outputs ROC and PRC plots for each ML modeling algorithm displaying individual n-fold CV runs and average the average curve.
 * Outputs boxplots for each classification metric comparing ML modeling performance (across n-fold CV).
 * Outputs boxplots of feature importance estimation for each ML modeling algorithm (across n-fold CV).
 * Outputs our proposed 'composite feature importance plots' to examine feature importance estimate consistency (or lack of consistency) across all ML models (i.e. all algorithms)
 * Outputs summary ROC and PRC plots comparing average curves across all ML algorithms.
 * Collects run-time information on each phase of the pipeline and for the training of each ML algorithm model.
 * For each dataset, Kruskall-Wallis and subsequent pairwise Mann-Whitney U-Tests evaluates statistical significance of ML algorithm modeling performance differences for all metrics.
 * The same statistical tests (Kruskall-Wallis and Mann-Whitney U-Test) are conducted comparing datasets using the best performing modeling algorithm (for a given metric and dataset).
 * Outputs boxplots comparing performance of multiple datasets analyzed either across CV runs of a single algorithm and metric, or across average for each algorithm for a single metric.
 * A formatted PDF report is automatically generated giving a snapshot of all key pipeline results.
 * A script is included to apply all trained (and 'pickled') models to an external replication dataset to further evaluate model generalizability. This script (1) conducts an exploratory analysis of the new dataset, (2) uses the same scaling, imputation, and feature subsets determined from n-fold cv training, yielding 'n' versions of the replication dataset to be applied to the respective models, (3) applies and evaluates all models with these respective versions of the replication data, (4) outputs the same set of aforementioned boxplots, ROC, and PRC plots, and (5) automatically generates a new, formatted PDF report summarizing these applied results.
