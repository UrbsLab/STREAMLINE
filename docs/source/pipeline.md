#  Detailed Pipeline Walkthrough
This section is for users who want a more detailed understanding of (1) what STREAMLINE does, (2) what happens in durring each phase, (3) why it's designed the way it has been, (4) what user options are available to customize a run, and (5) what to expect when running a given phase. Phases 1-6 make up the core automated pipeline, with Phase 7 and beyond being run optionally based on user needs. Phases are organized to both encapsulate related pipeline elements, as well as to address practical computational needs. STREAMLINE includes reliable default run parameters so that it can easily be used 'as-is', but these parameters can be adjusted for further customization. We refer to a single run of the entire STREAMLINE pipeline as an 'experiment', with all outputs saved to a single 'experiment folder' for later examination and re-use.

To avoid confusion on 'dataset' terminology we briefly review our definitions here:
1. **Target dataset** - A whole dataset (minus any instances the user may wish to hold out for replication) that has not yet undergone any other data partitioning and is intended to be used in the training and testing of models within STREAMLINE. Could also be referred to as the 'development dataset'.
2. **Training dataset** - A generally larger partition of the target dataset used in training a model
3. **Testing dataset** - A generally smaller partition of the target dataset used to evaluate the trained model
4. **Validation dataset** - The temporary, secondary hold-out partition of a given training dataset used for hyperparameter optimization. This is the product of using nested (aka double) cross-validation in STREAMLINE as a whole.
5. **Replication dataset** - Further data that is withheld from STREAMLINE phases 1-7 to (1) compare model evaluations on the same hold-out data and (2) verify the replicatability and generalizability of model performance on data collected from other sites or sample populations. A replication dataset should have at least all of the features present in the target dataset which it seeks to replicate. 

***
## Phase 1: Data Exploration & Processing
This phase (1) provides the user with key information about the target dataset(s) they wish to analyze, via an initial exploratory data analysis (EDA) (2) numerically encodes any text-based feature values in the data, (3) applies basic data cleaning and feature engineering to process the data, (4) informs the user how the data has been changed by the data processing, via a secondary, more in-depth EDA, and then (5) partitions the data using k-fold cross validation. 

* **Parallizability:** Runs once for each target dataset to be analyzed
* **Run Time:** Typically fast, except when evaluating and visualizing feature correlation in datasets with a large number of features

### Initial EDA
Characterizes the orignal dataset as loaded by the user, including: data dimensions, feature type counts, missing value counts, class balance, other standard pandas data summaries (i.e. describe(), dtypes(), nunique()) and feature correlations (pearson). 

For precision, we strongly suggest users identify which features in their data should be treated as categorical vs. quanatiative using the `categorical_feature_path` and/or `quantitative_feature_path` run parameters. However, if not specified by the user, STREAMLINE will attempt to automatically determine feature types relying on the `categorical_cutoff` parameter. Any features with fewer unique values than `categorical_cutoff` will be treated as categorical, and all others will be treated as quantitative. 

* **Output:** (1) CSV files for all above data characteristics, (2) bar plot of class balance, (3) histogram of missing values in data, (4) feature correlation heatmap

### Numerical Encoding of Text-based Features
Detects any features in the data with non-numeric values and applies [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) to make them numeric as required by scikit-learn machine learning packages.

### Basic Data Cleaning and Feature Engineering
Applies the following steps to the target data, keeping track of changes to all data counts along the way:
1. Remove any instances that are missing an outcome label (as these can not be used while conducting supervised learning)
2. Remove any features identified by the user with `ignore_features_path` (a convenience for users that may wish to exclude one or more features from the analysis without changing the original dataset)
3. Engineer/add 'missingness' features. Any original feature with a missing value proportion greater than `featureeng_missingness` will have a new feature added to the dataset that encodes missingness with 0 = not missing and 1 = missing. This allows the user to examine whether missingness is 'not at random', and is predictive of outcome itself.
4. Remove any features with a missingness greater than `cleaning_missingness`. Afterwards, remove any instances in the data that may have a missingness greater than `cleaning_missingness`.
5. Engineer/add [one-hot-encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) for any categorical features in the data. This ensures that all categorical features are treated as such throughout all aspects of the pipeline. For example, a single categorical feature with 3 possible states will be encoded as 3 separate binary-valued features indicating whether an instance has that feature's state or not. Feature names are automatically updated by STREAMLINE to reflect this change.
6. Remove highly correlated features based on `correlation_removal_threshold`. Randomly removes one feature of a highly correlated feature pair (Pearson). While perfectly correlated features can be safely cleaned in this way, there is a chance of information loss when removing less correlated features.

* **Output:** CSV file summarizing changes to data counts during these cleaning and engineering steps.

### Processed Data EDA
Completes a more comprehensive EDA of the processed dataset including: everything examined in the initial EDA, as well as a univariate association analysis of all features using Chi-Square (for categorical features), or Mann-Whitney U-Test (for quantitative features). 

* **Output:** (1) CSV files for all above data characteristics, (2) bar plot of class balance, (3) histogram of missing values in data, (4) feature correlation heatmap, (5) a CSV file summarizing the univariate analyses including the test applied, test statistic, and p-value for each feature, (6) for any feature with a univariate analysis p-value less than `sig_cutoff` (i.e. significant association with outcome), a bar-plot will be generated if it is categorical, and a box-plot will be generated if it is quanatiative.

### k-fold Cross Validation (CV) Partitioning
For k-fold CV, STREAMLINE uses 'Stratified' partitioning by default, which aims to maintain the same/similar class balance within the 'k' training and testing datasets. The value of 'k' can be adjusted with `n_splits`. However using `partition_method`, users can also select 'Random' or 'Group' partitioning. 

Of note, 'Group' partitioning requires the dataset to include a column identified by `match_label`. This column includes a group membership identifier for each instance which indicates that any instance with the same group ID should be kept within the same partition during cross validation. This was originally intended for running STREAMLINE on epidemiological data that had been matched for one or more covariates (e.g. age, sex, race) in order to adjust for their effects during modeling.

* **Output:** CSV files for each training and testing dataset generated following partitioning. Note, by default STREAMLINE will overwrite these files as the working datasets undergo imputation, scaling and feature selection in subsequent phases. However, the user can keep copies of these intermediary CV datasets for review using the parameter `overwrite_cv`.

***
## Phase 2: Imputation and Scaling
This phase conducts additional data preparation elements of the pipeline that occur after CV partitioning, i.e. missing value imputation and feature scaling. Both elements 
are 'trained' and applied separately to each individual training dataset. The respective testing datasets are not looked at when running imputation or feature scaling learning to avoid potential data leakage. However the learned imputation and scaling patterns are applied in the same way to the testing data as they were in the training data. Both imputation and scaling can optionally be turned off using the parameters `impute_data` and `scale_data`, respectively for some specific use cases, however imputation must be on when missing data is present in order to run most scikit-learn modeling algorithms, and scaling should be on for certain modeling algorithms learn effectively (e.g. artificial neural networks), and for if the user wishes to infer feature importances directly from certain algorithm's internal estimators (e.g. logistic regression).

* **Parallizability:** Runs 'k' times for each target dataset being analyzed (where k is number of CV partitions)
* **Run Time:** Typically fast, with the exception of imputing larger datasets with many missing values

### Imputation
This phase first conducts imputation to replace any remaining missing values in the dataset with a 'value guess'. While missing value imputation could be resonably viewed as data manufacturing, it is a common practice and viewed here as a necessary 'evil' in order to run scikit-learn modeling algorithms downstream (which mostly require complete datasets). Imputation is completed prior to scaling so that 

Missing value imputation seeks to make a reasonable, educated guess as to the value of a given missing data entry. By default, STREAMLINE uses 'mode imputation' for all categorical values, and [multivariate imputation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) for all quantitative features. However, for larger datasets, multivariate imputation can be slow and require alot of memory. Therefore, the user can deactivate multiple imputation with the `multi_impute`
parameter, and STREAMLINE will use median imputaion for quantitative features instead.

### Scaling
Second, this phase conducts feature scaling with [StandardScalar](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to transform features to have a mean at zero with unit variance. This is only necessary for certain modeling algorithms, but it should not hinder the performance of other algorithms. The primary drawback to scaling prior to modeling is that any future data applied to the model will need to be scaled in the same way prior to making predictions. Furthermore, for algorithms that have directly interpretable models (e.g. decision tree), the values specified by these models need to be un-scaled in order to understand the model in the context of the original data values. STREAMLINE includes a [Useful Notebook](more.md) that can generate direct model visualizations for decision tree and genetic programming models. This code automatically un-scales the values specified in these models so they retain their interpretability.

* **Output:** (1) Learned imputation and scaling strategies for each training dataset are saved as pickled objects allowing any replication or other future data to be identically processed prior to running it through the model. (2) If `overwrite_cv` is False, new imputed and scaled copies of the training and testing datasets are saved as CSV output files, otherwise the old dataset files are overwritten with these new ones to save space.

***
## Phase 3: Feature Importance Estimation
This phase applies feature importance estimation algorithms (i.e. [Mutual information (MI)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) and [MultiSURF](https://github.com/UrbsLab/scikit-rebate), found in the ReBATE software package) often used as filter-based feature selection algorithms. Both algorithms are run by default, however the user can deactivate either using `do_mutual_info` or `do_multisurf`, respectively. MI scores features based on their univariate association with outcome, while MultiSURF scores features in a manner that is sensitive to both univariate and epistatic (i.e. multivariate feature interaction) associations.

For datasets with a larger number of features (i.e. > 10,000) we recommend turning on the TuRF wrapper algorithm with `use_turf`, which has been shown to improve the sensitivity of MultiSURF to interactions, particularly in larger feature spaces. Users can increase the number of TuRF iterations (and performance) by decreasing `turf_pct` from 0.5, to approaching 0. However, this will dramatically increase run time.

Overall, this phase is important not only for subsequent feature selection, but as an opportunity to evaluate feature importance estimates prior to modeling outside of the initial univariate analyses (conducted on the entire dataset). Further, comparing feature rankings between MI, and MultiSURF can highlight features that may have little or no univariate effects, but that are involved in epistatic interactions that are predictive of outcome.

* **Parallizability:** Runs 'k' times for each algorithm (MI and MultiSURF) and each target dataset being analyzed (where k is number of CV partitions)
* **Run Time:** Typically reasonably fast, but takes more time to run MultiSURF, in particular as the number of training instances approaches the default `instance_subset` parameter of 2000 instances, or if this parameter set higher in larger datasets. This is because MultiSURF scales quadratically with the number of training instances.

* **Output:** CSV files of feature importance scores for both algorithms and each CV partition ranked from largest to smallest scores.

***
## Phase 4: Feature Selection
This phase uses the feature importance estimates learned in the prior phase to conduct feature selection using a 'collective' feature selection approach. By default, STREAMLINE will remove any features from the training data that scored 0 or less by both feature importance algorithms (i.e. features deamed uninformative). Users can optionally ensure retention of all features prior to modeling by setting `filter_poor_features` to False. Users can also specify a maximum number of features to retain in each training dataset using `max_features_to_keep` (which can help reduce overall pipeline runtime and make learning easier for modeling algorithms). If after removing 'uninformative features' there are still more features present than the user specified maximum, STREAMLINE will pick the unique top scoring features from one algorithm then the next until the maximum is reached and all other features are removed. Any features identified for removal from the training data are similarly removed from the testing data.

* **Parallizability:** Runs 'k' times for each target dataset being analyzed (where k is number of CV partitions)
* **Run Time:** Fast

* **Output:** (1) CSV files summarizing feature selection for a target dataset (i.e. how many features were identified as informative or uninformative within each CV partition) and (2) a barplot of average feature importance scores (across CV partitions). The user can specify the maximum number of top scoring features to be plotted using `top_features`.

***
## Phase 5: Machine Learning (ML) Modeling
At the heart of STREAMLINE, this phase conducts (1) machine learning modeling using the training data, (2) model feature importance estimation (also with the training data), and (3) model evaluation on testing data. STREAMLINE uniquely includes 3 rule-based machine learning algorithms: ExSTraCS, XCS, and eLCS. These 'learning classifier systems' have been demonstrated to be able to detect complex associations while providing human interpretable models in the form of IF:THEN rule-sets. The ExSTraCS algorithm was developed by our research group to specifically handle the challenges of scalability, noise, and detection of epistasis and genetic heterogeneity in biomedical data mining. 

* **Parallizability:** Runs 'k' times for each algorithm and each target dataset being analyzed (where k is number of CV partitions)
* **Run Time:** Slowest phase, but can be sped up by reducing the set of ML methods selected to run, or deactivating ML methods that run slowly on large datasets

### Model Selection
The first step is to decide which modeling algorithms to run. By default, STREAMLINE applies 14 of the 16 algorithms (excluding eLCS and XCS) it currently has built in. Users can specify a specific subset of algorithms to run using `algorithms`, or alternatively indicate a list of algorithms to exclude from all available algorithms using `exclude`. STREAMLINE is also set up so that more advanced users can add other scikit-learn compatible modeling algorithms to run within the pipeline (as explained in [Adding New Modeling Algorithms](models.md)). This allows STREAMLINE to be used as a rigourous framework to easily benchmark new modeling algorithms in comparison to other established algorithms.

Modeling algorithms vary in the implicit or explict assumptions they make, the manner in which they learn, how they represent a solution (as a model), how well they handle different patterns of association in the data, how long they take to run, and how complex and/or interpretable the resulting model can be. To reduce runtime in datasets with a large number of training instances, users can specify a limited random number of training instances to use in training algorithms that run slowly in such datasets using `training_subsample`.

In the STREAMLINE demo, we run only the fastest/simplest three algorithms (Naive Bayes, Logistic Regression, and Decision Tree), however these algorithms all have known limitations in their ability to detect complex associations in data. We encourage users to utilize the full variety of 14 algorithms in their analyses to give STREAMLINE the best opportunity to identify a best performing model for the given task (which is effectively impossible to predict for a given dataset ahead of time). We recommend users utilize at least the following set of algorithms within STREAMLINE: Naive Bayes, Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, ANN, and ExSTraCS as we have found these to be a reliable set of algorithms with an array of complementary strengths and weaknesses on different problems. 

### Hyperparameter Optimization
Most machine learning algorithms have a variety of hyperparameter options that influence how the algorithm runs and performs on a given dataset. In designing STREAMLINE we sought to identify the full set of important hyperparameters for each algorithm, along with a comprehensive range of possible settings and hard-coded these into the pipeline.  STREAMLINE adopts the [Optuna](https://optuna.org/) package to conduct automated Bayesian optimization of hyperparameters for most algorithms by default. The evaluation metric 'balanced accuracy' is used to optimize hyperparameters as it takes class imbalance into account and places equal weight on the accurate prediction of both class outcomes. However, users can select an alternative metric with `primary_metric` and whether that metric needs to be maximized or minimized using the `metric_direction` parameter. To conduct the hyperparameter sweep, Optuna splits a given training dataset further, applying 3-fold cross validation to generate further internal training and validation partitions with which to evaluate different hyperparameter combinations.

Users can also configure how Optuna operates in STREAMLINE with `n_trials` and `timeout` which controls the target number of hyperparameter value combination trials to conduct, as well as how much total time to try and complete these trials before picking the best hypercombination found. To ensure reproducibility of the pipeline, note that `timeout` should be set to 'None' (however this can take much longer to complete depending on other pipeline settings).

Notable exceptions to most algorithms; Naive Bayes has no parameters to optimize, and rule-based (i.e. LCS) ML algorithms including ExSTraCS, eLCS, and XCS can be computationally expensive thus STREAMLINE is set to use their default hyperparameter settings without a sweep unless the user updates `do_lcs_sweep` and `lcs_timeout`.  While these LCS algorithms have many possible hyperparameters to manipulate they have largely stable performance when leaving most of their parameters to default values. Exeptions to this include the key LCS hyperparameters (1) number of learning iterations, (2) maximum rule-population size, and (3) accuracy pressure (nu), which can be manually controled without a hyperparmater sweep by `lcs_iterations`, `lcs_N`, and `lcs_nu`, respectively.

* **Output:** CSV files specifying the optimized hyperparameter settings found by Optuna for each partition and algorithm combination.

### Train 'Optimized' Model
Having selected the best hyperparameter combination identified for a given training dataset and algorithm, STREAMLINE now retrains each model on the entire training dataset using those respective hyperparameter settings. This yields a total of 'k' potentially 'optimized' models for each algorithm. 

* **Output:** All trained models are pickled as python objects that can be loaded and applied later.

### Model Feature Importance Estimation
Next, STREAMINE estimates and summarizes model feature importance scores for every algorithm run. This is distinct from the initial feature importance estimation phase, in that these estimates are specific to a given model as a useful part of model interpretation/explanation. By default, STREAMLINE employes [permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) for estimating feature importances scores in the same uniform manner across all algorithms. Some ML algorithms that have a build in strategy to gather model feature importance estimates (i.e. LR,DT,RF,XGB,LGB,GB,eLCS,XCS,ExSTraCS). The user can instead use these estimates by setting `use_uniform_fi` to 'False'. This will direct STREAMLINE to report any available internal feature importance estimate for a given algorithm, while still utilizing permutation feature importance for algorithms with no such internal estimator. 

* **Output:** All feature importance scores are pickled as python objects for use in the next phase of the pipeline.

### Evaluate Performance
The last step in this phase is to evaluate all trained models using their respective testing datasets.  A total of 16 standard classification metrics calculated for each model including: balanced accuracy, standard accuracy, F1 Score, sensitivity (recall), specificity, precision (positive predictive value), true positives, true negatives, false postitives, false negatives, negative predictive value, likeliehood ratio positive, likeliehood ratio negative, area under the ROC, area under the PRC, and average precision of PRC. 

* **Output:** All evaluation metrics are pickled as python objects for use in the next phase of the pipeline.

***
## Phase 6: Post-Analysis
This phase combines all modeling results to generate summary statistics files, generate results plots, and conduct non-parametric statistical significance analysis comparing ML performance across CV runs.

* **Output (Evaluation Metrics):**
    1. Testing data evaluation metrics for each CV partition - CSV file for each modeling algorithm 
    2. Average testing data evaluation metrics for each modeling algorithm - CSV file for mean, median, and standard deviation - 
    3. ROC and PRC curves for each CV partition in contrast with the average curve - ROC and PRC plot for each modeling algorithm
        * Plot generation be turned off with `plot_ROC` and `plot_PRC`, respectively
    4. Summary ROC and PRC plots - compares average ROC or PRC curves over all CV partitions for each modeling algorithm
    5. Boxplots for each evaluation metric - comparing algorithm performance over all CV partitions
        * Plot generation can be turned off with `plot_metric_boxplots`

* **Output (Model Feature Importance):**
    1. Model feature importance estimates for each CV partition - CSV file for each modeling algorithm
    2. Boxplots comparing feature importance scores for each CV partition - CSV file for each modeling algorithm
    3. Composite feature importance barplots illustrating average feature importance scores across all algorithms (2 versions)
        1. Feature importance scores are normalized prior to visualization
        2. Feature importance scores are normalized and weighted by average model performance metric (balanced accuracy by default)
            * The metric used to weight this plot can be changed with `metric_weight`
        * Number of top scoring features illustrated in feature importance plots is controled by `top_model_features`

* **Output (Statistical Significance):**
    1. Kruskal Wallis test results assessing whether there is a significant difference for each performance metric among all algorithms - CSV file
        * For any metric that yields a significant difference based on `sig_cutoff`, pairwise statistical tests between algorithms will be conducted using both the Mann Whitney U-test and the Wilcoxon Rank Test 
    2. Pairwise statistical tests between algorithms using both the Mann Whitney U-test and the Wilcoxon Rank Test - CSV file for each statistic and significance test.

* **Parallizability:** Runs once for each target dataset being analyzed.
* **Run Time:** Moderately fast - turning off some figure generation can make this phase faster

***
## Phase 7: Compare Datasets
This phase should be run when STREAMLINE was applied to more than one target dataset during the 'experiment'. It applies further non-parametric statistical significance testing between datasets to identify if performance differences were observed among datasets comparing (1) the best performing algorithms or (2) on an algorithm by algorithm basis. It also generates plots to compare performance across datasets and examine algorithm performance consistency. 

* **Parallizability:** Runs once - not parallizable
* **Run Time:** Fast

* **Output (Comparing Best Performing Algorithms for each Metric):**
    1. Kruskal Wallist test results assessing whether there is a significant difference between median CV performance metric among all datasets focused on the best performing algorithm for each dataset - CSV file
        * For any metric that yields a significant difference based on `sig_cutoff`, pairwise statistical tests between datasets will be conducted using both the Mann Whitney U-test and the Wilcoxon Rank Test 
    2. Pairwise statistical tests between datasets focused on the best performing algorithm for each dataset using both the Mann Whitney U-test and the Wilcoxon Rank Test - CSV file for each significance test.

* **Output (Comparing Algorithms Independently):**
    1. Kruskal Wallist test results assessing whether there is a significant difference for each median CV performance metric among all datasets - CSV file for each algorithm
        * For any metric that yields a significant difference based on `sig_cutoff`, pairwise statistical tests between datasets will be conducted using both the Mann Whitney U-test and the Wilcoxon Rank Test 
    2. Pairwise statistical tests between datasets using both the Mann Whitney U-test and the Wilcoxon Rank Test - CSV file for each significance test and algorithm

***
## Phase 8: Replication
This phase of STREAMLINE is only run when the user has further hold out data , i.e. one or more 'replication' datasets, which will be used to re-evaluate all models trained on a given target dataset. This means that this phase would need to be run once to evalute the models of each original target dataset Notably, this phase would be the first time that all models are evaluated on the same set of data which is useful for more confidently picking a 'best' model and further evaluating model generalizability and it's ability to replicate performance on data collected at different times, sites, or populations.
To run this phase the user needs to specify the filepath to the target dataset to be replicated with `dataset_for_rep` as well as the folderpath to the folder containing one or more replication datasets using `rep_data_path`.

This phase begins by conducting an initial exploratory data analyis (EDA) on the new replication dataset(s), followed by processing the dataset in the same way as the original target dataset, yielding the same number of features (but not necessarily the same number of instances). This processing includes cleaning, feature engineering, missing value imputation, feature scaling, and feature selection. Then EDA is repeated to confirm processing of the replication dataset and generate a feature correlation heatmap, however univariate analyses are not repeated on the replication data.

Next all models previously trained for the target datset are re-evaluted across all metrics using each replication dataset with results saved separately. Similarly all model evaluation plots (with the exception of feature importance plots) are automatically generated. As before non-parametric statistical tests are applied to examine differences in algorithm performance. 

* **Parallizability:** Runs once for each replication dataset being analyzed for a given target dataset.
* **Run Time:** Moderately fast

* **Output:** Similar outputs to Phase 1 minus univariate analyses, and similar outputs to Phase 6 minus feature importance assessments. 

***
## Phase 9: Summary Report
This final phase generates a pre-formatted PDF report summarizing (1) STREAMLINE run parameters, (2) key exploratory analysis for the processed target data, (3) key ML modeling results (including metrics and feature importances), (4) dataset comparisons (if relevant), (5) key statistical significance comparisons, and (6) runtime. STREAMLINE collects run-time information on each phase of the pipeline and for the training of each ML algorithm model.

Separate reports are generated representing the findings from running Phases 1-7, i.e. 'Testing Data Evaluation Report', as well as for Phase 8 if run on replication data, i.e. 'Replication Data Evaluation Report'. 

* **Parallizability:** Runs once - not parallizable
* **Run Time:** Moderately fast

* **Output:** One or more PDF reports