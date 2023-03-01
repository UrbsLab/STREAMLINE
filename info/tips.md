# Guidelines for Setting Parameters

## Reducing runtime
Conducting a more effective ML analysis typically demands a much larger amount of computing power and runtime. However, we provide general guidelines here for limiting overall runtime of a STREAMLINE experiment.
1. Run on a fewer number of datasets at once.
2. Run using fewer ML algorithms at once:
    * Naive Bayes, Logistic Regression, and Decision Trees are typically fastest.
    * Genetic Programming, eLCS, XCS, and ExSTraCS often take the longest (however other algorithms such as SVM, KNN, and ANN can take even longer when the number of instances is very large).
3. Run using a smaller number of `cv_partitions`.
4. Run without generating plots (i.e. `export_feature_correlations`, `export_univariate_plots`, `plot_PRC`, `plot_ROC`, `plot_FI_box`, `plot_metric_boxplots`).
5. In large datasets with missing values, set `multi_impute` to 'False'. This will apply simple mean imputation to numerical features instead.
6. Set `use_TURF` as 'False'. However we strongly recommend setting this to 'True' in feature spaces > 10,000 in order to avoid missing feature interactions during feature selection.
7. Set `TURF_pct` no lower than 0.5.  Setting at 0.5 is by far the fastest, but it will operate more effectively in very large feature spaces when set lower.
8. Set `instance_subset` at or below 2000 (speeds up multiSURF feature importance evaluation at potential expense of performance).
9. Set `max_features_to_keep` at or below 2000 and `filter_poor_features` = 'True' (this limits the maximum number of features that can be passed on to ML modeling).
10. Set `training_subsample` at or below 2000 (this limits the number of sample used to train particularly expensive ML modeling algorithms). However avoid setting this too low, or ML algorithms may not have enough training instances to effectively learn.
11. Set `n_trials` and/or timeout to lower values (this limits the time spent on hyperparameter optimization).
12. If using eLCS, XCS, or ExSTraCS, set `do_lcs_sweep` to 'False', `iterations` at or below 200000, and `N` at or below 2000.

## Improving Modeling Performance
* Generally speaking, the more computational time you are willing to spend on ML, the better the results. Doing the opposite of the above tips for reducing runtime, will likely improve performance.
* In certain situations, setting `feature_selection` to 'False', and relying on the ML algorithms alone to identify relevant features will yield better performance.  However, this may only be computationally practical when the total number of features in an original dataset is smaller (e.g. under 2000).
* Note that eLCS, XCS, and ExSTraCS are newer algorithm implementations developed by our research group.  As such, their algorithm performance may not yet be optimized in contrast to the other well established and widely utilized options. These learning classifier system (LCS) algorithms are unique however, in their ability to model very complex associations in data, while offering a largely interpretable model made up of simple, human readable IF:THEN rules. They have also been demonstrated to be able to tackle both complex feature interactions as well as heterogeneous patterns of association (i.e. different features are predictive in different subsets of the training data).
* In problems with no noise (i.e. datasets where it is possible to achieve 100% testing accuracy), LCS algorithms (i.e. eLCS, XCS, and ExSTraCS) perform better when `nu` is set larger than 1 (i.e. 5 or 10 recommended).  This applies significantly more pressure for individual rules to achieve perfect accuracy.  In noisy problems this may lead to significant overfitting.

## Other Guidelines
* SVM and ANN modeling should only be applied when data scaling is applied by the pipeline.
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline.  Otherwise `use_uniform_FI` should be True.
* While the STREAMLINE includes `impute_data` as an option that can be turned off in `DataPreprocessing`, most algorithm implementations (all those standard in scikit-learn) cannot handle missing data values with the exception of eLCS, XCS, and ExSTraCS. In general, STREAMLINE is expected to fail with an errors if run on data with missing values, while `impute_data` is set to 'False'.