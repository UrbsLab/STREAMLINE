def hyperparameters(random_state, do_lcs_sweep, nu, iterations, N, feature_names):
    """
    Hardcodes valid hyperparameter sweep ranges (specific to binary classification) for each machine learning algorithm.
    When run in the associated jupyter notebook, user can adjust these ranges directly.
    Here a user would need to edit the codebase to adjust the hyperparameter range as they are not included as
    run parameters of the pipeline (for simplicity). These hyperparameter
    ranges for each algorithm were chosen based on various recommendations from online tutorials,
    input from colleagues, and ML analysis
    papers. However, these settings will not necessarily be optimally efficient or effective for all problems,
    but instead offer a reasonable and rigorous range of hyperparameter options for this hyperparameter sweep.
    Learning classifier systems (eLCS, XCS, and ExSTraCS) are all evolutionary algorithms (i.e. stochastic)
    and thus can be computationally expensive. Often reasonable hyper parameters can be set
    for these algorithms without a hyperparameter sweep so the option is included to run or not run a
    hyperparameters sweep for the included learning classifier system algorithms.
    """
    # Add new algorithm hyperparameters here using same formatting...

    param_grid = {}
    # Naive Bayes - Has no hyper parameters
    # Logistic Regression (Note: can take longer to run in data with larger instance spaces)
    # Note some hyperparameter combinations are known to be invalid,
    # hyperparameter sweep will lose a trial attempt whenever this occurs.
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    param_grid_LR = {'penalty': ['l2', 'l1'], 'C': [1e-5, 1e5], 'dual': [True, False],
                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     'class_weight': [None, 'balanced'], 'max_iter': [10, 1000],
                     'random_state': [random_state]}
    # Decision Tree
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?
    # highlight=decision%20tree%20classifier#sklearn.tree.DecisionTreeClassifier
    param_grid_DT = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [1, 30],
                     'min_samples_split': [2, 50], 'min_samples_leaf': [1, 50], 'max_features': [None, 'auto', 'log2'],
                     'class_weight': [None, 'balanced'],
                     'random_state': [random_state]}
    # Random Forest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?
    # highlight=random%20forest#sklearn.ensemble.RandomForestClassifier
    param_grid_RF = {'n_estimators': [10, 1000], 'criterion': ['gini', 'entropy'], 'max_depth': [1, 30],
                     'min_samples_split': [2, 50], 'min_samples_leaf': [1, 50], 'max_features': [None, 'auto', 'log2'],
                     'bootstrap': [True], 'oob_score': [False, True], 'class_weight': [None, 'balanced'],
                     'random_state': [random_state]}
    # Gradient Boosting Trees
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?
    # highlight=gradient%20boosting#sklearn.ensemble.GradientBoostingClassifier
    param_grid_GB = {'n_estimators': [10, 1000], 'loss': ['deviance', 'exponential'], 'learning_rate': [.0001, 0.3],
                     'min_samples_leaf': [1, 50],
                     'min_samples_split': [2, 50], 'max_depth': [1, 30], 'random_state': [random_state]}
    # XG Boost (Note: Not great for large instance spaces (limited completion) and class weight balance
    # is included as option internally (Note: uses 'seed' instead of 'random_state')
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param_grid_XGB = {'booster': ['gbtree'], 'objective': ['binary:logistic'], 'verbosity': [0],
                      'reg_lambda': [1e-8, 1.0],
                      'alpha': [1e-8, 1.0], 'eta': [1e-8, 1.0], 'gamma': [1e-8, 1.0], 'max_depth': [1, 30],
                      'grow_policy': ['depthwise', 'lossguide'], 'n_estimators': [10, 1000],
                      'min_samples_split': [2, 50],
                      'min_samples_leaf': [1, 50], 'subsample': [0.5, 1.0], 'min_child_weight': [0.1, 10],
                      'colsample_bytree': [0.1, 1.0], 'nthread': [1], 'seed': [random_state]}
    # LG Boost (Note: class weight balance is included as option internally (still takes a while on
    # large instance spaces)) (Note: uses 'seed' instead of 'random_state')
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    param_grid_LGB = {'objective': ['binary'], 'metric': ['binary_logloss'], 'verbosity': [-1],
                      'boosting_type': ['gbdt'],
                      'num_leaves': [2, 256], 'max_depth': [1, 30], 'lambda_l1': [1e-8, 10.0],
                      'lambda_l2': [1e-8, 10.0],
                      'feature_fraction': [0.4, 1.0], 'bagging_fraction': [0.4, 1.0], 'bagging_freq': [1, 7],
                      'min_child_samples': [5, 100], 'n_estimators': [10, 1000], 'num_threads': [1],
                      'seed': [random_state]}
    # CatBoost - (Note this is newly added, and further optimization to this configuration is possible)
    # (Note: uses 'random_seed' instead of 'random_state')
    # https://catboost.ai/en/docs/references/training-parameters/
    param_grid_CGB = {'learning_rate': [.0001, 0.3], 'iterations': [10, 500], 'depth': [1, 10], 'l2_leaf_reg': [1, 9],
                      'loss_function': ['Logloss'], 'random_seed': [random_state]}
    # Support Vector Machine (Note: Very slow in large instance spaces)
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    param_grid_SVM = {'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1000], 'gamma': ['scale'], 'degree': [1, 6],
                      'probability': [True], 'class_weight': [None, 'balanced'], 'random_state': [random_state]}
    # Artificial Neural Network (Note: Slow in large instances spaces, and poor performer in small instance spaces)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?
    # highlight=artificial%20neural%20network
    param_grid_ANN = {'n_layers': [1, 3], 'layer_size': [1, 100],
                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'], 'momentum': [.1, .9],
                      'solver': ['sgd', 'adam'], 'batch_size': ['auto'], 'alpha': [0.0001, 0.05], 'max_iter': [200],
                      'random_state': [random_state]}
    # K-Nearest Neighbor Classifier (Note: Runs slowly in data with large instance space)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?
    # highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier
    param_grid_KNN = {'n_neighbors': [1, 100], 'weights': ['uniform', 'distance'], 'p': [1, 5],
                      'metric': ['euclidean', 'minkowski']}
    # Genetic Programming Symbolic Classifier
    # https://gplearn.readthedocs.io/en/stable/reference.html
    param_grid_GP = {'population_size': [100, 1000], 'generations': [10, 500], 'tournament_size': [3, 50],
                     'init_method': ['grow', 'full', 'half and half'],
                     'function_set': [['add', 'sub', 'mul', 'div'],
                                      ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'],
                                      ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min',
                                       'sin', 'cos', 'tan']],
                     'parsimony_coefficient': [0.001, 0.01], 'feature_names': [feature_names], 'low_memory': [True],
                     'random_state': [random_state]}
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    if eval(do_lcs_sweep):  # Contuct hyperparameter sweep of LCS algorithms (can be computationally expensive)
        # eLCS
        param_grid_eLCS = {'learning_iterations': [100000, 200000, 500000], 'N': [1000, 2000, 5000], 'nu': [1, 10],
                           'random_state': [random_state]}
        # XCS
        param_grid_XCS = {'learning_iterations': [100000, 200000, 500000], 'N': [1000, 2000, 5000], 'nu': [1, 10],
                          'random_state': [random_state]}
        # ExSTraCS
        param_grid_ExSTraCS = {'learning_iterations': [100000, 200000, 500000], 'N': [1000, 2000, 5000], 'nu': [1, 10],
                               'random_state': [random_state], 'rule_compaction': ['None', 'QRF']}
    else:  # Run LCS algorithms with fixed hyperparameters
        # eLCS
        param_grid_eLCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state]}
        # XCS
        param_grid_XCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state]}
        # ExSTraCS
        param_grid_ExSTraCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu],
                               'random_state': [random_state], 'rule_compaction': ['QRF']}  # 'QRF', 'None'

    param_grid['Naive Bayes'] = {}
    param_grid['Logistic Regression'] = param_grid_LR
    param_grid['Decision Tree'] = param_grid_DT
    param_grid['Random Forest'] = param_grid_RF
    param_grid['Gradient Boosting'] = param_grid_GB
    param_grid['Extreme Gradient Boosting'] = param_grid_XGB
    param_grid['Light Gradient Boosting'] = param_grid_LGB
    param_grid['Category Gradient Boosting'] = param_grid_CGB
    param_grid['Support Vector Machine'] = param_grid_SVM
    param_grid['Artificial Neural Network'] = param_grid_ANN
    param_grid['K-Nearest Neightbors'] = param_grid_KNN
    param_grid['Genetic Programming'] = param_grid_GP
    param_grid['eLCS'] = param_grid_eLCS
    param_grid['XCS'] = param_grid_XCS
    param_grid['ExSTraCS'] = param_grid_ExSTraCS
    # Add new algorithms here...
    return param_grid
