classifier_parameters = {
    'Naive Bayes': {},
    'Logistic Regression': {'penalty': ['l2', 'l1'],
                            'C': [1e-05, 100000.0],
                            'dual': [True, False],
                            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                            'class_weight': [None, 'balanced'],
                            'max_iter': [10, 1000]},
    'Decision Tree': {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [1, 30],
                      'min_samples_split': [2, 50],
                      'min_samples_leaf': [1, 50],
                      'max_features': [None, 'auto', 'log2'],
                      'class_weight': [None, 'balanced']},
    'Random Forest': {'n_estimators': [10, 1000],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [1, 30],
                      'min_samples_split': [2, 50],
                      'min_samples_leaf': [1, 50],
                      'max_features': [None, 'auto', 'log2'],
                      'bootstrap': [True],
                      'oob_score': [False, True],
                      'class_weight': [None, 'balanced']},
    'Gradient Boosting': {'n_estimators': [10, 1000],
                          'loss': ['deviance', 'exponential'],
                          'learning_rate': [0.0001, 0.3],
                          'min_samples_leaf': [1, 50],
                          'min_samples_split': [2, 50],
                          'max_depth': [1, 30]},
    'Extreme Gradient Boosting': {'booster': ['gbtree'],
                                  'objective': ['binary:logistic'],
                                  'verbosity': [0],
                                  'reg_lambda': [1e-08, 1.0],
                                  'alpha': [1e-08, 1.0],
                                  'eta': [1e-08, 1.0],
                                  'gamma': [1e-08, 1.0],
                                  'max_depth': [1, 30],
                                  'grow_policy': ['depthwise', 'lossguide'],
                                  'n_estimators': [10, 1000],
                                  'min_samples_split': [2, 50],
                                  'min_samples_leaf': [1, 50],
                                  'subsample': [0.5, 1.0],
                                  'min_child_weight': [0.1, 10],
                                  'colsample_bytree': [0.1, 1.0],
                                  'nthread': [1]},
    'Light Gradient Boosting': {'objective': ['binary'],
                                'metric': ['binary_logloss'],
                                'verbosity': [-1],
                                'boosting_type': ['gbdt'],
                                'num_leaves': [2, 256],
                                'max_depth': [1, 30],
                                'reg_alpha': [1e-08, 10.0],
                                'reg_lambda': [1e-08, 10.0],
                                'colsample_bytree': [0.4, 1.0],
                                'subsample': [0.4, 1.0],
                                'subsample_freq': [1, 7],
                                'min_child_samples': [5, 100],
                                'n_estimators': [10, 1000],
                                'num_threads': [1]},
    'Category Gradient Boosting': {'learning_rate': [0.0001, 0.3],
                                   'iterations': [10, 500],
                                   'depth': [1, 10],
                                   'l2_leaf_reg': [1, 9],
                                   'loss_function': ['Logloss'],
                                   'verbose': [False]},
    'Support Vector Machine': {'kernel': ['linear', 'poly', 'rbf'],
                               'C': [0.1, 1000],
                               'gamma': ['scale'],
                               'degree': [1, 6],
                               'probability': [True],
                               'class_weight': [None, 'balanced']},
    'Artificial Neural Network': {'n_layers': [1, 3],
                                  'layer_size': [1, 100],
                                  'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                  'momentum': [0.1, 0.9],
                                  'solver': ['sgd', 'adam'],
                                  'batch_size': ['auto'],
                                  'alpha': [0.0001, 0.05],
                                  'max_iter': [200]},
    'K-Nearest Neighbors': {'n_neighbors': [1, 100],
                            'weights': ['uniform', 'distance'],
                            'p': [1, 5],
                            'metric': ['euclidean', 'minkowski']},
    'Genetic Programming': {'population_size': [100, 1000],
                            'generations': [10, 500],
                            'tournament_size': [3, 50],
                            'init_method': ['grow', 'full', 'half and half'],
                            'function_set': [['add', 'sub', 'mul', 'div'],
                                             ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                              'abs', 'neg', 'inv', 'max', 'min'],
                                             ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                              'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']],
                            'parsimony_coefficient': [0.001, 0.01],
                            'low_memory': [True]},

    # eLCS
    "eLCS": {'learning_iterations': [100000, 200000, 500000], 'N': [1000, 2000, 5000],
             'nu': [1, 10], },
    # XCS
    "XCS": {'learning_iterations': [100000, 200000, 500000], 'N': [1000, 2000, 5000],
            'nu': [1, 10], },
    # ExSTraCS
    "ExSTraCS": {'learning_iterations': [100000, 200000, 500000], 'N': [1000, 2000, 5000],
                 'nu': [1, 10],
                 'rule_compaction': ['None', 'QRF']}
}

regressor_parameters = {
    # Linear Regression
    'Linear Regression': {},

    # Elastic Net Regressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
    'Elastic Net': {'alpha': [1e-3, 1], 'l1_ratio': [0, 1],
                    'max_iter': [2000, 2500]},

    # Random Forest Regressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    'Random Forest': {'n_estimators': [10, 1000], 'max_depth': [1, 30], 'min_samples_split': [2, 50],
                      'min_samples_leaf': [1, 50], 'max_features': [None, 'auto', 'log2'],
                      'bootstrap': [True], 'oob_score': [False, True]},

    # AdaBoost Regressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    'AdaBoost': {'n_estimators': [10, 1000], 'learning_rate': [.0001, 0.3],
                 'loss': ['linear', 'square', 'exponential']},

    # GradientBoosting Regressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    'Gradient Boost': {'learning_rate': [.0001, 0.3], 'n_estimators': [10, 1000],
                       'min_samples_leaf': [1, 50], 'min_samples_split': [2, 50], 'max_depth': [1, 30]},

    # Epsilon-Support Vector Regression
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    'Support Vector Regression': {'kernel': ['poly', 'rbf'], 'C': [0.1, 1000], 'gamma': ['scale'],
                                  'degree': [1, 6]},

    # Group Lasso Regressor
    # https://group-lasso.readthedocs.io/en/latest/api_reference.html#
    'Group Lasso': {'group_reg': [1e-3, 1],  # 'l1_reg':[0,1],
                    'n_iter': [2000, 2500],
                    'scale_reg': ['group_size', 'none', 'inverse_group_size'],
                    # 'subsampling_scheme': [0.1,0.9],
                    # 'frobenius_lipschitz': [True],
                    }
}


def get_parameters(algorithm_name, model_type="Classification"):
    """
    Get default model parameter range by model name
    Args:
        algorithm_name: name of model
        model_type: type of mode (one of Classification, Regression)

    Returns: default parameter grid as dict

    """
    if model_type == "Classification":
        return classifier_parameters[algorithm_name]
    elif model_type == "Regression":
        return regressor_parameters[algorithm_name]
