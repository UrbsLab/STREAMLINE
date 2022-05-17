"""
File:ModelJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 5 of STREAMLINE - This 'Job' script is called by ModelMain.py and runs machine learning modeling using respective training datasets.
            This pipeline currently includes the following 13 ML modeling algorithms for binary classification:
            * Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LGBoost, CatBoost, Support Vector Machine (SVM), Artificial Neural Network (ANN),
            * k Nearest Neighbors (k-NN), Educational Learning Classifier System (eLCS), X Classifier System (XCS), and the Extended Supervised Tracking and Classifying System (ExSTraCS)
            This phase includes hyperparameter optimization of all algorithms (other than naive bayes), model training, model feature importance estimation (using internal algorithm
            estimations, if available, or via permutation feature importance), and performance evaluation on hold out testing data. This script runs for a single combination of a
            cv dataset (for each original target dataset) and ML modeling algorithm.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import sys
import time
import random
import pandas as pd
import numpy as np
import os
import pickle
import copy
import math
#Model Packages:
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from gplearn.genetic import SymbolicClassifier
#import gplearn as gp
from skeLCS import eLCS
from skXCS import XCS
from skExSTraCS import ExSTraCS
#Evalutation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import metrics
#Other packages
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import optuna #hyperparameter optimization

def job(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric,algAbrev,jupyterRun):
    """ Specifies hardcoded (below) range of hyperparameter options selected for each ML algorithm and then runs the modeling method. Set up this way so that users can easily modify ML hyperparameter settings when running from the Jupyter Notebook. """
    #Add spaces back to algorithm names
    algorithm = algorithm.replace("_", " ")
    if eval(jupyterRun):
        print('Running '+str(algorithm)+' on '+str(train_file_path))
    #Get header names for current CV dataset for use later in GP tree visulaization
    data_name = full_path.split('/')[-1]
    feature_names = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_'+str(cvCount)+'_Test.csv').columns.values.tolist()
    if instance_label != 'None':
        feature_names.remove(instance_label)
    feature_names.remove(class_label)
    #Get hyperparameter grid
    param_grid = hyperparameters(random_state,do_lcs_sweep,nu,iterations,N,feature_names)[algorithm]
    runModel(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric,param_grid,algAbrev)


def runModel(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric,param_grid,algAbrev):
    """ Run all elements of modeling: loading data, hyperparameter optimization, model training, and evaluation on hold out testing data.  Each ML algorithm has its own method below to handle these steps. """
    job_start_time = time.time() #for tracking phase runtime
    # Set random seeds for replicatability
    random.seed(random_state)
    np.random.seed(random_state)
    #Load training and testing datasets separating features from outcome for scikit-learn-based modeling
    trainX,trainY,testX,testY = dataPrep(train_file_path,instance_label,class_label,test_file_path)
    #Run ml modeling algorithm specified-------------------------------------------------------------------------------------------------------------------------------------------------------------
    if algorithm == 'Naive Bayes':
        ret = run_NB_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Logistic Regression':
        ret = run_LR_full(trainX,trainY,testX,testY, random_state, cvCount,param_grid,n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Decision Tree':
        ret = run_DT_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Random Forest':
        ret = run_RF_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Gradient Boosting':
        ret = run_GB_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Extreme Gradient Boosting':
        ret = run_XGB_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots,full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'Light Gradient Boosting':
        ret = run_LGB_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Category Gradient Boosting':
        ret = run_CGB_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'Support Vector Machine':
        ret = run_SVM_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'Artificial Neural Network':
        ret = run_ANN_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'K-Nearest Neightbors':
        ret = run_KNN_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'Genetic Programming':
        ret = run_GP_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,training_subsample,use_uniform_FI,primary_metric)
    #Experimental ML modeling algorithms (developed by our research group)
    elif algorithm == 'eLCS':
        ret = run_eLCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'XCS':
        ret = run_XCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'ExSTraCS':
        ret = run_ExSTraCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,export_hyper_sweep_plots, full_path,filter_poor_features,instance_label,class_label,use_uniform_FI,primary_metric)
    ### Add new algorithms here...
    #Pickle all evaluation metrics for ML model training and evaluation
    if not os.path.exists(full_path+'/model_evaluation/pickled_metrics'):
        os.mkdir(full_path+'/model_evaluation/pickled_metrics')
    pickle.dump(ret, open(full_path + '/model_evaluation/pickled_metrics/' + algAbrev + '_CV_' + str(cvCount) + "_metrics.pickle", 'wb'))
    #Save runtime of ml algorithm training and evaluation
    saveRuntime(full_path,job_start_time,algAbrev,algorithm,cvCount)
    # Print phase completion
    print(full_path.split('/')[-1] + " [CV_" + str(cvCount) + "] ("+algAbrev+") training complete. ------------------------------------")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_model_' + full_path.split('/')[-1] + '_' + str(cvCount) +'_' +algAbrev+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def dataPrep(train_file_path,instance_label,class_label,test_file_path):
    """ Loads target cv training dataset, separates class from features and removes instance labels."""
    train = pd.read_csv(train_file_path)
    if instance_label != 'None':
        train = train.drop(instance_label,axis=1)
    trainX = train.drop(class_label,axis=1).values
    trainY = train[class_label].values
    del train #memory cleanup
    test = pd.read_csv(test_file_path)
    if instance_label != 'None':
        test = test.drop(instance_label,axis=1)
    testX = test.drop(class_label,axis=1).values
    testY = test[class_label].values
    del test #memory cleanup
    return trainX,trainY,testX,testY

def saveRuntime(full_path,job_start_time,algAbrev,algorithm,cvCount):
    """ Save ML algorithm training and evaluation runtime for this phase."""
    runtime_file = open(full_path + '/runtime/runtime_'+algAbrev+'_CV'+str(cvCount)+'.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

def hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric):
    """ Run hyperparameter evaluation for a given ML algorithm using Optuna. Uses further k-fold cv within target training data for hyperparameter evaluation."""
    cv = StratifiedKFold(n_splits=hype_cv, shuffle=True, random_state=randSeed)
    model = clone(est).set_params(**params)
    #Flexibly handle whether random seed is given as 'random_seed' or 'seed' - scikit learn uses 'random_seed'
    for a in ['random_state','seed']:
        if hasattr(model,a):
            setattr(model,a,randSeed)
    performance = np.mean(cross_val_score(model,x_train,y_train,cv=cv,scoring=scoring_metric,verbose=0))
    return performance

def modelEvaluation(clf,model,x_test,y_test):
    """ Runs commands to gather all evaluations for later summaries and plots. """
    #Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)
    #Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])
    return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_

#Naive Bayes #############################################################################################################################
def run_NB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Naive Bayes model training, evaluation, and model feature importance estimation. No hyperparameters to optimize."""
    #Train model using 'best' hyperparameters - Uses default 3-fold internal CV (training/validation splits)
    clf = GaussianNB()
    model = clf.fit(x_train, y_train)
    #Save model
    pickle.dump(model, open(full_path+'/models/pickledModels/NB_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    #Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Logistic Regression ###################################################################################################################
def objective_LR(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Logistic Regression hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'penalty' : trial.suggest_categorical('penalty',param_grid['penalty']),
			  'dual' : trial.suggest_categorical('dual', param_grid['dual']),
			  'C' : trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
			  'solver' : trial.suggest_categorical('solver',param_grid['solver']),
			  'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
			  'max_iter' : trial.suggest_loguniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1]),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_LR_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Logistic Regression hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = LogisticRegression()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_LR(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/LR_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = LogisticRegression()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/LR_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/LR_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/LR_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = pow(math.e,model.coef_[0])
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Decision Tree #####################################################################################################################################
def objective_DT(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Decision Tree hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'splitter' : trial.suggest_categorical('splitter', param_grid['splitter']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_DT_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Decision Tree hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = tree.DecisionTreeClassifier()

    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_DT(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/DT_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = tree.DecisionTreeClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/DT_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/DT_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/DT_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Random Forest ######################################################################################################################################
def objective_RF(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Random Forest hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'bootstrap' : trial.suggest_categorical('bootstrap',param_grid['bootstrap']),
                'oob_score' : trial.suggest_categorical('oob_score',param_grid['oob_score']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_RF_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Random Forest hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = RandomForestClassifier()

    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_RF(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/RF_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = RandomForestClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/RF_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/RF_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/RF_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Gradient Boosting Classifier #####################################################################################################################
def objective_GB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Gradient Boosting hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'loss': trial.suggest_categorical('loss', param_grid['loss']),
                'learning_rate': trial.suggest_loguniform('learning_rate', param_grid['learning_rate'][0],param_grid['learning_rate'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0],param_grid['min_samples_leaf'][1]),
                'min_samples_split': trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0],param_grid['min_samples_split'][1]),
                'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_GB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Gradient Boosting hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = GradientBoostingClassifier()

    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_GB(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/GB_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = GradientBoostingClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/GB_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/GB_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/GB_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#XGBoost ###################################################################################################################################################
def objective_XGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares XGBoost hyperparameter variables for Optuna run hyperparameter optimization. """
    posInst = sum(y_train)
    negInst = len(y_train) - posInst
    classWeight = negInst/float(posInst)
    params = {'booster' : trial.suggest_categorical('booster',param_grid['booster']),
                'objective' : trial.suggest_categorical('objective',param_grid['objective']),
                'verbosity' : trial.suggest_categorical('verbosity',param_grid['verbosity']),
                'reg_lambda' : trial.suggest_loguniform('reg_lambda', param_grid['reg_lambda'][0], param_grid['reg_lambda'][1]),
                'alpha' : trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
                'eta' : trial.suggest_loguniform('eta', param_grid['eta'][0], param_grid['eta'][1]),
                'gamma' : trial.suggest_loguniform('gamma', param_grid['gamma'][0], param_grid['gamma'][1]),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'grow_policy' : trial.suggest_categorical('grow_policy',param_grid['grow_policy']),
                'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'subsample' : trial.suggest_uniform('subsample', param_grid['subsample'][0], param_grid['subsample'][1]),
                'min_child_weight' : trial.suggest_loguniform('min_child_weight', param_grid['min_child_weight'][0], param_grid['min_child_weight'][1]),
                'colsample_bytree' : trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0], param_grid['colsample_bytree'][1]),
                'scale_pos_weight' : trial.suggest_categorical('scale_pos_weight', [1.0, classWeight]),
                'nthread' : trial.suggest_categorical('nthread',param_grid['nthread']),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_XGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    """ Run XGBoost hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Utilize a subset of training instances to reduce runtime on a very large dataset
    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For XGB training sample reduced to '+str(x_train.shape[0])+' instances')
    #Specify algorithm for hyperparameter optimization
    est = xgb.XGBClassifier()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_XGB(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/XGB_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = xgb.XGBClassifier()
        clf = clone(est).set_params(**best_trial.params)
        export_best_params(full_path + '/models/XGB_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
        setattr(clf, 'random_state', randSeed)
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/XGB_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/XGB_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#LGBoost #########################################################################################################################################
def objective_LGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares LGBoost hyperparameter variables for Optuna run hyperparameter optimization. """
    posInst = sum(y_train)
    negInst = len(y_train) - posInst
    classWeight = negInst / float(posInst)
    params = {'objective': trial.suggest_categorical('objective', param_grid['objective']),
              'metric': trial.suggest_categorical('metric', param_grid['metric']),
              'verbosity': trial.suggest_categorical('verbosity', param_grid['verbosity']),
              'boosting_type': trial.suggest_categorical('boosting_type', param_grid['boosting_type']),
              'num_leaves': trial.suggest_int('num_leaves', param_grid['num_leaves'][0], param_grid['num_leaves'][1]),
              'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
              'lambda_l1': trial.suggest_loguniform('lambda_l1', param_grid['lambda_l1'][0],param_grid['lambda_l1'][1]),
              'lambda_l2': trial.suggest_loguniform('lambda_l2', param_grid['lambda_l2'][0],param_grid['lambda_l2'][1]),
              'feature_fraction': trial.suggest_uniform('feature_fraction', param_grid['feature_fraction'][0],param_grid['feature_fraction'][1]),
              'bagging_fraction': trial.suggest_uniform('bagging_fraction', param_grid['bagging_fraction'][0],param_grid['bagging_fraction'][1]),
              'bagging_freq': trial.suggest_int('bagging_freq', param_grid['bagging_freq'][0],param_grid['bagging_freq'][1]),
              'min_child_samples': trial.suggest_int('min_child_samples', param_grid['min_child_samples'][0],param_grid['min_child_samples'][1]),
              'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0],param_grid['n_estimators'][1]),
              'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, classWeight]),
              'num_threads' : trial.suggest_categorical('num_threads',param_grid['num_threads']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_LGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run LGBoost hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = lgb.LGBMClassifier()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_LGB(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/LGB_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = lgb.LGBMClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/LGB_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/LGB_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/LGB_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#CatBoost #########################################################################################################################################
def objective_CGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Catboost hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'learning_rate': trial.suggest_loguniform('learning_rate',param_grid['learning_rate'][0],param_grid['learning_rate'][1]),
              'iterations': trial.suggest_int('iterations',param_grid['iterations'][0],param_grid['iterations'][1]),
              'depth': trial.suggest_int('depth',param_grid['depth'][0],param_grid['depth'][1]),
              'l2_leaf_reg': trial.suggest_int('l2_leaf_reg',param_grid['l2_leaf_reg'][0],param_grid['l2_leaf_reg'][1]),
              'loss_function': trial.suggest_categorical('loss_function',param_grid['loss_function']),
              'random_seed' : trial.suggest_categorical('random_seed',param_grid['random_seed'])}

    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)


def run_CGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run CGBoost hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = cgb.CatBoostClassifier(verbose=0)
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_CGB(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/CGB_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = cgb.CatBoostClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/CGB_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/CGB_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train, verbose=0)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/CGB_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Support Vector Machines #####################################################################################################################################
def objective_SVM(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Support Vector Machine hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'kernel': trial.suggest_categorical('kernel', param_grid['kernel']),
              'C': trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
              'gamma': trial.suggest_categorical('gamma', param_grid['gamma']),
              'degree': trial.suggest_int('degree', param_grid['degree'][0], param_grid['degree'][1]),
              'probability': trial.suggest_categorical('probability', param_grid['probability']),
              'class_weight': trial.suggest_categorical('class_weight', param_grid['class_weight']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_SVM_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    """ Run Support Vector Machine hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Utilize a subset of training instances to reduce runtime on a very large dataset
    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For SVM, training sample reduced to '+str(x_train.shape[0])+' instances')
    #Specify algorithm for hyperparameter optimization
    est = SVC()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)  #.CRITICAL
        study.optimize(lambda trial: objective_SVM(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/SVM_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = SVC()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/SVM_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/SVM_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/SVM_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates (SVM can only automatically obtain feature importance estimates (coef_) for linear kernel)
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Artificial Neural Networks #######################################################################################################################
def objective_ANN(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Artificial Neural Network hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'activation': trial.suggest_categorical('activation', param_grid['activation']),
              'learning_rate': trial.suggest_categorical('learning_rate', param_grid['learning_rate']),
              'momentum': trial.suggest_uniform('momentum', param_grid['momentum'][0], param_grid['momentum'][1]),
              'solver': trial.suggest_categorical('solver', param_grid['solver']),
              'batch_size': trial.suggest_categorical('batch_size', param_grid['batch_size']),
              'alpha': trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
              'max_iter': trial.suggest_categorical('max_iter', param_grid['max_iter']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    n_layers = trial.suggest_int('n_layers', param_grid['n_layers'][0], param_grid['n_layers'][1])
    layers = []
    for i in range(n_layers):
        layers.append(
            trial.suggest_int('n_units_l{}'.format(i), param_grid['layer_size'][0], param_grid['layer_size'][1]))
        params['hidden_layer_sizes'] = tuple(layers)
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_ANN_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    """ Run Artificial Neural Network hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Utilize a subset of training instances to reduce runtime on a very large dataset
    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For ANN, training sample reduced to '+str(x_train.shape[0])+' instances')
    #Specify algorithm for hyperparameter optimization
    est = MLPClassifier()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_ANN(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            try:
                fig = optuna.visualization.plot_parallel_coordinate(study)
                fig.write_image(full_path+'/models/ANN_ParamOptimization_'+str(i)+'.png')
            except:
                pass
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Handle special parameter requirement for ANN
        layers = []
        for j in range(best_trial.params['n_layers']):
            layer_name = 'n_units_l' + str(j)
            layers.append(best_trial.params[layer_name])
            del best_trial.params[layer_name]
        best_trial.params['hidden_layer_sizes'] = tuple(layers)
        del best_trial.params['n_layers']
        # Specify model with optimized hyperparameters
        est = MLPClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/ANN_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/ANN_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/ANN_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#K-Neighbors Classifier ####################################################################################################################################
def objective_KNN(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares K Nearest Neighbor hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', param_grid['n_neighbors'][0], param_grid['n_neighbors'][1]),
        'weights': trial.suggest_categorical('weights', param_grid['weights']),
        'p': trial.suggest_int('p', param_grid['p'][0], param_grid['p'][1]),
        'metric': trial.suggest_categorical('metric', param_grid['metric'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_KNN_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    """ Run K Nearest Neighbors Classifier hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Utilize a subset of training instances to reduce runtime on a very large dataset
    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For KNN, training sample reduced to '+str(x_train.shape[0])+' instances')
    #Specify algorithm for hyperparameter optimization
    est = KNeighborsClassifier()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_KNN(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/KNN_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = KNeighborsClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/KNN_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/KNN_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/KNN_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Genetic Programming (Symbolic classification) ####################################################################################################################################
def objective_GP(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares genetic programming hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'population_size': trial.suggest_int('population_size', param_grid['population_size'][0], param_grid['population_size'][1]),
              'generations': trial.suggest_int('generations', param_grid['generations'][0], param_grid['generations'][1]),
              'tournament_size': trial.suggest_int('tournament_size', param_grid['tournament_size'][0], param_grid['tournament_size'][1]),
              'function_set': trial.suggest_categorical('function_set', param_grid['function_set']),
              'init_method': trial.suggest_categorical('init_method', param_grid['init_method']),
              'parsimony_coefficient': trial.suggest_float('parsimony_coefficient', param_grid['parsimony_coefficient'][0], param_grid['parsimony_coefficient'][1]),
              'feature_names': trial.suggest_categorical('feature_names', param_grid['feature_names']),
              'low_memory': trial.suggest_categorical('low_memory', param_grid['low_memory']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}

    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_GP_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    """ Run Genetic Programming Symbolic Classifier hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Utilize a subset of training instances to reduce runtime on a very large dataset
    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For GP, training sample reduced to '+str(x_train.shape[0])+' instances')
    #Specify algorithm for hyperparameter optimization
    est = SymbolicClassifier()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_GP(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user #Currently because some hyperparameters are lists, this breaks the visualization
        #if eval(do_plot):
        #    fig = optuna.visualization.plot_parallel_coordinate(study)
        #    fig.write_image(full_path+'/models/GP_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        est = SymbolicClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/GP_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/GP_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/GP_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#Experimental ML modeling algorithms (developed by our research group)
#eLCS (educational learning Classifier System) ##################################################################################################################################
def objective_eLCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares Eductional Learning Classifier System hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
        'N': trial.suggest_categorical('N', param_grid['N']),
        'nu': trial.suggest_categorical('nu', param_grid['nu']),
        'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_eLCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run Educational Learning Classifier System hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = eLCS()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_eLCS(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/eLCS_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/eLCS_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/eLCS_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/eLCS_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.get_final_attribute_specificity_list()
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#XCS ('X' Learning classifier system) ############################################################################################################################################
def objective_XCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares X Learning Classifier System hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
        'N': trial.suggest_categorical('N', param_grid['N']),
        'nu': trial.suggest_categorical('nu', param_grid['nu']),
        'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_XCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    """ Run X Learning Classifier System (XCS) hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = XCS()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_XCS(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/XCS_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/models/XCS_bestparams' + str(i) + '.csv', best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/XCS_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/XCS_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.get_final_attribute_specificity_list()
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]

#ExSTraCS (Extended supervised tracking and classifying system) #############################################################################
def objective_ExSTraCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    """ Prepares ExSTraCS hyperparameter variables for Optuna run hyperparameter optimization. """
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
              'N': trial.suggest_categorical('N', param_grid['N']), 'nu': trial.suggest_categorical('nu', param_grid['nu']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state']),
              'expert_knowledge':param_grid['expert_knowledge'],
              'rule_compaction':trial.suggest_categorical('rule_compaction', param_grid['rule_compaction'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def get_FI_subset_ExSTraCS(full_path,i,instance_label,class_label,filter_poor_features):
    """ For ExSTraCS, gets the MultiSURF (or MI if MS not availabile) FI scores for the feature subset being analyzed here in modeling"""
    scores = [] #to be filled in, in filted dataset order.
    data_name = full_path.split('/')[-1]
    if os.path.exists(full_path + "/multisurf/pickledForPhase4/"):  # If MultiSURF was done previously:
        algorithmlabel = 'multisurf'
    elif os.path.exists(full_path + "/mutualinformation/pickledForPhase4/"):  # If MI was done previously and MS wasn't:
        algorithmlabel = 'mutualinformation'
    else:
        scores = []
        return scores
    if eval(filter_poor_features): #obtain feature importance scores for feature subset analyzed (in correct training dataset order)
        #Load current data ordered_feature_names
        header = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_'+str(i)+'_Test.csv').columns.values.tolist()
        if instance_label != 'None':
            header.remove(instance_label)
        header.remove(class_label)
        #Load orignal dataset multisurf scores
        scoreInfo = full_path+ "/"+algorithmlabel+"/pickledForPhase4/"+str(i)+'.pickle'
        file = open(scoreInfo, 'rb')
        rawData = pickle.load(file)
        file.close()
        scoreDict = rawData[1]
        #Generate filtered multisurf score list with same order as working datasets
        for each in header:
            scores.append(scoreDict[each])
    else: #obtain feature importances scores for all features (i.e. no feature selection was conducted)
        #Load orignal dataset multisurf scores
        scoreInfo = full_path+ "/"+algorithmlabel+"/pickledForPhase4/"+str(i)+'.pickle'
        file = open(scoreInfo, 'rb')
        rawData = pickle.load(file)
        file.close()
        scores = rawData[0]
    return scores

def run_ExSTraCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,filter_poor_features,instance_label,class_label,use_uniform_FI,primary_metric):
    """ Run Extended Supervised Tracking and Classifying System (ExSTraCS) hyperparameter optimization, model training, evaluation, and model feature importance estimation. """
    #Grab feature importance weights from multiSURF, used by ExSTraCS
    scores = get_FI_subset_ExSTraCS(full_path,i,instance_label,class_label,filter_poor_features)
    param_grid['expert_knowledge'] = scores
    #Check whether hyperparameters are fixed (i.e. no hyperparameter sweep required) or whether a set/range of values were specified for any hyperparameter (conduct hyperparameter sweep)
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1 and key != 'expert_knowledge':
            isSingle = False
    #Specify algorithm for hyperparameter optimization
    est = ExSTraCS()
    if not isSingle: #Run hyperparameter sweep
        #Apply Optuna-----------------------------------------
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(lambda trial: objective_ExSTraCS(trial, est, x_train, y_train, randSeed, 3, param_grid,primary_metric), n_trials=n_trials, timeout=timeout,catch=(ValueError,))
        #Export hyperparameter optimization search visualization if specified by user
        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/models/ExSTraCS_ParamOptimization_'+str(i)+'.png')
        #Print results and hyperparamter values for best hyperparameter sweep trial
        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))
        # Specify model with optimized hyperparameters
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path+'/models/ExSTraCS_bestparams'+str(i)+'.csv',best_trial.params) #Export final model hyperparamters to csv file
    else: #Specify hyperparameter values (no sweep)
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            if key == 'expert_knowledge':
                params[key] = value
            else:
                params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/models/ExSTraCS_usedparams' + str(i) + '.csv', params) #Export final model hyperparamters to csv file
    print(clf) #Print basic classifier info/hyperparmeters for verification
    #Train final model using whole training dataset and 'best' hyperparameters
    model = clf.fit(x_train, y_train)
    # Save model with pickle so it can be applied in the future
    pickle.dump(model, open(full_path+'/models/pickledModels/ExSTraCS_'+str(i)+'.pickle', 'wb'))
    #Evaluate model
    metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_ = modelEvaluation(clf,model,x_test,y_test)
    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.get_final_attribute_specificity_list()
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi, probas_]


def export_best_params(file_name,param_grid):
    """ Exports best hyperparameter scores to output file."""
    best_params_copy = param_grid
    for best in best_params_copy:
        best_params_copy[best] = [best_params_copy[best]]
    df = pd.DataFrame.from_dict(best_params_copy)
    df.to_csv(file_name, index=False)

def classEval(y_true, y_pred):
    """ Calculates standard classification metrics including:
    True positives, false positives, true negative, false negatives, standard accuracy, balanced accuracy
    recall, precision, f1 score, negative predictive value, likelihood ratio positive, and likelihood ratio negative"""
    #Calculate true positive, true negative, false positive, and false negative.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #Calculate Accuracy metrics
    ac = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    #Calculate Precision and Recall
    re = recall_score(y_true, y_pred) #a.k.a. sensitivity or TPR
    pr = precision_score(y_true, y_pred)
    #Calculate F1 score
    f1 = f1_score(y_true, y_pred)
    # Calculate specificity, a.k.a. TNR
    if tn == 0 and fp == 0:
        sp = 0
    else:
        sp = tn / float(tn + fp)
    # Calculate Negative predictive value
    if tn == 0 and fn == 0:
        npv = 0
    else:
        npv = tn/float(tn+fn)
    # Calculate likelihood ratio postive
    if sp == 1:
        lrp = 0
    else:
        lrp = re/float(1-sp)  #sensitivity / (1-specificity).... a.k.a. TPR/FPR... or TPR/(1-TNR)
    # Calculate likeliehood ratio negative
    if sp == 0:
        lrm = 0
    else:
        lrm = (1-re)/float(sp) #(1-sensitivity) / specificity... a.k.a. FNR/TNR ... or (1-TPR)/TNR
    return [bac, ac, f1, re, sp, pr, tp, tn, fp, fn, npv, lrp, lrm]

def hyperparameters(random_state,do_lcs_sweep,nu,iterations,N,feature_names): #### Add new algorithm hyperparameters here using same formatting...
    """ Hardcodes valid hyperparamter sweep ranges (specific to binary classification) for each machine learning algorithm.
    When run in the associated jupyer notebook, user can adjust these ranges directly.  Here a user would need to edit the codebase
    to adjust the hyperparameter range as they are not included as run parameters of the pipeline (for simplicity). These hyperparameter
    ranges for each algorithm were chosen based on various recommendations from online tutorials, input from colleagues, and ML analysis
    papers. However, these settings will not necessarily be optimally efficient or effective for all problems, but instead offer a
    reasonable and rigorous range of hyperparameter options for this hyperparameter sweep. Learning classifier systems (eLCS, XCS, and ExSTraCS)
    are all evolutionary algorithms (i.e. stochastic) and thus can be computationally expensive. Often reasonable hyperparameters can be set
    for these algorithms without a hyperparameter sweep so the option is included to run or not run a hyperparameters sweep for the included
    learning classifier system algorithms. """
    param_grid = {}
    # Naive Bayes - Has no hyperparameters
    # Logistic Regression (Note: can take longer to run in data with larger instance spaces) Note some hyperparameter combinations are known to be invalid, hyperparameter sweep will lose a trial attempt whenever this occurs.
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    param_grid_LR = {'penalty': ['l2', 'l1'],'C': [1e-5, 1e5],'dual': [True, False],
                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     'class_weight': [None, 'balanced'],'max_iter': [10, 1000],
                     'random_state':[random_state]}
    # Decision Tree
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree%20classifier#sklearn.tree.DecisionTreeClassifier
    param_grid_DT = {'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'],'max_depth': [1, 30],
                     'min_samples_split': [2, 50],'min_samples_leaf': [1, 50],'max_features': [None, 'auto', 'log2'],
                     'class_weight': [None, 'balanced'],
                     'random_state':[random_state]}
    # Random Forest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier
    param_grid_RF = {'n_estimators': [10, 1000],'criterion': ['gini', 'entropy'],'max_depth': [1, 30],
                     'min_samples_split': [2, 50],'min_samples_leaf': [1, 50],'max_features': [None, 'auto', 'log2'],
                     'bootstrap': [True],'oob_score': [False, True],'class_weight': [None, 'balanced'],
                     'random_state':[random_state]}
    # Gradient Boosting Trees
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gradient%20boosting#sklearn.ensemble.GradientBoostingClassifier
    param_grid_GB = {'n_estimators': [10, 1000],'loss': ['deviance', 'exponential'], 'learning_rate': [.0001, 0.3], 'min_samples_leaf': [1, 50],
                     'min_samples_split': [2, 50], 'max_depth': [1, 30],'random_state':[random_state]}
    # XG Boost (Note: Not great for large instance spaces (limited completion) and class weight balance is included as option internally
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param_grid_XGB = {'booster': ['gbtree'],'objective': ['binary:logistic'],'verbosity': [0],'reg_lambda': [1e-8, 1.0],
                      'alpha': [1e-8, 1.0],'eta': [1e-8, 1.0],'gamma': [1e-8, 1.0],'max_depth': [1, 30],
                      'grow_policy': ['depthwise', 'lossguide'],'n_estimators': [10, 1000],'min_samples_split': [2, 50],
                      'min_samples_leaf': [1, 50],'subsample': [0.5, 1.0],'min_child_weight': [0.1, 10],
                      'colsample_bytree': [0.1, 1.0],'nthread':[1],'random_state':[random_state]}
    # LG Boost (Note: class weight balance is included as option internally (still takes a while on large instance spaces))
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    param_grid_LGB = {'objective': ['binary'],'metric': ['binary_logloss'],'verbosity': [-1],'boosting_type': ['gbdt'],
                      'num_leaves': [2, 256],'max_depth': [1, 30],'lambda_l1': [1e-8, 10.0],'lambda_l2': [1e-8, 10.0],
                      'feature_fraction': [0.4, 1.0],'bagging_fraction': [0.4, 1.0],'bagging_freq': [1, 7],
                      'min_child_samples': [5, 100],'n_estimators': [10, 1000],'num_threads':[1],'random_state':[random_state]}
    # CatBoost - (Note this is newly added, and further optimization to this configuration is possible)
    # https://catboost.ai/en/docs/references/training-parameters/
    param_grid_CGB = {'learning_rate':[.0001, 0.3],'iterations':[10,500],'depth':[1,10],'l2_leaf_reg': [1,9],
                      'loss_function': ['Logloss'], 'random_seed': [random_state]}
    # Support Vector Machine (Note: Very slow in large instance spaces)
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    param_grid_SVM = {'kernel': ['linear', 'poly', 'rbf'],'C': [0.1, 1000],'gamma': ['scale'],'degree': [1, 6],
                      'probability': [True],'class_weight': [None, 'balanced'],'random_state':[random_state]}
    # Artificial Neural Network (Note: Slow in large instances spaces, and poor performer in small instance spaces)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=artificial%20neural%20network
    param_grid_ANN = {'n_layers': [1, 3],'layer_size': [1, 100],'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],'momentum': [.1, .9],
                      'solver': ['sgd', 'adam'],'batch_size': ['auto'],'alpha': [0.0001, 0.05],'max_iter': [200],'random_state':[random_state]}
    # K-Nearest Neighbor Classifier (Note: Runs slowly in data with large instance space)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier
    param_grid_KNN = {'n_neighbors': [1, 100], 'weights': ['uniform', 'distance'], 'p': [1, 5],
                     'metric': ['euclidean', 'minkowski']}
    # Genetic Programming Symbolic Classifier
    # https://gplearn.readthedocs.io/en/stable/reference.html
    param_grid_GP = {'population_size': [100, 1000], 'generations': [10, 500], 'tournament_size': [3, 50],'init_method': ['grow', 'full','half and half'],
                     'function_set': [['add', 'sub', 'mul', 'div'], ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'], ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min','sin','cos','tan']],
                     'parsimony_coefficient': [0.001,0.01],'feature_names': [feature_names], 'low_memory': [True],'random_state': [random_state]}
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    if eval(do_lcs_sweep): #Contuct hyperparameter sweep of LCS algorithms (can be computationally expensive)
        # eLCS
        param_grid_eLCS = {'learning_iterations': [100000,200000,500000],'N': [1000,2000,5000],'nu': [1,10],
                           'random_state':[random_state]}
        # XCS
        param_grid_XCS = {'learning_iterations': [100000,200000,500000],'N': [1000,2000,5000],'nu': [1,10],
                          'random_state':[random_state]}
        # ExSTraCS
        param_grid_ExSTraCS = {'learning_iterations': [100000,200000,500000],'N': [1000,2000,5000],'nu': [1,10],
                               'random_state':[random_state],'rule_compaction':['None','QRF']}
    else: #Run LCS algorithms with fixed hyperparameters
        # eLCS
        param_grid_eLCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state]}
        # XCS
        param_grid_XCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state]}
        # ExSTraCS
        param_grid_ExSTraCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state], 'rule_compaction': ['QRF']} #'QRF', 'None'

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
    ### Add new algorithms here...
    return param_grid

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),sys.argv[8],sys.argv[9],sys.argv[10],int(sys.argv[11]),int(sys.argv[12]),sys.argv[13],sys.argv[14],int(sys.argv[15]),int(sys.argv[16]),int(sys.argv[17]),int(sys.argv[18]),sys.argv[19],sys.argv[20],sys.argv[21],sys.argv[22])
