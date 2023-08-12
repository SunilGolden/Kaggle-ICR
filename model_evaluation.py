import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier

from utils import reset_random, random_under_sampler, process_dataframe, fit_and_predict, impute_null_values, balanced_log_loss
import json


def main():
    with open('config.json') as config_file:
        config = json.load(config_file)

    TRAIN_PATH = config['train_path']
    TEST_PATH = config['test_path']
    K_FOLD = config['k_fold']
    RANDOM_SEED = config['random_seed']


    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Perform Random Undersampling
    train = random_under_sampler(train, random_seed=RANDOM_SEED)

    train = impute_null_values(train, method='zero')
    test = impute_null_values(test, method='zero')

    processed_train, processed_test = process_dataframe(train.drop('Class', axis=1), test)

    # Create an instance of KFold
    reset_random(RANDOM_SEED)
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_SEED)

    tabpfn_valid_accuracies = []
    xgb_valid_accuracies = []
    cb_valid_accuracies = []
    hgb_valid_accuracies = []
    rf_valid_accuracies = []
    lgbm_valid_accuracies = []
    ada_valid_accuracies = []
    gbm_valid_accuracies = []
    svm_valid_accuracies = []

    tabpfn_losses = []
    xgb_losses = []
    cb_losses = []
    hgb_losses = []
    rf_losses = []
    lgbm_losses = []
    ada_losses = []
    gbm_losses = []
    svm_losses = []
    ensemble_losses = []
    ensemble_losses_2 = []
    ensemble_losses_3 = []
    ensemble_losses_4 = []
    ensemble_losses_5 = []
    ensemble_losses_6 = []

    feature_importances = []

    fold = 0

    for train_index, val_index in kf.split(processed_train):
        # Split the data into training and validation sets
        X_train, X_val = processed_train.iloc[train_index], processed_train.iloc[val_index]
        y_train, y_val = train['Class'].iloc[train_index], train['Class'].iloc[val_index]
        
        print('Fold: ', fold)
        
        reset_random(RANDOM_SEED)
        tabpfn_clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=12, seed=RANDOM_SEED)
        tabpfn_clf.fit(X_train, y_train)
        y_eval, p_eval = tabpfn_clf.predict(X_val, return_winning_probability=True)
        tabpfn_accuracy = accuracy_score(y_val, y_eval)
        print('TabPFN:\t\t', tabpfn_accuracy)
        tabpfn_valid_accuracies.append(tabpfn_accuracy)
        tabpfn_pred = np.column_stack([(1 - p_eval), p_eval])    
        tabpfn_loss = balanced_log_loss(y_val, tabpfn_pred)
        print('Loss:\t\t', tabpfn_loss)
        print('------------------------------------')
        tabpfn_losses.append(tabpfn_loss)

        reset_random(RANDOM_SEED)
        xgb_clf = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.15, alpha=0.32, tree_method='auto', seed=RANDOM_SEED)
        xgb_pred, xgb_accuracy = fit_and_predict(xgb_clf, X_train, y_train, X_val, y_val)
        xgb_valid_accuracies.append(xgb_accuracy)
        print('XGBoost:\t', xgb_accuracy)
        xgb_loss = balanced_log_loss(y_val, xgb_pred)
        print('Loss:\t\t', xgb_loss)
        print('------------------------------------')
        xgb_losses.append(xgb_loss)
        
        reset_random(RANDOM_SEED)
        cb_clf = CatBoostClassifier(loss_function='Logloss', boosting_type='Ordered', n_estimators=1000, learning_rate=0.03, l2_leaf_reg=5.0, random_state=RANDOM_SEED, verbose=0)
        cb_pred, cb_accuracy = fit_and_predict(cb_clf, X_train, y_train, X_val, y_val)
        cb_valid_accuracies.append(cb_accuracy)
        print('CatBoost:\t', cb_accuracy)
        cb_loss = balanced_log_loss(y_val, cb_pred)
        print('Loss:\t\t', cb_loss)
        print('------------------------------------')
        cb_losses.append(cb_loss)
        
        reset_random(RANDOM_SEED)
        hgb_clf = HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.055, l2_regularization=0.1, random_state=RANDOM_SEED)
        hgb_pred, hgb_accuracy = fit_and_predict(hgb_clf, X_train, y_train, X_val, y_val)
        hgb_valid_accuracies.append(hgb_accuracy)
        print('HGBoost:\t', hgb_accuracy)
        hgb_loss = balanced_log_loss(y_val, hgb_pred)
        print('Loss:\t\t', hgb_loss)
        print('------------------------------------')
        hgb_losses.append(hgb_loss)
        
        reset_random(RANDOM_SEED)
        lgbm_clf = LGBMClassifier(boosting_type='dart', learning_rate=0.13, n_estimators=100, reg_alpha=0.0005, objective='binary', random_state=RANDOM_SEED)
        lgbm_pred, lgbm_accuracy = fit_and_predict(lgbm_clf, X_train, y_train, X_val, y_val)
        lgbm_valid_accuracies.append(lgbm_accuracy)
        print('Light GBM:\t', lgbm_accuracy)
        lgbm_loss = balanced_log_loss(y_val, lgbm_pred)
        print('Loss:\t\t', lgbm_loss)
        print('------------------------------------')
        lgbm_losses.append(lgbm_loss)

        reset_random(RANDOM_SEED)
        rf_clf = RandomForestClassifier(criterion="log_loss", random_state=RANDOM_SEED)
        rf_pred, rf_accuracy = fit_and_predict(rf_clf, X_train, y_train, X_val, y_val)
        rf_valid_accuracies.append(rf_accuracy)
        print('RandomForest:\t', rf_accuracy)
        rf_loss = balanced_log_loss(y_val, rf_pred)
        print('Loss:\t\t', rf_loss)
        print('------------------------------------')
        rf_losses.append(rf_loss)
        
        reset_random(RANDOM_SEED)
        ada_clf = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=RANDOM_SEED)
        ada_pred, ada_accuracy = fit_and_predict(ada_clf, X_train, y_train, X_val, y_val)
        ada_valid_accuracies.append(ada_accuracy)
        print('AdaBoost:\t', ada_accuracy)
        ada_loss = balanced_log_loss(y_val, ada_pred)
        print('Loss:\t\t', ada_loss)
        print('------------------------------------')
        ada_losses.append(ada_loss)

        reset_random(RANDOM_SEED)
        gbm_clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
        gbm_pred, gbm_accuracy = fit_and_predict(gbm_clf, X_train, y_train, X_val, y_val)
        gbm_valid_accuracies.append(gbm_accuracy)
        print('GBM:\t\t', gbm_accuracy)
        gbm_loss = balanced_log_loss(y_val, gbm_pred)
        print('Loss:\t\t', gbm_loss)
        print('------------------------------------')
        gbm_losses.append(gbm_loss)
        
        reset_random(RANDOM_SEED)
        svm_clf = SVC(probability=True, kernel='poly', degree=3, C=1.25, coef0=4.2, random_state=RANDOM_SEED)
        svm_pred, svm_accuracy = fit_and_predict(svm_clf, X_train, y_train, X_val, y_val)
        svm_valid_accuracies.append(svm_accuracy)
        print('SVM:\t\t', svm_accuracy)
        svm_loss = balanced_log_loss(y_val, svm_pred)
        print('Loss:\t\t', svm_loss)
        print('------------------------------------')
        svm_losses.append(svm_loss)
        
        # Ensemble
        print('\nEnsemble')
        ensemble = (np.array(xgb_pred) + np.array(lgbm_pred)) / 2
        ensemble_loss = balanced_log_loss(y_val, ensemble)
        print('Loss:\t\t', ensemble_loss)
        print('------------------------------------')
        ensemble_losses.append(ensemble_loss)
    
        print('Ensemble 2')
        ensemble_2 = (np.array(xgb_pred) + np.array(cb_pred) + np.array(hgb_pred) + np.array(lgbm_pred)) / 4
        ensemble_loss_2 = balanced_log_loss(y_val, ensemble_2)
        print('Loss:\t\t', ensemble_loss_2)
        print('------------------------------------')
        ensemble_losses_2.append(ensemble_loss_2)

        print('Ensemble 3')
        ensemble_3 = (3*np.array(xgb_pred) + 3*np.array(cb_pred) + 2*np.array(hgb_pred) + 2*np.array(lgbm_pred) + np.array(rf_pred) + np.array(ada_pred) + np.array(gbm_pred) + np.array(svm_pred)) / 14
        ensemble_loss_3 = balanced_log_loss(y_val, ensemble_3)
        print('Loss:\t\t', ensemble_loss_3)
        print('------------------------------------')
        ensemble_losses_3.append(ensemble_loss_3)
        
        print('Ensemble 4')
        ensemble_4 = (tabpfn_pred + 3*np.array(xgb_pred) + 3*np.array(cb_pred) + 2*np.array(hgb_pred) + 2*np.array(lgbm_pred) + np.array(rf_pred) + np.array(ada_pred) + np.array(gbm_pred) + np.array(svm_pred)) / 15
        ensemble_loss_4 = balanced_log_loss(y_val, ensemble_4)
        print('Loss:\t\t', ensemble_loss_4)
        print('------------------------------------')
        ensemble_losses_4.append(ensemble_loss_4)
        
        print('Ensemble 5')
        ensemble_5 = (np.array(xgb_pred) + np.array(cb_pred) + np.array(hgb_pred) + np.array(lgbm_pred) + np.array(rf_pred) + np.array(gbm_pred)) / 6
        ensemble_loss_5 = balanced_log_loss(y_val, ensemble_5)
        print('Loss:\t\t', ensemble_loss_5)
        print('------------------------------------')
        ensemble_losses_5.append(ensemble_loss_5)
        
        print('Ensemble 6')
        ensemble_6 = (tabpfn_pred + np.array(xgb_pred) + np.array(cb_pred) + np.array(hgb_pred) + np.array(lgbm_pred) + np.array(rf_pred) + np.array(ada_pred) + np.array(gbm_pred) + np.array(svm_pred)) / 9
        ensemble_loss_6 = balanced_log_loss(y_val, ensemble_6)
        print('Loss:\t\t', ensemble_loss_6)
        print('------------------------------------')
        ensemble_losses_6.append(ensemble_loss_6)
        
        feature_importances.append(xgb_clf.feature_importances_)
        feature_importances.append(cb_clf.feature_importances_)
        feature_importances.append(lgbm_clf.feature_importances_)
        feature_importances.append(rf_clf.feature_importances_)
        feature_importances.append(ada_clf.feature_importances_)
        feature_importances.append(gbm_clf.feature_importances_)

        print()
        fold += 1
        
        
    # Print the average validation accuracy
    print("\nAverage Validation Accuracy:\n")
    print('TabPFN:\t\t\t' ,sum(tabpfn_valid_accuracies) / len(tabpfn_valid_accuracies))
    print('XGBoost:\t\t' ,sum(xgb_valid_accuracies) / len(xgb_valid_accuracies))
    print('CatBoost:\t\t' ,sum(cb_valid_accuracies) / len(cb_valid_accuracies))
    print('HGBoost:\t\t' ,sum(hgb_valid_accuracies) / len(hgb_valid_accuracies))
    print('Light GBM:\t\t' ,sum(lgbm_valid_accuracies) / len(lgbm_valid_accuracies))
    print('RandomForest:\t\t' ,sum(rf_valid_accuracies) / len(rf_valid_accuracies))
    print('AdaBoost:\t\t', sum(ada_valid_accuracies) / len(ada_valid_accuracies))
    print('GBM:\t\t\t', sum(gbm_valid_accuracies) / len(gbm_valid_accuracies))
    print('SVM:\t\t\t', sum(svm_valid_accuracies) / len(svm_valid_accuracies))

    print("\n\nAverage Losses:\n")
    print('TabPFN:\t\t\t' ,sum(tabpfn_losses) / len(tabpfn_losses))
    print('XGBoost:\t\t' ,sum(xgb_losses) / len(xgb_losses))
    print('CatBoost:\t\t' ,sum(cb_losses) / len(cb_losses))
    print('HGBoost:\t\t' ,sum(hgb_losses) / len(hgb_losses))
    print('Light GBM:\t\t' ,sum(lgbm_losses) / len(lgbm_losses))
    print('RandomForest:\t\t' ,sum(rf_losses) / len(rf_losses))
    print('AdaBoost:\t\t', sum(ada_losses) / len(ada_losses))
    print('GBM:\t\t\t', sum(gbm_losses) / len(gbm_losses))
    print('SVM:\t\t\t', sum(svm_losses) / len(svm_losses))
    print('Ensemble 1:\t\t', sum(ensemble_losses) / len(ensemble_losses))
    print('Ensemble 2:\t\t', sum(ensemble_losses_2) / len(ensemble_losses_2))
    print('Ensemble 3:\t\t', sum(ensemble_losses_3) / len(ensemble_losses_3))
    print('Ensemble 4:\t\t', sum(ensemble_losses_4) / len(ensemble_losses_4))
    print('Ensemble 5:\t\t', sum(ensemble_losses_5) / len(ensemble_losses_5))
    print('Ensemble 6:\t\t', sum(ensemble_losses_6) / len(ensemble_losses_6))

    # Calculate the mean feature importance across all folds for each model
    mean_feature_importance = np.mean(feature_importances, axis=0)

    # Create a dictionary to map each feature to its importance score
    feature_importance_dict = dict(zip(processed_train.columns, mean_feature_importance))

    # Sort the dictionary based on feature importance values in descending order
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    # Print the sorted feature importance
    print("\n\nFeature Importance:\n")
    for feature, importance in sorted_feature_importance.items():
        print(f"{feature}: {importance}")


if __name__ == '__main__':
	main()