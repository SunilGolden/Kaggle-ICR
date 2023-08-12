import pandas as pd
import numpy as np

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

from utils import reset_random, random_under_sampler, process_dataframe, impute_null_values
import json


def main():
    with open('config.json') as config_file:
        config = json.load(config_file)

    TRAIN_PATH = config['train_path']
    TEST_PATH = config['test_path']
    RANDOM_SEED = config['random_seed']

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    # Perform Random Undersampling
    train = random_under_sampler(train, random_seed=RANDOM_SEED)
    
    train = impute_null_values(train, method='zero')
    test = impute_null_values(test, method='zero')
    
    processed_train, processed_test = process_dataframe(train.drop('Class', axis=1), test)

    xgb_proba = []
    cb_proba = []
    hgb_proba = []
    lgbm_proba = []
    rf_proba = []
    gbm_proba = []

    print('Started Training ...')

    reset_random(RANDOM_SEED)
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.15, alpha=0.32, tree_method='auto', seed=RANDOM_SEED)
    xgb_clf.fit(processed_train, train['Class'])
    xgb_proba.append(xgb_clf.predict_proba(processed_test))

    reset_random(RANDOM_SEED)
    cb_clf = CatBoostClassifier(loss_function='Logloss', boosting_type='Ordered', n_estimators=1000, learning_rate=0.03, l2_leaf_reg=5.0, random_state=RANDOM_SEED, verbose=0)
    cb_clf.fit(processed_train, train['Class'])
    cb_proba.append(cb_clf.predict_proba(processed_test))

    reset_random(RANDOM_SEED)
    hgb_clf = HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.055, l2_regularization=0.1, random_state=RANDOM_SEED)
    hgb_clf.fit(processed_train, train['Class'])
    hgb_proba.append(hgb_clf.predict_proba(processed_test))

    reset_random(RANDOM_SEED)
    lgbm_clf = LGBMClassifier(boosting_type='dart', learning_rate=0.13, n_estimators=100, reg_alpha=0.0005, objective='binary', random_state=RANDOM_SEED)
    lgbm_clf.fit(processed_train, train['Class'])
    lgbm_proba.append(lgbm_clf.predict_proba(processed_test))

    reset_random(RANDOM_SEED)
    rf_clf = RandomForestClassifier(criterion="log_loss", random_state=RANDOM_SEED)
    rf_clf.fit(processed_train, train['Class'])
    rf_proba.append(rf_clf.predict_proba(processed_test))
        
    reset_random(RANDOM_SEED)
    gbm_clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    gbm_clf.fit(processed_train, train['Class'])
    gbm_proba.append(gbm_clf.predict_proba(processed_test))

    print('Training and Inference Complete')

    # Ensemble
    y_pred_proba = (np.array(xgb_proba) + np.array(cb_proba) + np.array(hgb_proba) + np.array(lgbm_proba) + np.array(rf_proba) + np.array(gbm_proba)) / 6

    results = pd.DataFrame(data=y_pred_proba[0], columns=['class_0', 'class_1'])
    results['Id'] = test['Id']
    results = results[['Id', 'class_0', 'class_1']]

    # Save Results
    results.to_csv('submission.csv', index=False)

    print('Results saved as submission.csv')


if __name__ == '__main__':
	main()