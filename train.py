#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@av.it.pt'
__status__ = 'Development'

import os
import numpy as np
import argparse
import pathlib
import exectimeit.timeit as timeit
from sklearn.utils import shuffle


import joblib
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf


# Set an environment variable
os.environ['OPENBLAS_NUM_THREADS'] = '64'

import warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

from config import DATASETS


model_mapping ={'LOG': LogisticRegression,
                'KNN': KNeighborsClassifier,
                'SVM': SVC,
                'NB': GaussianNB,
                'DT': DecisionTreeClassifier,
                'RF': RandomForestClassifier,
                'ABC': AdaBoostClassifier,
                'GBC': GradientBoostingClassifier}


@timeit.exectime(5)
def fit(cls, X, y, is_sklearn):
    if is_sklearn:
        with joblib.parallel_backend(backend='loky', n_jobs=-1):
            cls.fit(X, y)
            return cls
    else:
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="temp_model.keras",
            monitor='accuracy',
            mode='max',
            save_best_only=True)
        cls = cls()
        cls.fit(X, y, batch_size=64, epochs=200, verbose=0, callbacks=[early_stop_callback, model_checkpoint_callback])
        cls = tf.keras.models.load_model("temp")
        return cls

@timeit.exectime(5)
def predict(cls, X, is_sklearn):
    if is_sklearn:
        with joblib.parallel_backend(backend='loky', n_jobs=-1):
            return cls.predict(X)
    else:
        return cls.predict(X, verbose=0)


def optimize(cls_name, parameters, X_train, y_train, cv=5):
    with joblib.parallel_backend(backend='loky', n_jobs=-1):
        cls = model_mapping[cls_name]()
        grid = GridSearchCV(cls, param_grid=parameters, scoring='f1_weighted', cv=cv, n_jobs=-1, refit=best_score)
        grid.fit(X_train, y_train)
        cls = model_mapping[cls_name](**grid.best_params_)
        return cls

def best_score(cv_results_):
    indices = np.where(cv_results_["mean_test_score"] == np.nanmax(cv_results_["mean_test_score"]))[0]
    if len(indices) > 1:
        indices_2 = np.where(cv_results_["mean_fit_time"] == np.nanmin(cv_results_["mean_fit_time"][indices]))[0]
        indices = list(set(indices_2).intersection(set(indices)))
        if len(indices) > 1:
            indices_2 = np.where(cv_results_["mean_score_time"] == np.nanmin(cv_results_["mean_score_time"][indices]))[0]
            indices = list(set(indices_2).intersection(set(indices)))
            return indices[0]
        return indices[0]
    return indices[0]

def train_models(X_train, y_train, X_test, y_test, model_fn, seed, results_folder):
    results_file = open(results_folder/"results.md", "w")
    models = [#('DNN', {}),
              ('LOG', {'random_state':[seed], 'penalty': ['l1','l2'], 'C': [0.001, 0.01, 0.1, 1, 10], 
                       'solver' :['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}),
              ('KNN', {'weights': ['uniform', 'distance'], 'n_neighbors': [3,5,7,9], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                       'leaf_size': [1, 10, 30, 60, 90, 120]}),
              ('SVM', {'random_state':[seed], 'C': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}),
              ('NB',  {'var_smoothing': np.logspace(0,-9, num=100)}),
              ('DT',  {'random_state':[seed], 'criterion':['gini','entropy'], 'max_depth':[3,5,7,9], 'max_features': ['auto', 'sqrt', 'log2']}),
              ('RF',  {'random_state':[seed], 'n_estimators':[5, 10, 50, 100], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[3,5,7,9]}),
              ('ABC', {'random_state':[seed], 'n_estimators':[5, 10, 50, 100]}),
              ('GBC', {'random_state':[seed], 'n_estimators':[5, 10, 50, 100], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[3,5,7,9]})
              ]

    print('| Model name | Train time | Infer time | ACC | F1  | MCC |')
    print('| ---------- | ---------- | ---------- | --- | --- | --- |')
    results_file.write('| Model name | Train time | Infer time | ACC | F1  | MCC |\n')
    results_file.write('| ---------- | ---------- | ---------- | --- | --- | --- |\n')
    for cls_name, parameters in models:
        is_sklearn = cls_name in model_mapping
        if is_sklearn:
            if len(X_train) > 50000:
                x, y = shuffle(X_train, y_train, random_state=42, n_samples=50000)
            else:
                x, y = X_train, y_train
            cls = optimize(cls_name, parameters, x, y)
        else:
            cls = model_fn
            cls()
        mtt, std_tt , cls = fit(cls, X_train, y_train, is_sklearn)
        mti, std_ti , y_pred = predict(cls, X_test, is_sklearn)
        y_pred = y_pred if is_sklearn else [np.argmax(y) for y in y_pred]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        print(f'| {cls_name:<10} | {round(mtt,4):>6}±{round(std_tt,4):<6} | {round(mti,4):<6}±{round(std_ti,4):<6} | {round(acc,2):<6} | {round(f1,2):<3} | {round(mcc,2):<3} |')
        
        results_file.write(f'| {cls_name:<10} | {round(mtt,4):>6}±{round(std_tt,4):<6} | {round(mti,4):<6}±{round(std_ti,4):<6} | {round(acc,2):<3} | {round(f1,2):<3} | {round(mcc,2):<3} |\n')
        if is_sklearn:
            joblib.dump(cls, results_folder/ f'{cls_name}.joblib')
        else:
            cls.save(results_folder/f"dnn_model")
    results_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train/test the DNNs.')
    parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default=None)
    parser.add_argument('-s', type=int, help='Seed used for data shuffle', default=42)
    parser.add_argument('-r', type=str, help='Results folder', default='results')
    args = parser.parse_args()

    if args.d not in DATASETS.keys() and args.d != None:
        raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())} or None")
    
    tf.keras.utils.set_random_seed(args.s)
    results = pathlib.Path(args.r)
    
    if args.d == None:
        for dataset in DATASETS.keys():
            #if dataset not in ["Slicing5G", "NetworkSlicing5G", "NetSlice5G", "UNSW", "IOT_DNL"]:
            print(f"Running dataset: {dataset}")
            d = DATASETS[dataset]()

            results = results / d.name
            results.mkdir(parents=True, exist_ok=True)

            X_train, y_train = d.load_training_data()
            X_test, y_test = d.load_test_data()

            train_models(X_train, y_train, X_test, y_test, d.create_model, args.s, results)
    else:
        d = DATASETS[args.d]()

        results = results / d.name
        results.mkdir(parents=True, exist_ok=True)

        X_train, y_train = d.load_training_data()

        X_test, y_test = d.load_test_data()

        train_models(X_train, y_train, X_test, y_test, d.create_model, args.s, results)