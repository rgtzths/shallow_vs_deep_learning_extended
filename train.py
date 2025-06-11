#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@av.it.pt'
__status__ = 'Development'

import os
import argparse
import pathlib
import joblib
import warnings
from pickle import dump
import json
import psutil

import numpy as np
import exectimeit.timeit as timeit
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

from energy_monitor import calculate_energy

from config import DATASETS

tf.config.threading.set_intra_op_parallelism_threads(64)
tf.config.threading.set_inter_op_parallelism_threads(64)
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

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
        psutil.cpu_percent(interval=None, percpu=False)
        with joblib.parallel_backend(backend='loky', n_jobs=-1):
            cls = cls[0](**cls[1])
            cls.fit(X, y)
            utilization = psutil.cpu_percent(interval=None, percpu=False)/100
            return cls, utilization
    else:
        psutil.cpu_percent(interval=None, percpu=False)
        model = cls[0]()
        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                )
                    
        history = model.fit(X, y, batch_size=1024, epochs=200, verbose=0, validation_split=0.2, callbacks=[cls[1], cls[2]])
        model = tf.keras.models.load_model("temp_model.keras")
        utilization = psutil.cpu_percent(interval=None, percpu=False)/100
        return model, utilization, history
    
@timeit.exectime(5)
def predict(cls, X, is_sklearn):
    if is_sklearn:
        psutil.cpu_percent(interval=None, percpu=False)
        with joblib.parallel_backend(backend='loky', n_jobs=-1):
            prediction = cls.predict(X)
            utilization = psutil.cpu_percent(interval=None, percpu=False)/100
            return prediction, utilization
    else:
        psutil.cpu_percent(interval=None, percpu=False)
        prediction =  cls.predict(X, verbose=0)
        utilization = psutil.cpu_percent(interval=None, percpu=False)/100
        return prediction, utilization


def optimize(cls_name, parameters, X_train, y_train, cv=5):
    with joblib.parallel_backend(backend='loky', n_jobs=-1):
        cls = model_mapping[cls_name]()
        grid = GridSearchCV(cls, param_grid=parameters, scoring='f1_weighted', cv=cv, n_jobs=-1, refit=best_score)
        grid.fit(X_train, y_train)
        return grid.best_params_, grid

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
    models = [('DNN', {}),
              ('KNN', {'weights': ['uniform', 'distance'], 'n_neighbors': [3,5,7,9], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                     'leaf_size': [1, 10, 30, 60, 90, 120]}),
              ('NB',  {'var_smoothing': np.logspace(0,-9, num=100)}),
              ('DT',  {'random_state':[seed], 'criterion':['gini','entropy'], 'max_depth':[3,5,7,9], 'max_features': ['auto', 'sqrt', 'log2']}),
              ('RF',  {'random_state':[seed], 'n_estimators':[5, 10, 50, 100], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[3,5,7,9]}),
              ('ABC', {'random_state':[seed], 'n_estimators':[5, 10, 50, 100]}),
              ('GBC', {'random_state':[seed], 'n_estimators':[5, 10, 50, 100], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[3,5,7,9]})
              ]

    print('| Model name | Train time | Train Energy | Infer time | Infer Energy | ROC_AUC | F1 | MCC |')
    print('| ---------- | ---------- | ------------ | ---------- | ------------ | ------- | -- | --- |')
    results_file.write('| Model name | Train time | Train Energy | Infer time | Infer Energy | ROC_AUC | F1 | MCC |\n')
    results_file.write('| ---------- | ---------- | ------------ | ---------- | ------------ | ------- | -- | --- |\n')
    enc = OneHotEncoder(handle_unknown='ignore')
    y_test_enc = enc.fit_transform(y_test).toarray()

    for cls_name, parameters in models:
        is_sklearn = cls_name in model_mapping
        if is_sklearn:
            if os.path.exists(results_folder/f"{cls_name}_best_params.json"):
                best_params = json.load(open(results_folder/f"{cls_name}_best_params.json"))
                cls = (model_mapping[cls_name], best_params)
            else:
                if len(X_train) > 50000:
                    x, _, y, _ = train_test_split(X_train, y_train, test_size=50000/len(y_train), random_state=42, shuffle=True, stratify=y_train)
                else:
                    x, y = X_train, y_train
                best_params, grid = optimize(cls_name, parameters, x, y)

                cls = (model_mapping[cls_name], best_params)
                        
                joblib.dump(gs, results_folder/f"{cls_name}_grid_search.joblib")
                with open(results_folder/f"{cls_name}_best_params.json", "w") as f:
                    json.dump(best_params, f, indent=2)
        else:
            cls = (model_fn, 
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5), 
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="temp_model.keras",
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)
            )
        m_tt, std_tt, train_data = fit(cls, X_train, y_train, is_sklearn)
        m_tt = abs(m_tt)
        if is_sklearn:
            cls, utilization_train = train_data
        else:
            cls, utilization_train, history = train_data
            with open(results_folder/"dnn_history.json", "w") as f:
                json.dump(history.history, f, indent=2)

        print(utilization_train)
        m_et = calculate_energy(utilization_train, m_tt) if m_tt > 0.1 else 0
        std_et = calculate_energy(utilization_train, std_tt) if std_tt > 0.1 else 0

        m_ti, std_ti , test_data = predict(cls, X_test, is_sklearn)
        m_ti = abs(m_ti)

        y_pred, utilization_test = test_data

        m_ei = calculate_energy(utilization_test, m_ti) if m_ti > 0.1 else 0
        std_ei = calculate_energy(utilization_test, std_ti) if std_ti > 0.1 else 0

        y_pred = y_pred if is_sklearn else [np.argmax(y) for y in y_pred]
        y_pred_enc = enc.transform(np.array(y_pred).reshape(-1, 1)).toarray()

        f1 = f1_score(y_test, y_pred, average='macro')
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test_enc, y_pred_enc, average="macro", multi_class='ovr')

        print(f'| {cls_name:<10} | {round(m_tt,4):>6}±{round(std_tt,4):<6} | {round(m_et,4):<6}±{round(std_et,4):<6} | {round(m_ti,4):<6}±{round(std_ti,4):<6} | {round(m_ei,4):<6}±{round(std_ei,4):<6} | {round(roc_auc,2):<6} | {round(f1,2):<3} | {round(mcc,2):<3} |')
        
        results_file.write(f'| {cls_name:<10} | {round(m_tt,4):>6}±{round(std_tt,4):<6} | {round(m_et,4):<6}±{round(std_et,4):<6} | {round(m_ti,4):<6}±{round(std_ti,4):<6} | {round(m_ei,4):<6}±{round(std_ei,4):<6} | {round(roc_auc,2):<6} | {round(f1,2):<3} | {round(mcc,2):<3} |\n')
        if is_sklearn:
            joblib.dump(cls, results_folder/ f'{cls_name}.joblib')
        else:
            cls.save(results_folder/"dnn_model.keras")
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
            
            print(f"Running dataset: {dataset}")
            d = DATASETS[dataset]()

            results_dir = results / d.name
            results_dir.mkdir(parents=True, exist_ok=True)

            X_train, y_train = d.load_training_data()
            X_test, y_test = d.load_test_data()

            train_models(X_train, y_train, X_test, y_test, d.create_model, args.s, results_dir)
    else:
        d = DATASETS[args.d]()

        results = results / d.name
        results.mkdir(parents=True, exist_ok=True)

        X_train, y_train = d.load_training_data()

        X_test, y_test = d.load_test_data()

        train_models(X_train, y_train, X_test, y_test, d.create_model, args.s, results)