import time
import argparse
import pathlib
import json
import warnings
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from config import DATASETS

warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

parser = argparse.ArgumentParser(description='Train/test the DNNs.')
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default=None)
parser.add_argument('-s', type=int, help='Seed used for data shuffle', default=42)
parser.add_argument('-r', type=str, help='Results folder', default='results')
args = parser.parse_args()

def model1(input, output):
    return tf.keras.models.Sequential([
            # flatten layer
            tf.keras.layers.Input(shape=(input,)),
            # hidden layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            # output layer
            tf.keras.layers.Dense(output, activation='softmax')
        ])

def model2(input, output):
    return tf.keras.models.Sequential([
            # flatten layer
            tf.keras.layers.Input(shape=(input,)),
            # hidden layers
            tf.keras.layers.Dense(100, activation='relu'),
            # output layer
            tf.keras.layers.Dense(output, activation='softmax')
        ])

def model3(input, output):
    return tf.keras.models.Sequential([
                # flatten layer
            tf.keras.layers.Input(shape=(input,)),
                # hidden layers
                tf.keras.layers.Dense(73, activation='relu'),
                tf.keras.layers.Dropout(0.5),

                # output layer
                tf.keras.layers.Dense(output, activation="softmax")
            ])

def model4(input, output):
    return tf.keras.models.Sequential([
            # flatten layer
            tf.keras.layers.Input(shape=(input,)),
            # hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            # output layer
            tf.keras.layers.Dense(output, activation='softmax')
        ])

def model5(input, output):
    return tf.keras.models.Sequential([
                # flatten layer
                tf.keras.layers.Input(shape=(input,)),
                # hidden layers
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(3, activation='tanh'),
                # output layer
                tf.keras.layers.Dense(output, activation="softmax")
            ])

def model7(input, output):
    return tf.keras.models.Sequential([
            # input layer
            tf.keras.layers.Input(shape=(input,)),
            # hidden layers
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            # output layer
            tf.keras.layers.Dense(output, activation='softmax')
        ])

def model8(input, output):
    return tf.keras.models.Sequential([
            # input layer
            tf.keras.layers.Input(shape=(input,)),
            # hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(96, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            # output layer
            tf.keras.layers.Dense(output, activation='softmax')
        ])

tf.keras.utils.set_random_seed(args.s)
tf.config.threading.set_intra_op_parallelism_threads(64)
tf.config.threading.set_inter_op_parallelism_threads(64)

results = pathlib.Path(args.r)

if args.d == None:
    datasets = ["QoS_QoE", "IoTID20", "Botnet_IOT"]
else:
    datasets = [args.d]

models = [('IOT_DNL', model1), ('KPI_KQI', model2), ('NetSlice5G', model3), ('RT_IOT', model4),
    ('Slicing5G', model5), ('UNAC', model7), ('UNSW', model8)]

for dataset in datasets:
    models_results = {} 
    d = DATASETS[dataset]()
    dataset_results = results / f"{d.name}/nn_search"
    dataset_results.mkdir(parents=True, exist_ok=True)

    X_train, y_train = d.load_training_data()
    if X_train.shape[0] > 100000:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=100000/X_train.shape[0], stratify=y_train)

    x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.2, stratify=y_train)

    input_shape = x_train.shape[1]
    output = len(np.unique(y_train))

    best_model = "None"
    best_mcc = -1
    best_time = 0
    best_roc_auc_res = 0
    best_f1_score_res = 0
    best_hisotry = {}

    enc = OneHotEncoder(categories=[list(range(output))])
    y_val_enc = enc.fit_transform(y_val).toarray()

    for model_name, model_fn in models:
        
        model = model_fn(input_shape, output)
        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.F1Score(average='macro'), tf.keras.metrics.AUC(multi_label=True)]
                )
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        start = time.time()
        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                             batch_size=1024, epochs=200, verbose=0,
                             callbacks=[callback])
        end = time.time()-start

        with open(dataset_results/f"{model_name}_training_history.json", "w") as f:
            json.dump(history.history, f, indent=2)

        model.save(dataset_results/f"{model_name}_model.keras")

        
        y_pred = model.predict(x_val, verbose=0)
        y_pred = [np.argmax(y) for y in y_pred]
        y_pred_enc = enc.transform(np.array(y_pred).reshape(-1, 1)).toarray()

        mcc = matthews_corrcoef(y_val, y_pred)
        f1_score_res = f1_score(y_val, y_pred, average="weighted")
        roc_auc_res = roc_auc_score(y_val_enc, y_pred_enc, average="macro", multi_class='ovr')

        models_results[model_name] = {"MCC": mcc, "F1-Score": f1_score_res, "ROC-AUC" : roc_auc_res, "Training Time" : round(end,2)} 

        if mcc > best_mcc or ( mcc == best_mcc and best_time > end):
            best_model = model_name
            best_mcc = mcc
            best_time = end
            best_f1_score_res = f1_score_res
            best_roc_auc_res = roc_auc_res
            best_hisotry = history.history

    print(f"Best model: {best_model}")
    print(f"Best mcc: {best_mcc*100:.2f}")
    print(f"Best training time: {best_time:.2f}")
    results_file = open(dataset_results/"results.txt", "w")
    results_file.write(f"Best model: {best_model}\n")
    results_file.write(f"Best mcc: {best_mcc*100:.2f}\n")
    results_file.write(f"Best f1-score: {best_f1_score_res*100:.2f}\n")
    results_file.write(f"Best auc_roc: {best_roc_auc_res*100:.2f}\n")
    results_file.write(f"Best training time: {best_time:.2f}\n")

    results_file.write("\n\n Results Table \n\n")
    text = '| Metric | '
    for model in models_results:
        text += f" {model} |"
    results_file.write(text +' \n')

    for metric in ["MCC", "F1-Score", "ROC-AUC", "Training Time"]:
        text = f"| {metric} |"
        for model in models_results:
            text += f" {models_results[model][metric]:.2f} |"
        text += "\n"
        results_file.write(text)

    results_file.close()


    
