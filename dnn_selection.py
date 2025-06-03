import time
import argparse
import pathlib
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

from config import DATASETS

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

def model6(input, output):
    return tf.keras.models.Sequential([
            # input layer
            tf.keras.layers.Input(shape=(input,)),
            # hidden layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            # output layer
            tf.keras.layers.Dense(output, activation='softmax')
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
results = pathlib.Path(args.r)
tf.config.threading.set_intra_op_parallelism_threads(128)
tf.config.threading.set_inter_op_parallelism_threads(128)

if args.d == None:
    datasets = ["QoS_QoE", "IoTID20", "Botnet_IOT"]
else:
    datasets = [args.d]

models = [('IOT_DNL', model1), ('KPI_KQI', model2), ('NetSlice5G', model3), ('RT_IOT', model4),
    ('Slicing5G', model5), ('TON_IOT', model6), ('UNAC', model7), ('UNSW', model8)]

for dataset in datasets:
    d = DATASETS[dataset]()
    dataset_results = results / f"{d.name}/nn_search"
    dataset_results.mkdir(parents=True, exist_ok=True)

    results_file = open(dataset_results/"results.txt", "w")
    model_history_file =  open(dataset_results/"training_history.json", "w")
    X_train, y_train = d.load_training_data()
    x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.2)

    input_shape = x_train.shape[1]
    output = len(np.unique(y_train))

    best_model = "None"
    best_mcc = -1
    best_time = 0
    best_roc_auc_res = 0
    best_f1_score_res = 0
    best_hisotry = {}
    for model_name, model_fn in models:
        model = model_fn(input_shape, output)
        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'] # adicionar ROC_AUC e f1_score
                )
        callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)

        start = time.time()
        history = model.fit(x_train, y_train, #Adicionar val_dataset
                             batch_size=1024, epochs=1, verbose=0,
                             callbacks=[callback])
        end = time.time()-start
        y_pred = model.predict(x_val, verbose=0)
        y_pred = [np.argmax(y) for y in y_pred]

        mcc = matthews_corrcoef(y_val, y_pred)
        f1_score_res = f1_score(y_val, y_pred, average="weighted")
        roc_auc_res = 0#roc_auc_score(y_val, y_pred, multi_class='ovr')

        if mcc > best_mcc or ( mcc == best_mcc and best_time > end):
            best_model = model_name
            best_mcc = mcc
            best_time = end
            best_f1_score_res = f1_score_res
            best_roc_auc_res = roc_auc_res
            best_hisotry = history.history

    print(f"Best model: {model_name}")
    print(f"Best mcc: {best_mcc:.2f}")
    print(f"Best training time: {best_time:.2f}")
    results_file.write(f"Best model: {model_name}\n")
    results_file.write(f"Best mcc: {best_mcc:.2f}\n")
    results_file.write(f"Best f1-score: {best_f1_score_res:.2f}\n")
    results_file.write(f"Best auc_roc: {best_roc_auc_res:.2f}\n")
    results_file.write(f"Best training time: {best_time:.2f}\n")
    results_file.close()
    json.dump(best_hisotry, model_history_file)