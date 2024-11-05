from os import makedirs, path
import time
import json
import pathlib
import pandas as pd
import gc

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from config import DATASETS
import numpy as np
from sklearn.utils import shuffle
import keras_tuner
import keras
from sklearn.model_selection import train_test_split

class DenseModel(keras_tuner.HyperModel):

    def __init__(self, hyperparameters):
        super().__init__()
        self.is_categorical = False
        self.hyperparameters = hyperparameters
        self.is_categorical = True
        
        self.metrics = ['accuracy'] 


    def build(self, hp):
        inputs = keras.Input(shape=(self.hyperparameters["input_shape"]))
        outputs = None

        optimizer = "Adam"
        learning_rate = hp.Float("learning_rate",  min_value=self.hyperparameters["lr"][0],
                                  max_value=self.hyperparameters["lr"][1], 
                                  step=self.hyperparameters["lr"][2])
            
        for i in range(hp.Int("n_dense_layers", min_value=self.hyperparameters["n_dense_layers"][0], 
                              max_value=self.hyperparameters["n_dense_layers"][1], 
                              step=self.hyperparameters["n_dense_layers"][2])):
            
            n_nodes = hp.Int(f"n_dense_nodes_{i}",  min_value=self.hyperparameters["n_dense_nodes"][0], 
                    max_value=self.hyperparameters["n_dense_nodes"][1], 
                    step=self.hyperparameters["n_dense_nodes"][2])
        
            activation = hp.Choice(f"dense_activation_{i}", self.hyperparameters["dense_activation_fn"])

            dropout_rate = hp.Float(f"dense_dropout_rate_{i}", min_value=self.hyperparameters["dense_dropout"][0], 
                                max_value=self.hyperparameters["dense_dropout"][1], 
                                step=self.hyperparameters["dense_dropout"][2])
            if outputs == None:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(inputs)
            else:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(outputs)

            outputs = keras.layers.Dropout(dropout_rate)(outputs)
        
        outputs = keras.layers.Dense(units=self.hyperparameters["n_outputs"], activation="softmax")(outputs)

            
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=self.metrics, 
        )
        return model

    def preprocess_data(self, x, y, validation_data, batch_size):
        ##Add transformations based on the dataset we use
        ##Flatten images, remove time dependencies etc.

        y = keras.utils.to_categorical(y)
        if validation_data:
            x_val, y_val = validation_data
            y_val = keras.utils.to_categorical(y_val)
            validation_data = (x_val, y_val)

        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size), tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)

    #In case we want to optimize anything of the training
    def fit(self, hp, model, x, y, validation_data=None, *args, **kwargs):
        
        train_data, val_data = self.preprocess_data(x, y, validation_data, 64)
        kwargs["callbacks"].append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=10))

        train_time = time.time()

        model.fit(
            x=train_data,
            epochs = self.hyperparameters["epochs"],
            validation_data=val_data,
            verbose = 0,
            shuffle=True,
            **kwargs
        )
        
        train_time = time.time() -  train_time

        infer_time = time.time()
        predictions = model.predict(val_data, verbose=0)
        infer_time = time.time() - infer_time

        predictions = [np.argmax(x) for x in predictions]
        
        results = {'mcc' : matthews_corrcoef(validation_data[1], predictions),
                'acc' : accuracy_score(validation_data[1], predictions), 
                'f1-score' : f1_score(validation_data[1], predictions, average="macro"),
                'train_time' : train_time,
                'infer_time' : infer_time}
        
        del train_data
        del val_data
        del validation_data
        del predictions

        return results
if __name__ == '__main__':
    for dataset in ["NetSlice5G"]:
        params = json.load(open("hyperparameter_values.json"))
        d = DATASETS[dataset]()
        x_train, y_train = d.load_training_data()

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.2)

        params["input_shape"] = x_train.shape[1:]
        params["n_outputs"] = len(np.unique(y_train))
        print(f"Running experiment for dataset:{dataset}")

        '''
        Just to understand the input shape
        '''
        if len(x_train) > 50000:
            x_train, y_train = shuffle(x_train, y_train, random_state=42, n_samples=50000)

        '''
            Results folder creation
        '''
        results_dir = f"results/{dataset}"
        if not path.exists(results_dir):
            makedirs(results_dir)

        '''
            Run search
        '''
        model = DenseModel(params)

        objective = keras_tuner.Objective("mcc", "max") \
                    if model.is_categorical else \
                        keras_tuner.Objective("mse", "min")

        tuner = keras_tuner.RandomSearch(
            model,
            max_trials=params["n_searches"],
            objective=objective,
            executions_per_trial=3,
            overwrite=False,
            directory=results_dir,
            project_name="DenseModel",
            seed=42
        )

        tuner.search_space_summary()

        tuner.search(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
        )

        tuner.results_summary()

        results_dir = pathlib.Path(results_dir) / "DenseModel"

        results_dict = {}

        for res_path in (results_dir).iterdir():
            if res_path.is_dir():
                trial_results = json.load(open(res_path/"trial.json"))
                for key in trial_results["hyperparameters"]["values"]:
                    if key not in results_dict:
                        results_dict[key] = [trial_results["hyperparameters"]["values"][key]]
                    else:
                        results_dict[key].append(trial_results["hyperparameters"]["values"][key])
                for key in trial_results["metrics"]["metrics"]:
                    if key not in results_dict:
                        results_dict[key] = [trial_results["metrics"]["metrics"][key]["observations"][0]["value"][0]]
                    else:
                        results_dict[key].append(trial_results["metrics"]["metrics"][key]["observations"][0]["value"][0])
                for key in results_dict:
                    if key not in trial_results["hyperparameters"]["values"] and key not in trial_results["metrics"]["metrics"]:
                        results_dict[key].append(None)

        df = pd.DataFrame(results_dict)

        df.to_csv(results_dir/"compiled_results.csv", index=None)

        del df
        del tuner
        del results_dict
        del objective

        del x_train
        del y_train
        del x_val
        del y_val

        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
