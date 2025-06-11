import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf

from Util import Util

class QoS_QoE(Util):

    def __init__(self):
        super().__init__("QoS_QoE")


    def data_processing(self):
        df = pd.read_csv(f"datasets/{self.name}/data/Dataset-QOE_prediction_ICC2018.csv", sep=",", index_col=None)
        le = LabelEncoder()
        output = f"datasets/{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)

        df.dropna(axis = 0, inplace = True)
        X = df.drop(columns=["RebufferingRatio", "AvgVideoBitRate", "AvgVideoQualityVariation", "StallLabel"])
        
        y = df["StallLabel"]

        X["DASHPolicy"] = le.fit_transform(X["DASHPolicy"])
        y = pd.Series(le.fit_transform(y), name='target')

        n_samples=X.shape[0]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        print(f"\nTotal samples {n_samples}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")

        # Save the data
        x_train.to_csv(f"{output}/X_train.csv", index=False)
        x_test.to_csv(f"{output}/X_test.csv", index=False)
        y_train.to_csv(f"{output}/y_train.csv", index=False)
        y_test.to_csv(f"{output}/y_test.csv", index=False)


    def create_model(self): #NetSlice model
        model = tf.keras.models.Sequential([
                # flatten layer
                tf.keras.layers.Input(shape=(47,)),
                # hidden layers
                tf.keras.layers.Dense(73, activation='relu'),
                tf.keras.layers.Dropout(0.5),

                # output layer
                tf.keras.layers.Dense(3, activation='softmax')
            ])

        return model