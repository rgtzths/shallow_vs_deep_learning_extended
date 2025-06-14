import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf

from Util import Util

class RT_IOT(Util):

    def __init__(self):
        super().__init__("RT_IOT")


    def data_processing(self):
        df = pd.read_csv(f"datasets/{self.name}/data/RT_IOT2022.csv")
        le = LabelEncoder()

        output = f"datasets/{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)
        
        df.dropna(axis = 0, inplace = True)

        X = df.drop(columns=["Attack_type", "Unnamed: 0"])
        y = df["Attack_type"]
        X["service"] = le.fit_transform(X["service"])
        X["proto"] = le.fit_transform(X["proto"])
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


    def create_model(self):
        model =  tf.keras.models.Sequential([
            # flatten layer
            tf.keras.layers.Input(shape=(83,)),
            # hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            # output layer
            tf.keras.layers.Dense(12, activation='softmax')
        ])
        
        return model