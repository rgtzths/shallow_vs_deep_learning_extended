import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf

from Util import Util

class NetSlice5G(Util):

    def __init__(self):
        super().__init__("NetSlice5G")


    def data_processing(self):
        df_train = pd.read_csv(f"datasets/{self.name}/data/train_dataset.csv")
        #df_test = pd.read_csv(f"datasets/{self.name}/data/test_dataset.csv")

        output = f"datasets/{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)

        df_train.dropna(axis = 0, inplace = True)

        scaler = StandardScaler()
        
        x_train = df_train.drop(columns=["slice Type"])
        y_train = df_train["slice Type"] -1 

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42, test_size=0.2, stratify=y_train)

        x_train["Time"] = scaler.fit_transform(x_train["Time"].values.reshape(-1,1))
        x_test["Time"] = scaler.transform(x_test["Time"].values.reshape(-1,1))

        x_train["Packet Loss Rate"] = scaler.fit_transform(x_train["Packet Loss Rate"].values.reshape(-1,1))
        x_test["Packet Loss Rate"] = scaler.transform(x_test["Packet Loss Rate"].values.reshape(-1,1))

        x_train["Packet delay"] = scaler.fit_transform(x_train["Packet delay"].values.reshape(-1,1))
        x_test["Packet delay"] = scaler.transform(x_test["Packet delay"].values.reshape(-1,1))


        print(f"\nTotal samples {df_train.values.shape[0]}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")
        
        x_train.to_csv(f"{output}/X_train.csv", index=False)
        x_test.to_csv(f"{output}/X_test.csv", index=False)
        y_train.to_csv(f"{output}/y_train.csv", index=False)
        y_test.to_csv(f"{output}/y_test.csv", index=False)
    
    def create_model(self):
        model = tf.keras.models.Sequential([
                # flatten layer
                tf.keras.layers.Input(shape=(16,)),
                # hidden layers
                tf.keras.layers.Dense(73, activation='relu'),
                tf.keras.layers.Dropout(0.5),

                # output layer
                tf.keras.layers.Dense(3, activation="softmax")
            ])

        return model