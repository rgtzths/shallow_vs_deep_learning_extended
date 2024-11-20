import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf

from Util import Util

class Slicing5G(Util):

    def __init__(self):
        super().__init__("Slicing5G")


    def data_processing(self):
        df = pd.read_excel(f"datasets/{self.name}/data/5G_Dataset_Network_Slicing_CRAWDAD_Shared.xlsx", sheet_name="Model_Inputs_Outputs")
        output = f"datasets/{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)

        le = LabelEncoder()
        del df["Unnamed: 0"]
        #Transform features into categories
        df["Use CaseType (Input 1)"] = le.fit_transform(df["Use CaseType (Input 1)"])
        df["LTE/5G UE Category (Input 2)"] = df["LTE/5G UE Category (Input 2)"].astype(str)
        df["LTE/5G UE Category (Input 2)"] = le.fit_transform(df["LTE/5G UE Category (Input 2)"])
        df["Technology Supported (Input 3)"] = le.fit_transform(df["Technology Supported (Input 3)"])
        df["Day (Input4)"] = le.fit_transform(df["Day (Input4)"])
        df["QCI (Input 6)"] = le.fit_transform(df["QCI (Input 6)"])
        df["Packet Loss Rate (Reliability)"] = le.fit_transform(df["Packet Loss Rate (Reliability)"])
        df["Packet Delay Budget (Latency)"] = le.fit_transform(df["Packet Delay Budget (Latency)"])
        df["Slice Type (Output)"] = le.fit_transform(df["Slice Type (Output)"])

        x = df.drop('Slice Type (Output)', axis=1)
        y = df['Slice Type (Output)']

        n_samples = x.shape[0]

        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
        
        print(f"\nTotal samples {df.values.shape[0]}")
        print(f"Shape of the train data: {X_train.shape}")
        print(f"Shape of the test data: {X_test.shape}\n")

        X_train.to_csv(f"{output}/X_train.csv", index=False)
        X_test.to_csv(f"{output}/X_test.csv", index=False)
        y_train.to_csv(f"{output}/y_train.csv", index=False)
        y_test.to_csv(f"{output}/y_test.csv", index=False)

    def create_model(self):
        model= tf.keras.models.Sequential([
                # flatten layer
                tf.keras.layers.Input(shape=(8,)),
                # hidden layers
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(3, activation='tanh'),
                # output layer
                tf.keras.layers.Dense(3, activation="softmax")
            ])
        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )

        return model