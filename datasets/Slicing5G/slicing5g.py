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

        enc = OneHotEncoder(handle_unknown='error')
        le = LabelEncoder()


        Path(output).mkdir(parents=True, exist_ok=True)

        del df["Unnamed: 0"]
        df.dropna(axis = 0, inplace = True)
        df["LTE/5G UE Category (Input 2)"] = df["LTE/5G UE Category (Input 2)"].astype(str)
        df["Slice Type (Output)"] = le.fit_transform(df["Slice Type (Output)"])

        #Transform features into one hot vectors
        encoded_features = df.drop("Time (Input 5)", axis=1)
        encoded_features = encoded_features.drop("Slice Type (Output)", axis=1)
        
        enc.fit(encoded_features)
        data = enc.transform(encoded_features).toarray()
        data = np.append(data, df["Time (Input 5)"].to_numpy().reshape(-1,1), axis=1)
        data = np.append(data, df["Slice Type (Output)"].to_numpy().reshape(-1,1), axis=1)

        X = data[:,:-1]
        y = data[:,-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        
        print(f"\nTotal samples {df.values.shape[0]}")
        print(f"Shape of the train data: {X_train.shape}")
        print(f"Shape of the test data: {X_test.shape}\n")

        np.savetxt(f"{output}/X_train.csv", X_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/X_test.csv", X_test, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_train.csv", y_train, delimiter=",", fmt="%d")
        np.savetxt(f"{output}/y_test.csv", y_test, delimiter=",", fmt="%d")

    def load_training_data(self):
        """
        Load the training data
        """
        x_train = np.loadtxt(f"datasets/{self.name}/data/X_train.csv", delimiter=",")
        y_train = np.loadtxt(f"datasets/{self.name}/data/y_train.csv", delimiter=",")
        return x_train, y_train

    def load_test_data(self):
        """
        Load the test data
        """
        x_test = np.loadtxt(f"datasets/{self.name}/data/X_test.csv", delimiter=",")
        y_test = np.loadtxt(f"datasets/{self.name}/data/y_test.csv", delimiter=",")
        return x_test, y_test

    def create_model(self):
        model= tf.keras.models.Sequential([
                # flatten layer
                tf.keras.layers.Input(shape=(60,)),
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