import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.utils import resample


from Util import Util

class UNAC(Util):

    def __init__(self):
        super().__init__("UNAC")


    def data_processing(self):
        dataset = f"datasets/{self.name}/data/output_full.csv"
        output = f"datasets/{self.name}/data"

        Path(output).mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(dataset, sep=";")
        data.dropna(axis = 0, inplace = True)
        y = data['output'] -1
        x = data.drop(columns=["output", "file"])

        n_samples = x.shape[0]

        scaler = StandardScaler()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

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
        model=  tf.keras.models.Sequential([
            # input layer
            tf.keras.layers.Input(shape=(21,)),
            # hidden layers
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            # output layer
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        return model