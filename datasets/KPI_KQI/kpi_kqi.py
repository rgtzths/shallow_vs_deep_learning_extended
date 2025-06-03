from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

from Util import Util

class KPI_KQI(Util):

    def __init__(self):
        super().__init__("KPI_KQI")

    def data_processing(self):
        dataset = f"datasets/{self.name}/data/database.csv"
        output = f"datasets/{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)
        le = LabelEncoder()

        data = pd.read_csv(dataset)
        data.dropna(axis = 0, inplace = True)
        print(data)
        X = data.drop(columns=[" Service"])        
        y = data[' Service']
        y = pd.Series(le.fit_transform(y), name='target')
        
        n_samples=X.shape[0]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

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
            tf.keras.layers.Input(shape=(13,)),
            # hidden layers
            tf.keras.layers.Dense(100, activation='relu'),
            # output layer
            tf.keras.layers.Dense(9, activation='softmax')
        ])

        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )

        return model