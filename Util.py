from pathlib import Path
import pandas as pd


class Util:

    def __init__(self, name):
        """
        Initialize the class
        """
        self.name = name

    def data_processing(self):
        """
        Process the raw data
        Save the processed data in the data folder as
        X_train.csv, X_val.csv, X_test.csv, y_train.csv, y_val.csv, y_test.csv
        to use default methods.
        """
        pass

    def load_training_data(self):
        """
        Load the training data
        """
        x_train = pd.read_csv(f"datasets/{self.name}/data/X_train.csv")
        y_train = pd.read_csv(f"datasets/{self.name}/data/y_train.csv")
        return x_train, y_train

    def load_test_data(self):
        """
        Load the test data
        """
        x_test = pd.read_csv(f"datasets/{self.name}/data/X_test.csv")
        y_test = pd.read_csv(f"datasets/{self.name}/data/y_test.csv")
        return x_test, y_test
    
    def create_model(self):
        pass