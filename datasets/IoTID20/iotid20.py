from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

from Util import Util

class IoTID20(Util):

    def __init__(self):
        super().__init__("IoTID20")

    def data_processing(self):
        dataset = f"datasets/{self.name}/data/IoT Network Intrusion Dataset.csv"
        output = f"datasets/{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)
        le = LabelEncoder()

        data = pd.read_csv(dataset)
        data.dropna(axis = 0, inplace = True)

        X = data.drop(columns=['Label', 'Cat', 'Sub_Cat', "Timestamp", "Active_Max", "Bwd_IAT_Max", "Bwd_Seg_Size_Avg", "Fwd_IAT_Max", "Fwd_Seg_Size_Avg", "Idle_Max", "PSH_Flag_Cnt",
                                "Pkt_Size_Avg", "Subflow_Bwd_Byts", "Subflow_Bwd_Pkts", "Subflow_Fwd_Byts", "Subflow_Fwd_Pkts", "Src_IP", "Dst_IP", "Flow_Duration", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
                                "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts", "Fwd_Pkt_Len_Std", "Bwd_Pkt_Len_Std", "Flow_Byts/s", "Flow_Pkts/s", "Flow_IAT_Mean", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", 
                                "Fwd_IAT_Tot", "Fwd_IAT_Mean", "Bwd_IAT_Mean", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Tot", "Bwd_IAT_Mean", "Bwd_IAT_Std", "Bwd_IAT_Max", "Bwd_IAT_Min", "Bwd_PSH_Flags",
                                "Bwd_URG_Flags", "Fwd_Header_Len", "Bwd_Header_Len", "Fwd_Pkts/s", "Bwd_Pkts/s", "Pkt_Len_Std", "Pkt_Len_Var", "FIN_Flag_Cnt", "SYN_Flag_Cnt", "RST_Flag_Cnt", "PSH_Flag_Cnt",
                                "URG_Flag_Cnt", "CWE_Flag_Count", "ECE_Flag_Cnt", "Subflow_Fwd_Pkts", "Subflow_Fwd_Byts", "Subflow_Bwd_Pkts", "Subflow_Bwd_Byts", "Fwd_Act_Data_Pkts", "Active_Mean",
                                "Active_Std", "Active_Max", "Active_Min", "Idle_Mean", "Idle_Std", "Idle_Max", "Idle_Min"])
        
        X["Flow_ID"] = le.fit_transform(X["Flow_ID"])
        
        y = pd.Series(le.fit_transform(data['Sub_Cat']), name='target')
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


    def create_model(self):#UNSW model
        model= tf.keras.models.Sequential([
            # input layer
            tf.keras.layers.Input(input_shape=(27,)),
            # hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(96, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            # output layer
            tf.keras.layers.Dense(9, activation='softmax')
        ])
        model.compile(
                    optimizer=tf.keras.optimizers.Adam(), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )

        return model