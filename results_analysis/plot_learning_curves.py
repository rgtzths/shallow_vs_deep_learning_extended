import pathlib
import re
import json
from matplotlib import pyplot as plt

results_dir =  pathlib.Path("results")

for folder in results_dir.glob("*"):
    if folder.is_dir():
        for file in folder.glob(f"*.json"):
            if "dnn" in file.name:
                json_file = json.load(open(file))
                
                plt.plot(json_file['loss'])
                plt.plot(json_file['val_loss'])
                plt.ylabel('Loss', fontsize=16)
                plt.xlabel('Epochs', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(['train', 'validation'], loc='upper right', fontsize=14)
                plt.savefig(folder/f"{folder.name}_learning_curve.pdf",  bbox_inches='tight')
                plt.close()