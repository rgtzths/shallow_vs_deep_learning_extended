import pathlib

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import joblib

import sys
sys.path.append('.')

from config import DATASETS


features = ['Use CaseType (Input 1)', 'LTE/5G UE Category (Input 2)',
       'Technology Supported (Input 3)', 'Day (Input4)', 'Time (Input 5)',
       'QCI (Input 6)', 'Packet Loss Rate (Reliability)',
       'Packet Delay Budget (Latency)']
labels = ['URLLC', 'eMBB', 'mMTC']
dataset_name = "Slicing5G"
results_dir =  pathlib.Path(f"results/{dataset_name}")

clf = joblib.load(results_dir/'DT.joblib')
plt.figure(figsize=(7, 7))
plot_tree(clf, feature_names=features, class_names=labels, impurity=False, fontsize=7)
plt.savefig(results_dir/f"{dataset_name}_decision_tree.pdf", bbox_inches='tight')  # Save as PDF
plt.close()
