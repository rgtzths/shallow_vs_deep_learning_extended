import argparse
import tensorflow as tf
from sklearn.utils import resample
import json

from config import DATASETS, XAI
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default=None)
parser.add_argument("-f", help=f"results folder", default="results")
parser.add_argument("-p", help="Percentage of the test dataset to use", default=10, type=float)
parser.add_argument("--seed", help="Seed used for the run", default=42, type=int)
args = parser.parse_args()

tf.keras.utils.set_random_seed(args.seed)

# all .keras and .h5 files in models folder
datasets = DATASETS.keys() if args.d == None else [args.d]

for dataset in datasets:
    model_path = f"{args.f}/{dataset}/dnn_model.keras"

    dataset_util = DATASETS[dataset]()
    x_train, y_train = dataset_util.load_training_data()
    x_test, y_test = dataset_util.load_test_data()

    if x_test.shape[0] > 100000:
        x_test, y_test = resample(x_test, y_test, n_samples=100000, random_state=args.seed, stratify=y_test)

    print(f"X_test shape: {x_test.shape}")

    model = tf.keras.models.load_model(model_path)
    feature_importance_list = []
    for xai in XAI.keys():
        feature_importance = XAI[xai](x_train, y_train, x_test, y_test, model)

        feature_importance_list.append(list(feature_importance.values()))
        with open(f"{args.f}/{dataset}/{xai}.json", "w") as f: 
            json.dump(feature_importance, f, indent=4)
    indices_to_pop = []
    for i in range(len(feature_importance_list[0])):
        if str(feature_importance_list[0][i]) == "nan" or str(feature_importance_list[1][i]) == "nan":
            indices_to_pop.append(i)
    for i in sorted(indices_to_pop, reverse=True):
        feature_importance_list[0].pop(i)
        feature_importance_list[1].pop(i)

    correlation = pearsonr(feature_importance_list[0], feature_importance_list[1])[0]

    print(correlation)
    with open(f"{args.f}/{dataset}/correlation.txt", "w") as f: 
            f.write(f"Correlation: {correlation}\n")
    
