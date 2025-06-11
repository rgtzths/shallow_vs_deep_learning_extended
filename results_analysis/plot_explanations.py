import argparse
import json
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import matplotlib

from config import DATASETS

font = {'size' : 14}

matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default=None)
parser.add_argument("-f", help="Results folder", default="results", type=str)
args = parser.parse_args()

datasets = DATASETS.keys() if args.d == None else [args.d]

correlations = {}
for dataset in datasets:
    in_file = open(f"{args.f}/{dataset}/correlation.txt", "r")
    correlations[dataset] = float(in_file.read().split(":")[1])

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_ylabel('Correlation', fontsize=20)
ax.set_xlabel('Dataset', fontsize=20)
ax.set(ylim=(-1, 1))
ax.bar(list(correlations.keys()), list(correlations.values()))

fig.savefig(f"{args.f}/barplot_correlations.pdf", bbox_inches='tight', orientation="landscape")