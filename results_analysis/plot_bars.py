import argparse
import json
import pathlib
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
results_dir =  pathlib.Path("results")

datasets = DATASETS.keys() if args.d == None else [args.d]

for dataset in datasets:
    pi_values = json.load(open(results_dir/args.d/"PI.json"))
    pdv_values = json.load(open(results_dir/args.d/"PDV.json"))

    plt.barh(pi_values.keys(), pi_values.values(), color='skyblue')
    plt.xlim(-1, 1)
    plt.savefig(results_dir/args.d/"pi_bar_plot.pdf",  bbox_inches='tight')
    plt.close()
    plt.barh(pdv_values.keys(), pdv_values.values(), color='skyblue')
    plt.xlim(-1, 1)
    plt.savefig(results_dir/args.d/"pdv_bar_plot.pdf",  bbox_inches='tight')
    plt.close()