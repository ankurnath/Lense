import networkx as nx
import random
import numpy as np
from functions import cut_value, max_cut_heuristic, make_graph_features_for_encoder
import time
import pickle
import sys
import getopt
import os
from argparse import ArgumentParser
from util import *


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Budget"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='Facebook',
        help="Dataset"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    BUDGET = args.budget
    # graph_name = "wiki_test"
    graph_name = args.dataset

    file_path=f'../../data/train/{graph_name}'
    graph= load_from_pickle(file_path)


    # args = sys.argv[1:]
    # opts, args = getopt.getopt(args, "g:b:")
    # for opt, arg in opts:
    #     if opt in ['-g']:
    #         graph_name = arg
    #     elif opt in ['-b']:
    #         BUDGET = int(arg)
    print(graph_name)

    # graph = nx.read_gpickle(f"{graph_name}/main")
    file_path=f'../../data/train/{graph_name}'
    graph= load_from_pickle(file_path)
    start = time.time()
    good_seeds = max_cut_heuristic(graph, BUDGET)
    best_score = cut_value(graph, good_seeds)
    end = time.time()
    print(f"It took {(end - start) / 60:.3f} minutes\n")

    root_folder='../../data/LeNSE/MaxCut/train'
    os.makedirs(root_folder,exist_ok=True)


    save_folder=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/")
    os.makedirs(save_folder,exist_ok=True)

    # graph_features = make_graph_features_for_encoder(graph, graph_name)


    # if not os.path.isdir(f"{graph_name}/budget_{BUDGET}/"):
    #     os.mkdir(f"{graph_name}/budget_{BUDGET}/")

    file_path=os.path.join(save_folder,'score_and_seeds')
    save_to_pickle(data=(good_seeds, best_score),file_path=file_path)

    # with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
    #     pickle.dump((good_seeds, best_score), f)

    # with open(f"{graph_name}/budget_{BUDGET}/time_taken_to_get_seeds", mode='w') as f:
    #     f.write(f"It took {(end - start) / 60} minutes to get a solution.")
