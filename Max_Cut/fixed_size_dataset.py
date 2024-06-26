import networkx as nx
import random
import numpy as np
from functions import make_graph_features_for_encoder, get_fixed_size_subgraphs, cut_value, max_cut_heuristic
import time
import pickle
import sys
import getopt
import glob
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
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples"
    )
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=1,
        help="Number of checkpoints"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Budget"
    )
    parser.add_argument(
        "--fixed_size",
        type=int,
        default=300,
        help="Fixed size "
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
    NUM_SAMPLES = args.num_samples
    NUM_CHECKPOINTS = args.num_checkpoints
    BUDGET = args.budget
    FIXED_SIZE = args.fixed_size
    graph_name = args.dataset
    # args = sys.argv[1:]
    # opts, args = getopt.getopt(args, "g:n:b:c:f:")
    # for opt, arg in opts:
    #     if opt in ['-g']:
    #         graph_name = arg
    #     elif opt in ['-b']:
    #         BUDGET = int(arg)
    #     elif opt in ['-n']:
    #         NUM_SAMPLES = int(arg)
    #     elif opt in ['-c']:
    #         NUM_CHECKPOINTS = int(arg)
    #     elif opt in ['-f']:
    #         FIXED_SIZE = int(arg)
    print(graph_name)

    file_path=f'../../data/train/{graph_name}'
    graph= load_from_pickle(file_path)

    root_folder='../../data/LeNSE/MaxCut/train'
    good_seeds, best_score = load_from_pickle(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/score_and_seeds"))

    # graph = nx.read_gpickle(f"{graph_name}/main")
    # try:
    #     with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="rb") as f:
    #         good_seeds, best_score = pickle.load(f)

    # except FileNotFoundError:
    #     scores = []
    #     start = time.time()
    #     good_seeds = max_cut_heuristic(graph, BUDGET)
    #     best_score = cut_value(graph, good_seeds)
    #     end = time.time()
    #     print(f"It took {(end - start) / 60:.3f} minutes\n")
    #     if not os.path.isdir(f"{graph_name}/budget_{BUDGET}/"):
    #         os.mkdir(f"{graph_name}/budget_{BUDGET}/")
    #     with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
    #         pickle.dump((good_seeds, best_score), f)

    graph_features = make_graph_features_for_encoder(graph, graph_name)
    N_PER_LOOP = NUM_SAMPLES // NUM_CHECKPOINTS
    count = 0
    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 1)
        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)
        # with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
        #     pickle.dump(subgraphs, f)
        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 2)
        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)
        # with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
        #     pickle.dump(subgraphs, f)
        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 3)
        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)
        # with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
        #     pickle.dump(subgraphs, f)
        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 4)

        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)
        # with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
        #     pickle.dump(subgraphs, f)
        del subgraphs

    subgraphs = []
    for fname in glob.glob(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_*")):
        with open(fname, mode="rb") as f:
            hold = pickle.load(f)
            subgraphs += hold

    # save_to_pickle(subgraphs,os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/graph_data", mode="wb"))

    with open(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/graph_data"), mode="wb") as f:
        pickle.dump(subgraphs, f)

    for fname in glob.glob(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_*")):
        os.remove(fname)
