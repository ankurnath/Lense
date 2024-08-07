import networkx as nx
import random
import numpy as np
from functions import make_graph_features_for_encoder, close_pool, get_fixed_size_subgraphs, cover, greedy_mvc
import time
import pickle
import sys
import getopt
import glob
import os
from util import *
from argparse import ArgumentParser



if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument( "--seed", type=int, default=1, help="Seed" ) 
    parser.add_argument( "--num_samples", type=int, default=100,required=True, help="Number of samples" ) 
    parser.add_argument( "--num_checkpoints", type=int, default=1, help="Number of checkpoints" ) 
    parser.add_argument( "--budget", type=int, default=100,required=True, help="Budget" ) 
    parser.add_argument( "--fixed_size", type=int, default=300,required=True, help="Fixed size " ) 
    parser.add_argument( "--dataset", type=str, default='Facebook',required=True, help="Dataset" )

    

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    NUM_SAMPLES=args.num_samples
    NUM_CHECKPOINTS=args.num_checkpoints
    BUDGET= args.budget
    FIXED_SIZE=args.fixed_size
    graph_name=args.dataset
    
    print('Graph',graph_name)

    file_path=f'../../data/train/{graph_name}'
    graph= load_from_pickle(file_path)

    root_folder='../../data/LeNSE/MVC/train'
    good_seeds, best_score = load_from_pickle(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/score_and_seeds"))
    
    

    graph_features = make_graph_features_for_encoder(graph, graph_name)
    N_PER_LOOP = NUM_SAMPLES // NUM_CHECKPOINTS
    count = 0
    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 1)
        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)
        
        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 2)
        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)

        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 3)
        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)

        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 4)
        file_path=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_{count}")
        save_to_pickle(subgraphs,file_path)
        del subgraphs

    subgraphs = []

    
    for fname in glob.glob(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_*")):
        with open(fname, mode="rb") as f:
            hold = pickle.load(f)
            subgraphs += hold


    with open(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/graph_data"), mode="wb") as f:
        pickle.dump(subgraphs, f)

    for fname in glob.glob(os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/data_*")):
        os.remove(fname)
