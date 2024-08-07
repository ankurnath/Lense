import networkx as nx
import random
import numpy as np
from functions import cover, greedy_mvc, make_graph_features_for_encoder
import time
import pickle
import sys
import getopt
import os
from util import *

from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument( "--seed", type=int, default=1, help="Seed" ) 
    parser.add_argument( "--budget", type=int, default=100, help="Budget" ) 
    parser.add_argument( "--dataset", type=str,required=True, default='Facebook', help="Dataset" )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    BUDGET = args.budget
    graph_name = args.dataset

    print(graph_name)

    file_path=f'../../data/train/{graph_name}'
    graph= load_from_pickle(file_path)
    all_seeds = set()
    scores = []
    start = time.time()
    good_seeds = greedy_mvc(graph, BUDGET)
    best_score = cover(graph, good_seeds)
    end = time.time()
    print(f"It took {(end - start) / 60:.3f} minutes\n")

    

    root_folder='../../data/LeNSE/MVC/train'
    os.makedirs(root_folder,exist_ok=True)


    save_folder=os.path.join(root_folder,f"{graph_name}/budget_{BUDGET}/")
    os.makedirs(save_folder,exist_ok=True)

    

    file_path=os.path.join(save_folder,'score_and_seeds')
    save_to_pickle(data=(good_seeds, best_score),file_path=file_path)

    