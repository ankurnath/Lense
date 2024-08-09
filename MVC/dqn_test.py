from environment import TestEnv, BigGraph
import torch
import networkx as nx
import pickle
import random
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from util import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument parser for the model configuration")
    # Add arguments
    parser.add_argument("--seed", type=int,default=1, help="Seed")
    # parser.add_argument('--train_graph_name', type=str, default="youtube_train", help='Name of the training graph')
    # parser.add_argument('--test_graph_name', type=str, default="youtube_test", help='Name of the test graph')
    parser.add_argument('--dataset',type=str,default='Facebook',help='Dataset')
    parser.add_argument('--num_eps', type=int, default=2, help='Number of episodes')
    parser.add_argument('--soln_budget', type=int, default=100, help='Solution budget')
    parser.add_argument('--subgraph_size', type=int, default=300, help='Subgraph size')
    parser.add_argument('--encoder_name', type=str, default="encoder", help='Name of the encoder')
    parser.add_argument('--action_limit', type=int, default=2000, help='Action limit')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # train_graph_name = "youtube_train"
    # test_graph_name = "youtube_test"
    num_eps = args.num_eps
    soln_budget = args.soln_budget
    subgraph_size = args.subgraph_size
    encoder_name = args.encoder_name
    action_limit = args.action_limit
    cuda = args.cuda
    # args = sys.argv[1:]
    # opts, args = getopt.getopt(args, "g:n:b:a:t:f:C:E:")
    # for opt, arg in opts:
    #     if opt in ['-g']:
    #         train_graph_name = arg
    #     elif opt in ["-n"]:
    #         num_eps = int(arg)
    #     elif opt in ["-b"]:
    #         soln_budget = int(arg)
    #     elif opt in ["-a"]:
    #         action_limit = int(arg)
    #     elif opt in ["-t"]:
    #         test_graph_name = arg
    #     elif opt in ["-f"]:
    #         subgraph_size = int(arg)
    #     elif opt in ["-C"]:
    #         cuda = bool(int(arg))
    #     elif opt in ["-E"]:
    #         encoder_name = arg

    # if not os.path.isdir(f"{test_graph_name}/budget_{soln_budget}/{encoder_name}"):
    #     os.mkdir(f"{test_graph_name}/budget_{soln_budget}/{encoder_name}")
    root_folder = '../../data/LeNSE/MVC/train'

    encoder_path = os.path.join(root_folder,f"{args.dataset}/budget_{soln_budget}/{encoder_name}/{encoder_name}")

    encoder = torch.load(encoder_path, map_location=torch.device("cpu"))
    graph = nx.read_edgelist(f'../../data/snap_dataset/{args.dataset}.txt', create_using=nx.Graph(), nodetype=int)
    # graph = nx.read_gpickle(f"../../data/")
    best_embeddings = None
    encoder.to("cpu")

    dqn_load_path=os.path.join(root_folder,f"{args.dataset}/budget_{soln_budget}/{encoder_name}/trained_dqn")
    with open(dqn_load_path, mode="rb") as f:
        dqn = pickle.load(f)
    dqn.epsilon = 0.01
    dqn.device = "cuda" if cuda else "cpu"
    dqn.net = dqn.net.to(dqn.device)
    # env = TestEnv(graph, soln_budget, subgraph_size, encoder, test_graph_name, action_limit=action_limit, beta=50, cuda=cuda)
    env = BigGraph(graph, soln_budget, subgraph_size, encoder, args.dataset, 
                   action_limit=action_limit, cuda=cuda)
    ratios = []
    rewards = []
    num_nodes = []
    num_edges = []
    for episode in range(num_eps):
        count = 0
        print(f"starting episode {episode+1}")
        state = env.reset()
        done = False
        while not done:
            count += 1
            if count % 100 == 0:
                print(count)
            action, state_for_buffer = dqn.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                num_nodes.append(env.state.number_of_nodes())
                num_edges.append(env.state.number_of_edges())
        print(f"Ratio of {env.ratios[-1]:.3f}, sum of rewards {sum(env.episode_rewards)}\n")
    root_folder = '../../data/LeNSE/MVC/train'

    if num_eps > 1:
        plt.plot(env.ratios)
        plt.xlabel("Test episode number")
        plt.ylabel("Ratio achieved on final subgraph achieved")
        plt.hlines(0.95, 0, num_eps-1, colors="red")
        figure_save_path= os.path.join(root_folder,f"{args.dataset}/budget_{soln_budget}/{encoder_name}/performance_on_test.pdf")
        plt.savefig(figure_save_path)
        plt.show()

        mean_r = np.mean(env.ratios)
        stderror_r = np.std(env.ratios) / np.sqrt(num_eps)

        N = graph.number_of_nodes()
        E = graph.number_of_edges()
        mean_n = np.mean(num_nodes)
        mean_e = np.mean(num_edges)
        stderror_n = np.std(num_nodes) / np.sqrt(num_eps)
        stderror_e = np.std(num_nodes) / np.sqrt(num_eps)
        print(f"Graph: {args.dataset}. Budget: {soln_budget}. Fixed graph size: {subgraph_size}. Number of nodes: {N}. Number of edges: {E}")
        print(f"Average ratio was {mean_r:.4f}, standard error was {stderror_r}")
        print(f"Average number of nodes in final in subgraphs: {mean_n} ({stderror_n}). This is a reduction of {(1 - mean_n / N) * 100}%")
        print(f"Average number of edges in final in subgraphs: {mean_e} ({stderror_e}). This is a reduction of {(1 - mean_e / E) * 100}%\n")

        lines = []
        lines.append(f"Graph: {args.dataset}. Budget: {soln_budget}. Fixed graph size: {subgraph_size}. Number of nodes: {N}. Number of edges: {E}")
        lines.append(f"Average ratio was {np.mean(env.ratios):.4f}, variance was {np.var(env.ratios)}")
        lines.append(f"Average number of nodes in final in subgraphs: {mean_n} ({stderror_n}). This is a reduction of {(1 - mean_n / N) * 100}%")
        lines.append(f"Average number of edges in final in subgraphs: {mean_e} ({stderror_e}). This is a reduction of {(1 - mean_e / E) * 100}%\n")
        lines.append(f"Average time taken for LeNSE was {np.mean(env.lense_times)} minutes.")
        lines.append(f"Average time taken for heuristic was {np.mean(env.heuristic_times)} minutes.")
        lines.append(f"Average total time taken was {np.mean([x + y for x, y in zip(env.lense_times, env.heuristic_times)])} minutes.")

        test_results_path=os.path.join(root_folder,f"{args.dataset}/budget_{soln_budget}/{encoder_name}/test_results.txt")

        with open(test_results_path, mode="w") as f:
            for line in lines:
                f.write(line)
                f.write("\n")
