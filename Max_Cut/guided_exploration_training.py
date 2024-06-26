from environment import GuidedExplorationEnv
import torch
import networkx as nx
from functions import get_best_embeddings, moving_average
from rl_algs import GuidedDQN, DQN
import matplotlib.pyplot as plt
import pickle
import random
import sys
import getopt
import numpy as np
import copy

from argparse import ArgumentParser
def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--seed", type=int,default=1, help="Seed")
    parser.add_argument("--dataset", type=str,default='Facebook', help="Dataset")

    parser.add_argument('--encoder_name', type=str, default="encoder", help='Name of the encoder')
    parser.add_argument('--num_eps', type=int, default=100, help='Number of episodes')
    parser.add_argument('--chunksize', type=int, default=28, help='Chunk size')
    parser.add_argument('--soln_budget', type=int, default=100, help='Solution budget')
    parser.add_argument('--subgraph_size', type=int, default=300, help='Subgraph size')
    parser.add_argument('--selection_budget', type=int, default=7500, help='Selection budget')
    parser.add_argument('--gnn_input', type=int, default=30, help='GNN input size')
    parser.add_argument('--max_memory', type=int, default=20000, help='Maximum memory')
    parser.add_argument('--embedding_size', type=int, default=10, help='Embedding size')
    parser.add_argument('--ff_size', type=int, default=128, help='Feed-forward size')
    parser.add_argument('--beta', type=float, default=1, help='Beta value')
    parser.add_argument('--decay_rate', type=float, default=0.999975, help='Decay rate')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value')
    args = parser.parse_args()





    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # graph_name = "vis_graph"
    graph_name = args.dataset

    # encoder_name = "encoder"

    num_eps = args.num_eps
    chunksize = args.chunksize
    soln_budget = args.soln_budget
    subgraph_size = args.subgraph_size
    selection_budget = args.selection_budget
    gnn_input = args.gnn_input
    max_memory = args.max_memory
    embedding_size = args.embedding_size
    ff_size = args.ff_size
    beta = args.beta
    decay_rate = args.decay_rate
    cuda = args.cuda
    alpha = args.alpha
    args = sys.argv[1:]
    # opts, args = getopt.getopt(args, "g:n:c:e:b:s:d:f:m:C:h:B:E:D:A:")
    # for opt, arg in opts:
    #     if opt in ['-g']:
    #         graph_name = arg
    #     elif opt in ["-n"]:
    #         num_eps = int(arg)
    #     elif opt in ["-c"]:
    #         chunksize = int(arg)
    #     elif opt in ["-e"]:
    #         embedding_size = int(arg)
    #     elif opt in ["-b"]:
    #         soln_budget = int(arg)
    #     elif opt in ["-s"]:
    #         selection_budget = int(arg)
    #     elif opt in ["-d"]:
    #         gnn_input = int(arg)
    #     elif opt in ["-f"]:
    #         subgraph_size = int(arg)
    #     elif opt in ["-m"]:
    #         max_memory = int(arg)
    #     elif opt in ["-C"]:
    #         cuda = bool(int(arg))
    #     elif opt in ["-h"]:
    #         ff_size = int(arg)
    #     elif opt in ["-B"]:
    #         beta = float(arg)
    #     elif opt in ["-E"]:
    #         encoder_name = arg
    #     elif opt in ['-D']:
    #         decay_rate = float(arg)
    #     elif opt in ['-A']:
    #         alpha = float(arg)

    root_folder = '../../data/LeNSE/MaxCut/train'

    encoder = torch.load(f"{graph_name}/budget_{soln_budget}/{encoder_name}/{encoder_name}", map_location=torch.device("cpu"))
    graph = nx.read_gpickle(f"{graph_name}/main")
    best_embeddings = get_best_embeddings(encoder, f"{graph_name}/budget_{soln_budget}/graph_data")
    encoder.to("cpu")
    dqn = GuidedDQN(gnn_input=gnn_input, batch_size=128, decay_rate=decay_rate, ff_hidden=ff_size, state_dim=embedding_size, gamma=0.95, max_memory=max_memory, cuda=cuda)
    env = GuidedExplorationEnv(graph, soln_budget, subgraph_size, encoder, best_embeddings, graph_name, action_limit=selection_budget, beta=beta, cuda=cuda)
    best_embedding = env.best_embedding_cpu.numpy()

    distances = []
    ratios = []
    rewards = []
    for episode in range(num_eps):
        print(f"starting episode {episode+1}")
        state = env.reset()
        done = False
        count = 0
        while not done:
            count += 1
            if count % 100 == 0:
                print(count)
            action, state_for_buffer = dqn.act(state)
            next_state, reward, done = env.step(action)
            dqn.remember(state_for_buffer, reward, next_state[0], done)
            if count % 2 == 0:
                dqn.experience_replay()
            state = next_state

        if dqn.epsilon > dqn.epsilon_min:
            print(f"Exploration rate currently at {dqn.epsilon:.3f}")
        final_dist = distance(env.subgraph_embedding, best_embedding)
        distances.append(-final_dist)
        print(f"final distance of {final_dist}")
        print(f"Ratio of {env.ratios[-1]:.3f}, sum of rewards {sum(env.episode_rewards)}\n")
        ratios.append(env.ratios[-1])
        plt.plot(env.episode_rewards)
        plt.savefig("ep_rewards.pdf")
        plt.clf()

        if (episode + 1) % 5 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot(env.ratios)
            ax1.plot(moving_average(env.ratios, 50))
            ax1.hlines(0.95, 0, len(env.ratios) - 1, colors="red")
            ax2.plot(distances)
            plt.savefig(f"{graph_name}/budget_{soln_budget}/{encoder_name}/dqn_training.pdf")
            plt.close(fig)

            with open(f"{graph_name}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="wb") as f:
                dqn_ = DQN(gnn_input, embedding_size, ff_size, 0.01, batch_size=0, cuda=cuda)
                dqn_.memory = ["hold"]
                dqn_.net = dqn.net
                pickle.dump(dqn_, f)
                del dqn_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(env.ratios)
    ax1.plot(moving_average(env.ratios, 50))
    ax1.hlines(0.95, 0, len(env.ratios) - 1, colors="red")
    ax2.plot(distances)
    plt.savefig(f"{graph_name}/budget_{soln_budget}/{encoder_name}/dqn_training.pdf")
    plt.close(fig)

    dqn.memory = ["hold"]
    dqn.batch_size = 0
    dqn_ = DQN(gnn_input, embedding_size, ff_size, 0.01, batch_size=0, cuda=cuda)
    dqn_.memory = dqn.memory
    dqn_.net = dqn.net
    with open(f"{graph_name}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="wb") as f:
        pickle.dump(dqn_, f)
