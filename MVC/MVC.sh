#!/bin/bash


# python get_scores_seeds.py --dataset YouTube --budget 100
# python fixed_size_dataset.py --dataset YouTube --num_samples 100 --budget 100 --fixed_size 1250
# python embedding_training.py --dataset YouTube --pooling True  --embedding_size 75  --output_size 25 --budget 100 
# python guided_exploration_training.py --dataset YouTube --num_eps 10  --soln_budget 100 --subgraph_size 1250 --selection_budget 1500 --gnn_input 75 --embedding_size 25 --alpha 0.1 --beta 30 
python dqn_test.py --dataset YouTube --num_eps 1 --soln_budget 100 --subgraph_size 1250 --action_limit 1250