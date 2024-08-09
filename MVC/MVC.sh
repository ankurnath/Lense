python get_scores_seeds.py --dataset YouTube --budget 100
python fixed_size_dataset.py --num_samples 100 --budget 100 --fixed_size 1250 --dataset YouTube
python embedding_training.py --dataset YouTube --embedding_size 75 --output_size 25 --learning_rate 0.001 
python guided_exploration_training.py --dataset YouTube --subgraph_size 1250 --gnn_input 75 --embedding_size 25 --beta 30 --alpha 0.1 --T_train 1500 --T_test 500 --c 2 
