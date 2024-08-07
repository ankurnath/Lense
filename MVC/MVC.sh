python get_scores_seeds.py --dataset YouTube --budget 100
python fixed_size_dataset.py --num_samples 100 --budget 100 --fixed_size 1250 --dataset YouTube
python embedding_training.py --dataset YouTube --embedding_size 75 --output_size 25 --learning_rate 0.001