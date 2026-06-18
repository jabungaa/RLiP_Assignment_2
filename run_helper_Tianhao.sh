#python train.py grid_configs/A1_grid.npy --no_gui --episodes 100000 --iter 200 --sigma 0
#python train.py grid_configs/A1_grid.npy --no_gui --episodes 100000 --iter 200 --sigma 0
#python train.py grid_configs/large_grid.npy --no_gui --episodes 100000 --iter 200 --sigma 0
#python train.py grid_configs/large_grid.npy --no_gui --episodes 100000 --iter 200 --sigma 0
#python train.py grid_configs/super_hard.npy --no_gui --episodes 2000 --iter 200 --sigma 0 --train_gamma 0.999

#this on works well
#python test_ppo.py --grid grid_configs/small_grid.npy --episodes 500 --reward high --gamma 0.999 --fourier_freqs 64 --move_distance 0.2 --replay_capacity 16384 --activation relu
#works on small restaurant
#python test_ppo.py --grid grid_configs/restaurant_small.npy --episodes 10000 --reward high --gamma 0.999 --fourier_freqs 16 --move_distance 0.5 --replay_capacity 16384 --save_train_images --activation relu --entropy_coef 0.005 --greedy_eval_interval 20
#test on middle restaurant
python test_ppo.py --grid grid_configs/restaurant_medium.npy --episodes 10000 --reward high --gamma 0.999 --fourier_freqs 16 --move_distance 0.5 --replay_capacity 16384 --save_train_images --activation relu --entropy_coef 0.005 --greedy_eval_interval 20 --train_start_mode fixed

#python test_dqn.py --grid grid_configs/small_grid.npy --episodes 2000 --reward low --gamma 0.99 --step_penalty_threshold 50
#python run_experiments.py --grid grid_configs/super_hard.npy --episodes 2000 --iter 200 --sigma 0 --train_gamma 0.999

# python test_ppo.py \
#     --grid grid_configs/large_grid.npy \
#     --episodes 3000 \
#     --policy_lr 1e-3 \
#     --value_lr 1e-3 \
#     --entropy_coef 0.05 \
#     --step_penalty_threshold 30 \
#     --reward low \
#     --replay_buffer_size 1000 \

# python test_ppo.py \
#     --grid grid_configs/small_grid.npy \
#     --reward bfs \
#     --activation tanh \
#     --entropy_coef 0.2 \
#     --policy_lr 1e-3 \
#     --episodes 500


