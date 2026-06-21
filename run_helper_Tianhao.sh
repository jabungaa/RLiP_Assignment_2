


#works on middle restaurant greedy eval interval 20
#python test_ppo.py --grid grid_configs/restaurant_medium.npy --episodes 10000 --reward high --gamma 0.999 --fourier_freqs 16 --move_distance 0.5 --replay_capacity 16384 --save_train_images --activation relu --entropy_coef 0.005 --greedy_eval_interval 20 --train_start_mode fixed
#testing stochastic start mode
python test_ppo.py --grid grid_configs/restaurant_medium.npy --episodes 10000 --reward high --gamma 0.999 --fourier_freqs 16 --move_distance 0.5 --replay_capacity 16384 --save_train_images --activation relu --entropy_coef 0.005 --greedy_eval_interval 20 --train_start_mode fixed --sigma 0.05

