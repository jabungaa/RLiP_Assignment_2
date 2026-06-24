# RL in practice Assignment 2

This repository contains the code for assignment 2 of 2AMC15 Reinfocement learning in practice. The project models a continuous navigation task in a restaurant using a DQN agent and PPO agent.

## Installation

If you want to test the results the required libraries are found in requirements.txt. These can be installed using the following command

pip install -r requirements.txt

## Training and Evaluation

run the following command from the repository root to reproce the experiment that was used to find the results from the report.

```bash
python train_and_evaluate.py --grid grid_configs/restaurant_medium.npy --agents both --eval_sigma 0.1 --seed 43 --start_pos 18,10 --ppo_max_steps_total 400000 --ppo_short_train 100000 --ppo_mid_train 200000 --ppo_eval_steps 2000 --ppo_max_steps_per_episode 2000 --device cpu --dqn_max_steps_total 400000 --dqn_short_train 100000 --dqn_mid_train 200000 --dqn_max_steps_per_episode 2000 --eval_episodes 30 --no_gui
```

Partial convergence is actually the most likely outcome. 

## Outputs

The script exports its output to the results folder. These outputs include:

- `combined_results_<timestamp>.json`: final metrics for the selected agents.
- `<timestamp>_DQN_convergence.png`: DQN training convergence plot.
- `<timestamp>_PPO_convergence.png`: PPO training convergence plot.
- `<timestamp>_auc_plot.png`: SPL-vs-training-steps comparison plot.

## Metrics

The main comparison uses:

- Training success rate.
- Evaluation success rate.
- Optimal-path rate.
- SPL (Success weighted by Path Length).
- Average number of failed moves/collisions.
- AUC over SPL at short, mid, and full training budgets.

## Code guide

The code is made up of 2 modules: 

1. `agent`
2. `world`

### The `agent` module

The `agent` module contains the `BaseAgent` class as well as some benchmark agents you may want to test against.

The `BaseAgent` is an abstract class and all RL agents for DIC must inherit from/implement it.
`PPO.py` contains our PPO agent and `DQN_agent.py` our DQN agent.

### The `world` module

The world module contains:
1. `grid_creator.py`
2. `environment_continuous.py`
3. `grid.py`
4. `grid_continuous.py`
5. `gui.py`

#### Grid creator
Run this file to create new grids.

```bash
$ python grid_creator.py
```

This will start up a web server where you create new grids, of different sizes with various elements arrangements.
To view the grid creator itself, go to `127.0.0.1:5000`.
All levels will be saved to the `grid_configs/` directory.

#### The Continuous Environment

`Environment_Continuous` is very important because it contains everything we hold dear, including ourselves [^1].
It is also the name of the class which our RL agent will act within. Most of the action happens in there.

The main interaction with `Environment_Continuous` is through the methods:

- `EnvironmentContinuous()` to initialize the environment
- `reset()` to reset the environment
- `step()` to actually take a time step with the environment

[^1]: In case you missed it, this sentence is a joke. Please do not write all your code in the `Environment` class.

#### The Grid

The `Grid` class is the the actual representation of the world on which the agent moves. It is a 2D Numpy array.

#### Grid Continuous
Converts the grid created by grid_creator.py to a continuous grid.

```bash
$ python grid_continuous.py
```

This will save a visualization of the continuous grid specified at line 251 of the file.

#### The GUI
The Graphical User Interface provides a way for you to actually see what the RL agent is doing.
While performant and written using PyGame, it is still about 1300x slower than not running a GUI.
Because of this, we recommend using it only while testing/debugging and not while training.


#### Optimal Path
Calculates the optimal path for a given grid and start position using a breadth-first-search algorithm.

```bash
$ python optimal_path.py
```

This will save a visualization of the optimal path of the continuous grid specified at line 207 of the file.

#### Evaluation
The sole purpose of this file is to hold the evaluation function `evaluate_agent()` which we use to evaluate our agents.

#### Train DQN
This file has 2 purposes. It holds the functions `train_DQN()` and `evaluate_DQN()` which we always use to respectively train and evaluate our DQN algorithm.

#### Train PPO
This file has 2 purposes. It holds the functions `train_PPO()` and `evaluate_PPO()` which we always use to respectively train and evaluate our PPO algorithm.

#### Train and Evaluate
This file trains and evaluates 1 or either of both DQN and PPO algorithms. Running this file will train and evaluate the specified algorithm(s) with the specified hyperparameters.

```bash
$ python train_and_evaluate.py
```

Running this file will train and evaluate the specified algorithm(s) with the specified hyperparameters.