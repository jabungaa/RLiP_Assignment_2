"""Standalone PPO success-rate runner.
Functions to be used by train_and_evaluate.py

"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from math import radians
from pathlib import Path
import copy

from matplotlib.pyplot import step
import numpy as np
from tqdm import trange
import random
import torch
import os

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)  # controls Python hash randomness


def train_ppo(agent, env,  start_pos, max_steps_total: int=500000, short_train_steps_eval: int=50000,  mid_train_steps_eval: int=100000, max_steps_per_episode: int=1000,
              train_images_dir=None, greedy_eval_interval=0, greedy_eval_fn=None, seed=0):
    set_all_seeds(seed)
    agent.set_training(True)
    training_history = []
    step_count=0 #initialize a step counter which determines when we evaluate and stop
    short_train_agent=None
    mid_train_agent=None
    print(f"Training PPO agent for a maximum of {max_steps_total} steps...")
    for episode in range(100000):#just a very high number we're never going to reach, we break based on total steps
        state = env.reset(agent_start_pos=start_pos)
        agent.new_episode(state)

        total_reward = 0.0
        terminated = False

        for step in range(max_steps_per_episode):
            action = agent.take_action(state)

            next_state, reward, terminated, info = env.step(action)

            agent.update(state, reward, info["actual_action"], terminated)

            state = next_state
            total_reward += reward
            step_count+=1
            #print progress every 10k steps
            if step_count % 10000 == 0:
                print(f"Step {step_count}/{max_steps_total}, Episode {episode}")
            if terminated:
                break
            #save models at different stages of training for evaluation later
            if short_train_agent is None and step_count >= short_train_steps_eval:
                print(f"Store agent for evaluation at step {step_count}/{max_steps_total}:")
                short_train_agent=copy.deepcopy(agent)
            if mid_train_agent is None and step_count >= mid_train_steps_eval:
                print(f"Store agent for evaluation at step {step_count}/{max_steps_total}:")
                mid_train_agent=copy.deepcopy(agent)
            if step_count== max_steps_total:
                print(f"Reached max steps {max_steps_total} on episode {episode}. Ending training.")
                break

        else:
            agent.finish_rollout(state)

        episode_info = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": step + 1,
            "terminated": terminated,
            "targets_reached": env.world_stats.get("total_targets_reached", 0),
            "failed_moves": env.world_stats.get("total_failed_moves", 0),
        }

        training_history.append(episode_info)
        #stop when max steps reached
        if step_count== max_steps_total:
            break 
    return agent, training_history, short_train_agent, mid_train_agent #return the final agent and the agents at the short and mid training points for evaluation


def evaluate_ppo(agent, Environment, grid_fp, reward_fn, start_pos, sigma,
                 seed, episodes, max_steps, agent_radius=0.2,
                 move_distance=0.2, turn_angle_deg=15.0, optimal_steps: int = 23):
    
    from evaluation import evaluate_agent

    res = evaluate_agent(
        agent, grid_fp,
        episodes=episodes,
        max_steps=max_steps,
        sigma=sigma,
        agent_start_pos=start_pos,
        seed=seed,
        reward_fn=reward_fn,
        move_distance=move_distance,
        optimal_steps=optimal_steps,
    )
    return {
        "eval_success_rate": res["eval_success_rate"],
        "SPL": res["eval_avg_spl"] if res["eval_avg_spl"] is not None else 0.0,
        "total_reward": res["eval_avg_reward"],
        "avg_steps": res["eval_avg_steps"],
        "avg_failed_moves": res["eval_avg_failed_moves"],
    }
    # agent.set_training(False)

    # cumulative_total_reward=0
    # step_counter=0
    # SPL=0
    # failed_moves=0
    # targets_reached=0

    # for ep in trange(episodes, desc="Evaluating PPO"):
    #     env = Environment(
    #         grid_fp=grid_fp,
    #         no_gui=True,
    #         sigma=sigma,
    #         agent_start_pos=start_pos,
    #         random_seed=seed + ep,
    #         reward_fn=reward_fn,
    #         target_fps=-1,
    #         agent_radius=agent_radius,
    #         move_distance=move_distance,
    #         turn_angle=radians(turn_angle_deg),
    #     )
    #     state = env.reset()
    #     path = [(env.x, env.y)]
    #     actions = []
    #     terminated = False
    #     total_reward=0
    #     for step in range(max_steps):
    #         action = agent.take_action(state)
    #         actions.append(action)
    #         state, _reward, terminated, _info = env.step(action)
    #         path.append((env.x, env.y))
    #         step_counter+=1
    #         total_reward+=_reward
    #         if terminated:
    #             break
        
    #     SPL+=terminated*23/(step+1) #this is currently hardcoded for the medium grid start position (18,10) where 23 is the optimal number of steps with step size of 0.5
    #     failed_moves+=env.world_stats.get("total_failed_moves", 0)
    #     targets_reached+=terminated
    #     cumulative_total_reward+=total_reward
    # SPL=SPL/episodes
    # avg_steps=step_counter/episodes
    # avg_total_reward=cumulative_total_reward/episodes
    # avg_failed_moves=failed_moves/episodes
    # avg_success_rate=targets_reached/episodes
    # eval_metrics= {
    #     "eval_success_rate": avg_success_rate if episodes > 0 else 0.0,
    #     "eval_spl": float(SPL),
    #     "total_reward": avg_total_reward,
    #     "eval_avg_steps": float(avg_steps),
    #     "eval_avg_failed_moves": float(avg_failed_moves),
    # }
    # return eval_metrics


