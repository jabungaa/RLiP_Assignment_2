from pathlib import Path
from typing import Any
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

from agents.DQN_agent import DQNAgent
from world.environment_continuous import EnvironmentContinuous

def train_DQN(
    grid: str | Path,
    n_episodes: int = 500,
    max_steps_per_episode: int = 1000,
    sigma: float = 0.1,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    batch_size: int = 64,
    replay_buffer_size: int = 10_000,
    target_update_frequency: int = 1000,
    random_seed: int = 0,
    agent_start_pos: tuple[int, int] | None = None,
    no_gui: bool = True,
    device: str | None = None,
) -> tuple[DQNAgent, list[dict[str, Any]]]:
    
    grid = Path(grid)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    env = EnvironmentContinuous(
        grid_fp= grid,
        no_gui= no_gui,
        sigma= sigma,
        agent_start_pos= agent_start_pos,
        random_seed= random_seed
        )
    
    agent = DQNAgent(
        input_dim= EnvironmentContinuous.STATE_SIZE,
        output_dim= EnvironmentContinuous.N_ACTIONS,
        gamma= gamma,
        learning_rate= learning_rate,
        epsilon_start= epsilon_start,
        epsilon_end= epsilon_end,
        batch_size= batch_size,
        replay_buffer_size= replay_buffer_size,
        target_update_frequency= target_update_frequency,
        device= device,
    )

    training_history = []

    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        agent.reset_episode()

        agent._set_linear_epsilon(episode, n_episodes)

        total_reward = 0.0
        terminated = False

        for step in range(max_steps_per_episode):
            action = agent.take_action(state)

            next_state, reward, terminated, info = env.step(action)

            agent.update(
                state= next_state,
                reward= reward,
                action= info.get("actual_action", action),
                done= terminated
            )

            state = next_state
            total_reward += reward

            if terminated:
                break
        
        episode_info = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": step + 1,
            "terminated": terminated,
            "epsilon": agent.epsilon,
            "targets_reached": env.world_stats.get("total_targets_reached", 0),
            "failed_moves": env.world_stats.get("total_failed_moves", 0),
        }

        training_history.append(episode_info)
        # print(episode_info)

    return agent, training_history 

def evaluate_DQN(
    agent: DQNAgent,
    grid: str | Path,
    max_steps_per_episode: int = 1000,
    sigma: float = 0.1,
    agent_start_pos: tuple[int, int] | None = None,
    no_gui: bool = True,
    random_seed: int = 0,
):
    grid = Path(grid)

    env = EnvironmentContinuous(
        grid_fp= grid,
        no_gui= no_gui,
        sigma= sigma,
        agent_start_pos= agent_start_pos,
        random_seed= random_seed,
    )

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset()
    agent.reset_episode()

    total_reward = 0.0
    terminated = False

    for step in tqdm(range(max_steps_per_episode)):
        action = agent.take_action(state)

        next_state, reward, terminated, info = env.step(action)
        state = next_state
        total_reward += reward

        if terminated:
            break

    agent.epsilon = old_epsilon

    episode_result = {
        "total_reward": total_reward,
        "steps": step + 1,
        "terminated": terminated,
        "targets_reached": env.world_stats.get("total_targets_reached", 0),
        "failed_moves": env.world_stats.get("total_failed_moves", 0),
    }

    return episode_result

# agent, history = train_DQN(
#     grid= "grid_configs/A1_grid.npy",
#     n_episodes= 500,
#     max_steps_per_episode= 500,
#     sigma= 0.1,
#     epsilon_start= 0.5,
#     epsilon_end= 0,
#     no_gui= True,
# )
# 
# print(evaluate_DQN(
#     agent= agent,
#     grid= "grid_configs/A1_grid.npy",
#     max_steps_per_episode= 500,
#     sigma= 0.0,
#     no_gui= False))
