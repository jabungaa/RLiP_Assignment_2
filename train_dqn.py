import copy
from pathlib import Path
from typing import Any
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm, trange

from agents.DQN_agent import DQNAgent
from world.environment_continuous import EnvironmentContinuous

def train_DQN(
    grid: str | Path,
    #evaluate the performance of the agent after 3 different amounts of training (short, mid, long) to see how performance improves with training and compare sample efficiency.
    short_train_steps_eval: int=100000, 
    mid_train_steps_eval: int=250000,
    max_steps_total: int=500000,
    n_episodes_epsilon_decay: int = 500, #sets an epsilon decay schedule but this may not match the number of actual episodes
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
    step_count=0 #initialize a step counter which determines when we evaluate and stop
    short_train_agent=None
    mid_train_agent=None
    print(f"Training DQN agent on grid {grid} for a maximum of {max_steps_total} steps...")
    for episode in range(100000):#just a very high number we're never going to reach, we break based on total steps
        state = env.reset()
        agent.reset_episode()

        agent._set_linear_epsilon(episode, n_episodes_epsilon_decay)

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
            step_count+=1
            if terminated:
                break
            #save models at different stages of training for evaluation later and print progress every 10k steps
            if step_count % 10000 == 0:
                print(f"Step {step_count}/{max_steps_total}, Episode {episode}")
            if short_train_agent is None and step_count >= short_train_steps_eval:
                print(f"Store agent for evaluation at step {step_count}/{max_steps_total}:")
                short_train_agent=copy.deepcopy(agent)
            if mid_train_agent is None and step_count >= mid_train_steps_eval:
                print(f"Store agent for evaluation at step {step_count}/{max_steps_total}:")
                mid_train_agent=copy.deepcopy(agent)
            if step_count== max_steps_total:
                print(f"Reached max steps {max_steps_total} on episode {episode}. Ending training.")
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
        #stop when max steps reached
        if step_count== max_steps_total:
            break 
    return agent, training_history, short_train_agent, mid_train_agent #return the final agent and the agents at the short and mid training points for evaluation

def evaluate_DQN(
    agent: DQNAgent,
    grid: str | Path,
    max_steps_per_episode: int = 1000,
    sigma: float = 0.1,
    agent_start_pos: tuple[int, int] | None = None,
    no_gui: bool = True,
    random_seed: int = 0,
    move_distance: float = 0.5,
    episodes: int = 100,  # multiple episodes for a more stable estimate
    reward_fn=None,
    optimal_steps: int = 25,  # SPL normalisation (25 = medium-grid optimum at step 0.5)
):
    """Greedy evaluation of a DQN agent via the shared `evaluate_agent`.

    Uses the exact same evaluation procedure as PPO; the result is remapped to
    the legacy DQN key names so existing callers (e.g. `train_and_evaluate.py`)
    keep working.
    """
    from evaluation import evaluate_agent

    res = evaluate_agent(
        agent, grid,
        episodes=episodes,
        max_steps=max_steps_per_episode,
        sigma=sigma,
        agent_start_pos=agent_start_pos,
        seed=random_seed,
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
