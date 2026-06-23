"""DQN Agent

"""
from agents import BaseAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNAgent(BaseAgent):

    def __init__(
            self,
            input_dim,
            output_dim,
            gamma = 0.99,
            learning_rate = 0.001,
            epsilon_start = 1.0,
            epsilon_end = 0.01,
            batch_size = 64,
            replay_buffer_size = 10000,
            target_update_frequency = 1000,
            hidden_sizes = (64, 128, 64),
            device = None
    ):
        super().__init__()

        self.device = device
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start

        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_frequency = target_update_frequency

        self.previous_state = None
        self.previous_action = None
        self.training_step = 0

        self.hidden_sizes = tuple(hidden_sizes)
        self.policy_network = DQNNetwork(input_dim, output_dim, self.hidden_sizes)
        self.target_network = DQNNetwork(input_dim, output_dim, self.hidden_sizes)
        self.replay_buffer = []

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr = self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.policy_network.to(self.device)
        self.target_network.to(self.device)
    
    def state_to_tensor(self, state):
        state = torch.tensor(state, dtype = torch.float32, device = self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        return state
    
    def take_action(self, state):
        original_state = state

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(range(self.output_dim))
        
        else:
            state_tensor = self.state_to_tensor(state)

            with torch.no_grad():
                Q = self.policy_network.forward(state_tensor)

            Q = Q.squeeze(0)
            best_Q = -np.inf
            best_action = []

            for action in range(self.output_dim):
                q_value = Q[action].item()

                if q_value > best_Q:
                    best_Q = q_value
                    best_action = [action]
                
                elif q_value == best_Q:
                    best_action.append(action)
                
            chosen_action =  np.random.choice(best_action)

        self.previous_state = original_state
        self.previous_action = chosen_action

        return chosen_action
    
    def update(self, state, reward, action, done=False):
        if self.previous_action is None or self.previous_state is None:
            return

        transition = (self.previous_state, self.previous_action, reward, state, done)

        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

        if len(self.replay_buffer) >= self.batch_size:
            self.train_step()

            if self.training_step % self.target_update_frequency == 0:
                self.update_target_network()

    def train_step(self):
        batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_network(states)
        current_q_values = q_values.gather(1, actions)
        current_q_values = current_q_values.squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim = 1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_step += 1

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def _set_linear_epsilon(self, step_count, total_steps):
        fraction = min(step_count / total_steps, 1.0)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        return


    def reset_episode(self):
        self.previous_state = None
        self.previous_action = None

class DQNNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes=(64, 128, 64)):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = tuple(hidden_sizes)

        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)