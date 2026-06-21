# PPO Running Branch

This document describes only the PPO execution path started by `run_helper_PPO.sh`. It does not cover `train.py`, DQN, the older PPO files, Bayesian search, or the other experiment runners.

## Run the branch

From the assignment directory:

```bash
bash run_helper_PPO.sh
```

The active command in the script is equivalent to:

```bash
python test_ppo.py \
  --grid grid_configs/restaurant_medium.npy \
  --episodes 10000 \
  --reward high \
  --gamma 0.999 \
  --fourier_freqs 16 \
  --move_distance 0.5 \
  --replay_capacity 16384 \
  --save_train_images \
  --activation relu \
  --entropy_coef 0.005 \
  --greedy_eval_interval 20 \
  --train_start_mode fixed \
  --sigma 0.05
```

Install the dependencies first if necessary:

```bash
pip install -r requirements.txt
```

## Files used by this execution path

| File | Responsibility |
|---|---|
| `run_helper_PPO.sh` | Selects the active experiment command. |
| `test_ppo.py` | Parses arguments, constructs the environment and agent, computes the baseline, coordinates training/evaluation, and writes results. |
| `train_ppo.py` | Implements the training loop, stochastic checkpoint evaluation, metric aggregation, and early stopping. |
| `agents/PPO.py` | Defines the actor, critic, rollout storage, GAE calculation, replay buffer, and PPO optimization. |
| `world/environment_continuous.py` | Implements continuous movement, lidar observations, collision checks, rewards, targets, and `sigma` action noise. |
| `evaluation.py` | Runs the agent in greedy policy mode and collects episode-level evaluation data. |
| `optimal_path.py` | Computes the approximate minimum-action baseline with state-lattice BFS. |

## Effective configuration

Arguments not shown in the shell script come from `test_ppo.py` defaults.

| Setting | Value |
|---|---:|
| Grid dimensions | 20 × 15 cells |
| Start cell | `(18, 10)` from the grid's start marker |
| Training start mode | Fixed |
| Maximum training episodes | 10,000 |
| Maximum steps per training episode | 1,000 |
| Environment stochasticity (`sigma`) | 0.05 |
| Evaluation interval | Every 20 training episodes |
| Final stochastic evaluations | 100 |
| Discount factor (`gamma`) | 0.999 |
| GAE lambda | 0.95 |
| PPO clipping epsilon | 0.2 |
| Actor/critic learning rate | 0.0003 / 0.0003 |
| Entropy coefficient | 0.005 |
| Optimization epochs per update | 4 |
| Minibatch size | 64 |
| Rollout size | 4,096 transitions |
| Replay capacity | 16,384 transitions |
| Hidden layers | 128, 128 |
| Activation | ReLU |
| Fourier frequency bands | 16 |
| Agent radius | 0.2 m |
| Move distance | 0.5 m |
| Turn angle | 15° |
| Seed | 1 |
| Device | CPU |

## Runtime process

### 1. Shell entry point

`run_helper_PPO.sh` invokes `test_ppo.py` with the stochastic medium-restaurant configuration above. The earlier greedy command in the script is commented out and is not executed.

### 2. Argument and dependency setup

`test_ppo.py` parses the command line, seeds NumPy, imports PyTorch and the continuous environment, and verifies the requested device. Because no explicit start position is provided, the first grid cell marked with value `4` is selected: `(18, 10)`.

The fixed training sampler always returns this same cell. Evaluation also uses this cell.

### 3. Environment construction

`EnvironmentContinuous` is created without a GUI. The agent starts at the center of the selected cell, `(18.5, 10.5)`, with the environment's current fixed initial heading of `-π` radians.

The state contains 22 normalized values:

- normalized `x` and `y`;
- `cos(theta)` and `sin(theta)`;
- 18 normalized lidar distances covering 360°.

The four actions are:

| Action | Meaning |
|---:|---|
| 0 | Move forward by 0.5 m |
| 1 | Turn left by 15° |
| 2 | Turn right by 15° |
| 3 | Move backward by 0.5 m |

Collision checks use the swept path of the agent disc against the Shapely wall geometry. An episode terminates when the disc overlaps a target region.

### 4. Environmental stochasticity

The policy proposes an action at every step. With `sigma = 0.05`, the environment replaces that proposal with a uniformly random action on 5% of calls. The random replacement can select the original action again, so the effective probability of executing a different action is 3.75%:

```text
0.05 × 3/4 = 0.0375
```

This noise is active during both training and evaluation. During evaluation the policy is greedy, but the environment remains stochastic.

### 5. Reward processing

The selected `high` reward function returns:

| Event | Raw reward | Reward stored by PPO (`÷ 10`) |
|---|---:|---:|
| Target reached | 1000.0 | 100.0 |
| Successful forward/backward move | 0.1 | 0.01 |
| Collision | -5.0 | -0.5 |
| Turn | -1.0 | -0.1 |

`test_ppo.py` automatically chooses a reward scale of 10. PPO divides every reward by this value before storing it.

### 6. Heuristic baseline

Before creating the PPO agent, `test_ppo.py` resets the environment and calls `optimal_path.approx_optimal_steps()`.

The planner performs an approximate breadth-first search over position and heading states. Its primitive actions are forward, left turn, and right turn; positions are deduplicated at 0.1 m resolution. The initial heading is treated as free for the baseline calculation. With the current medium grid and movement settings, the observed baseline is 22 actions, making the 150% threshold 33 actions.

This is a heuristic reference rather than a mathematical guarantee: the planner discretizes position, treats initial heading as free, and does not include the agent's backward action.

### 7. PPO model construction

The raw 22-value state is augmented with Fourier features for `x` and `y`. Each of the 16 frequency bands contributes four values (`sin` and `cos` for both coordinates), so the actor and critic each receive:

```text
22 + (16 × 4) = 86 input features
```

The actor and critic are separate two-layer MLPs:

```text
86 inputs → 128 ReLU → 128 ReLU → actor: 4 action logits
                                      critic: 1 state value
```

Linear layers use orthogonal initialization. The actor's final layer uses a small gain of 0.01 so the initial action distribution is close to uniform.

### 8. Training episode loop

`train_ppo()` sets the agent to training mode. For each episode:

1. Reset the environment at `(18, 10)`.
2. Reset the agent's per-episode pending action state.
3. Ask the actor to sample an action from its categorical distribution.
4. Let the environment apply `sigma` and execute the resulting action.
5. Return the next state, reward, terminal flag, and actual action.
6. Store the policy state/action/log-probability, scaled reward, value estimate, and terminal information.
7. Repeat until the target is reached or 1,000 steps have elapsed.

The `action` argument passed to `PPO_agent.update()` contains the environment's actual action, but the current implementation stores the policy's proposed action from `take_action()`. Therefore, action replacement is modeled as environmental transition noise rather than as an action relabeling operation.

### 9. Rollouts, GAE, replay, and optimization

Transitions accumulate across episode boundaries. PPO does not optimize after every episode. It normally updates after collecting 4,096 transitions.

True terminal states cut the GAE chain with a bootstrap value of zero. A 1,000-step episode truncation cuts the chain and bootstraps from the critic's value of the final state. When 4,096 transitions are available, the agent:

1. Calculates generalized advantages using `gamma = 0.999` and `lambda = 0.95`.
2. Calculates value targets as `advantage + old_value`.
3. Normalizes advantages.
4. Adds the processed transitions to the 16,384-transition replay buffer.
5. Optimizes over all transitions currently in replay for four epochs using shuffled minibatches of 64.
6. Applies PPO ratio clipping to `[0.8, 1.2]`, critic MSE loss, and the entropy bonus.

Using replay makes this branch a replay-augmented/off-policy variant of PPO: older transitions, old log probabilities, advantages, and returns are reused across several updates. Setting `--replay_capacity 0` would restore the on-policy rollout behavior.

The configured gradient-norm limit is 1,000,000, which is effectively non-restrictive in normal training.

### 10. Training trajectory images

Because `--save_train_images` is enabled, successful trajectories are saved under:

```text
results/<run-start-timestamp>_training/
```

The loop saves the first ten consecutive successes and then saves at doubling milestones such as 20, 40, 80, and so on. A failure resets this consecutive-success schedule.

### 11. Evaluation every 20 episodes

After every 20 training episodes, `test_ppo.py` requests 100 evaluations because `sigma` is nonzero.

For these checkpoint evaluations:

- `set_training(False)` makes the actor choose `argmax` actions instead of sampling;
- `sigma = 0.05` remains active in the environment;
- episode seeds begin at 1 and increase for each attempted run;
- each run is capped at the 33-step efficiency threshold;
- the batch stops immediately when one run fails to reach the target within 33 steps;
- no checkpoint trajectory image is saved.

The fail-fast behavior is implemented in `train_ppo.evaluate_ppo()`. It calls the shared evaluator one episode at a time. If all attempted runs pass, it continues until 100 have completed.

Checkpoint metrics require careful interpretation:

- `eval_total_episodes` is the number actually attempted before fail-fast termination;
- `eval_success_rate` is successes divided by attempted runs;
- `eval_within_150pct_baseline_count` is the number of qualifying runs;
- `eval_within_150pct_baseline_rate` divides qualifying runs by the requested 100, including skipped runs as not yet passed.

For example, if nine runs pass and the tenth fails, the ordinary success rate is `9/10 = 90%`, but the threshold completion rate is `9/100 = 9%`.

All checkpoint summaries are appended to `evaluation_history` with the corresponding training episode.

### 12. Early stopping

Training stops only when `eval_within_150pct_baseline_rate == 1.0`, meaning all 100 stochastic evaluations succeeded within 33 steps.

After the checkpoint triggers early stopping, `train_ppo()` calls `agent.finish_rollout()`. This flushes and optimizes any remaining training transitions. Consequently, the policy used for the final evaluation can differ slightly from the policy that passed the early-stop checkpoint.

If no checkpoint passes, training continues to the 10,000-episode limit.

### 13. Final evaluation

The final evaluation always performs all 100 stochastic episodes with the normal 1,000-step cap. It does not use checkpoint fail-fast behavior because final evaluation enables image saving.

The actor remains greedy while environmental `sigma` remains 0.05. Metrics include success count, average steps, step counts per episode, average reward, the number and rate within the 33-step threshold, and SPL.

The aggregate SPL reported by this branch is:

```text
success_rate × baseline / max(baseline, average_steps)
```

The final trajectory image represents only the last of the 100 evaluation episodes.

## Outputs

The run start timestamp is used for training images, checkpoint weights, and metrics:

```text
results/<run-start-timestamp>_training/
results/<run-start-timestamp>_checkpoint.pt
results/<run-start-timestamp>_metrics.json
```

The final evaluation PNG uses the evaluation completion timestamp instead:

```text
results/<evaluation-completion-timestamp>.png
```

This naming difference can make the final image appear to be missing when searching only for the run start timestamp.

The checkpoint contains actor weights, critic weights, and command-line settings. It does not contain optimizer state, replay-buffer contents, or pending rollout data, so it is not a complete training-resume checkpoint.

The metrics JSON contains:

- effective settings and checkpoint path;
- heuristic baseline;
- total and recent training success statistics;
- interval `evaluation_history`;
- training duration;
- final success, steps, reward, threshold, and SPL metrics;
- final episode position and trajectory.

## Process summary

```text
run_helper_PPO.sh
  → test_ppo.py parses the stochastic experiment
  → continuous environment and heuristic baseline are built
  → actor/critic PPO agent is initialized
  → sampled-policy training with sigma action noise
  → PPO/replay update every 4,096 collected transitions
  → every 20 episodes: greedy policy + stochastic environment evaluation
       ↳ stop checkpoint on first threshold miss
       ↳ stop training only after 100/100 threshold passes
  → flush remaining rollout
  → full final 100-episode stochastic evaluation
  → save PNG, checkpoint, and metrics JSON
```

## Practical notes

- Run the shell script from this assignment directory because its paths are relative.
- The active command requests CPU. Add `--device cuda` only when the installed PyTorch build reports CUDA support.
- Repeated checkpoint results can be identical between PPO updates because checkpoints occur every 20 episodes while model updates are transition-based at 4,096 steps.
- Checkpoint evaluation always reuses the same sequence of seeds, which makes policy comparisons reproducible but does not test unseen random sequences.
- For broader robustness evidence, repeat training with additional seeds and start positions.
