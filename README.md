

# DDPG & TD3 Implementation with PyBullet

This repository contains a Deep Deterministic Policy Gradient (DDPG) and Twin Delayed DDPG implementation for reinforcement learning environments provided by PyBullet.

## Structure

- `utils.py`: Contains utility classes and functions such as `ScheduledNoise` and `ReplayBuffer`.
- `models.py`: Contains neural network architectures for the `Actor` and `Critic`.
- `ddpg.py`: Contains the implementation of the DDPG agent (`DDPGAgent`).
- `TD3.py`: Contains the implementation of the DDPG agent (`TD3Agent`).
- `train.py`: The main script that contains the training loop.
- `env_config.py`: Configuration file specifying different environments available for training.

## Environment Configuration

The env_config.py file provides a dictionary of environment names and their respective string identifiers. By default, it includes environments like:

- HalfCheetah: HalfCheetahBulletEnv-v0
- Hopper: HopperBulletEnv-v0
- Walker2D: Walker2DBulletEnv-v0
- Humanoid: HumanoidBulletEnv-v0


You can easily extend this list by adding more environments to the env_config.py file. To train on a specific environment, set the env_name variable in train.py to the desired environment name from the dictionary.
## Setup

1. Install required libraries:
   ```bash
   pip install pybullet jax jaxlib gym dm-haiku optax tensorboardX
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/FayElhassan/DDPG_IMP_PYBULLET
   ```

3. Navigate to the directory:
   ```bash
   cd /path/to/DDPG_IMP_PYBULLET
   ```

## Training

To train the agent, run:
```bash
python train.py
```

This will train the agent on the `HalfCheetahBulletEnv-v0` environment from PyBullet and log metrics using TensorBoard.

## Visualization

Metrics such as reward, actor loss, critic loss, and noise standard deviation are logged using TensorBoard. You can visualize them by:

1. Installing TensorBoard:
   ```bash
   pip install tensorboard
   ```

2. Launching TensorBoard:
   ```bash
   tensorboard --logdir=./runs
   ```

Then, navigate to the URL provided (typically `http://localhost:6006/`) in your web browser.
