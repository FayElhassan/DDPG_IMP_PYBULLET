

# DDPG Implementation with PyBullet

This repository contains a Deep Deterministic Policy Gradient (DDPG) implementation for reinforcement learning environments provided by PyBullet.

## Structure

- `utils.py`: Contains utility classes and functions such as `ScheduledNoise` and `ReplayBuffer`.
- `models.py`: Contains neural network architectures for the `Actor` and `Critic`.
- `ddpg.py`: Contains the implementation of the DDPG agent (`DDPGAgent`).
- `train.py`: The main script that contains the training loop.

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
