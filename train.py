import jax
import jax.numpy as jnp
import numpy as np
import gym
import pybullet_envs
import pybullet_data
from tensorboardX import SummaryWriter
from ddpg import DDPGAgent
from utils import ReplayBuffer
# Define hyperparameters for quick testing
state_dim = 26
action_dim = 6
max_episodes = 100000
max_steps = 500
batch_size = 256
buffer_size = 1000000
lr_actor = 3e-4
lr_critic = 3e-4
gamma = 0.99
tau = 0.005


# Create the HalfCheetah environment from PyBullet
env_hc="HalfCheetahBulletEnv-v0"
env = gym.make(env_hc)
max_action = env.action_space.high[0]

# Initialize the DDPG agent
rng_key = jax.random.PRNGKey(0)
ddpg_agent = DDPGAgent(state_dim, action_dim, max_action ,lr_actor, lr_critic, gamma, tau)

# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# Initialize TensorBoard writer
writer = SummaryWriter(logdir="./run2")

warmup_episodes = 30
timestep = 0 

# Training loop
for episode in range(max_episodes):
    state = env.reset()
    # ddpg_agent.noise.reset()
    episode_reward = 0
    timestep += 1

    for step in range(max_steps):
        timestep += 1
        # Select an action using the actor network
        if episode < warmup_episodes:
            low = [-1. for _ in range(action_dim)]
            high = [1. for _ in range(action_dim)]
            action = np.random.uniform(low, high)
        else:
            action = ddpg_agent.act(jnp.array([state]))[0]

        # Take the action in the environment
        # print(action)
        next_state, reward, done, _ = env.step(action)

        # Store the experience in the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        episode_reward += reward
        state = next_state

        if len(replay_buffer.buffer) > batch_size and episode > warmup_episodes:
            ddpg_agent.update(replay_buffer, batch_size, writer, episode, timestep)

        if done:
            break

    # break
    # Log data to TensorBoard
    writer.add_scalar('Reward', episode_reward, episode)

# Close the environment and TensorBoard writer
env.close()
writer.close()
