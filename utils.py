import numpy as np
import pickle 
class ScheduledNoise:
    def __init__(self, decay_rate=0.00001):
        """Initialize parameters and noise process."""
        self.std_start = self.std = 1    # Initial standard deviation for noise
        self.std_min = 0.1    # Minimum standard deviation for noise
        self.decay_rate = decay_rate    # Rate at which the standard deviation decays

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        pass    # Not needed for this particular noise model

    def sample(self):
        """Update internal state and return it as a noise sample."""
        return self.std    # Return the current standard deviation as the noise sample

    def update(self, timestep):
        # Decay the standard deviation towards the minimum value
        self.std = self.std_min + (self.std_start - self.std_min) * np.exp(-1 * timestep * self.decay_rate)

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size    # Maximum number of experiences to store
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)    # Remove the oldest experience if the buffer is full
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)    #randomly sample experiences
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return states, actions, rewards, next_states, dones
environments_data = {
    "HalfCheetah": {"state_dim": 26, "action_dim": 6}
}

def load_params(directory):
  with open(directory, 'rb') as handle:
    params = pickle.load(handle)
  print("dictionary have been loaded!")
  return params

def load_agent_from_params_file(agent_class, params_file_path, env_name):
    state_dim = environments_data[env_name]["state_dim"]
    action_dim = environments_data[env_name]["action_dim"]
    policy_params = load_params(params_file_path)
    agent = agent_class(state_dim, action_dim, 1.)
    agent.actor_params = policy_params
    return agent
