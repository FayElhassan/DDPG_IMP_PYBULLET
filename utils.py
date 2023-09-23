import numpy as np

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
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)    #randombly sample experiences
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return states, actions, rewards, next_states, dones
