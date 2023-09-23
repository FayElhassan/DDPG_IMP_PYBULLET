import numpy as np

class ScheduledNoise:
    def __init__(self, decay_rate=0.00001):
        """Initialize parameters and noise process."""
        self.std_start = self.std = 1
        self.std_min = 0.1
        self.decay_rate = decay_rate

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        pass

    def sample(self):
        """Update internal state and return it as a noise sample."""
        return self.std

    def update(self, timestep):
        self.std = self.std_min + (self.std_start - self.std_min) * np.exp(-1 * timestep * self.decay_rate)

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return states, actions, rewards, next_states, dones
