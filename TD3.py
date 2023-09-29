@jax.jit
def soft_update(target_params: hk.Params, online_params: hk.Params, tau: float = 0.005) -> hk.Params:
    return jax.tree_map(lambda x, y: (1 - tau) * x + tau * y, target_params, online_params)



class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action ,lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001,policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        _, self.rng_key_actor = jax.random.split(jax.random.PRNGKey(0))
        self.std_schedule = ScheduledNoise()
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.policy_update_counter = 0

       


        # Wrap the initialization inside a function
        def init_networks(rng_key):
            actor = hk.transform(lambda s: Actor(action_dim)(s))
            critic = hk.transform(lambda s, a: Critic()(s, a))
            critic2 = hk.transform(lambda s, a: Critic()(s, a))
            target_actor = hk.transform(lambda s: Actor(action_dim)(s))
            target_critic = hk.transform(lambda s, a: Critic()(s, a))
            target_critic2 = hk.transform(lambda s, a: Critic()(s, a))

            state = jnp.ones((1, state_dim))
            action = jnp.ones((1, action_dim))
            rng_key_actor, rng_key_critic, _, _ = jax.random.split(rng_key, 4)

            actor_params = actor.init(rng_key_actor, state)
            critic_params = critic.init(rng_key_critic, state, action)  # Ensure both state and action are provided here
            critic2_params = critic2.init(rng_key_critic, state, action)
            # Use the same parameters for target networks initially
            target_actor_params = actor_params
            target_critic_params = critic_params
            target_critic2_params = critic2_params
            

            return actor, critic,critic2 , target_actor, target_critic,target_critic2 , actor_params, critic_params,critic2_params, target_actor_params, target_critic_params,target_critic2_params

        init_fn_transformed = hk.transform(init_networks)
        params = init_fn_transformed.init(jax.random.PRNGKey(0), jax.random.PRNGKey(0))

        # Use these params to get your networks and their parameters:
        (self.actor, self.critic, self.critic2,self.target_actor, self.target_critic,self.target_critic2,
         self.actor_params, self.critic_params,self.critic2_params ,self.target_actor_params, self.target_critic_params,self.target_critic2_params) = init_networks(jax.random.PRNGKey(0))

        self.actor_opt = optax.adam(lr_actor)
        self.critic_opt = optax.adam(lr_critic)
        self.critic2_opt=optax.adam(lr_critic)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)
        self.critic2_opt_state = self.critic2_opt.init(self.critic2_params)

        # Initialize PRNG key for actor's actions
        self.rng_key_actor, _ = jax.random.split(jax.random.PRNGKey(0))
  

    def act(self, state):
        # Get the action from the actor network
        action = jax.lax.stop_gradient(self.actor.apply(self.actor_params, self.rng_key_actor, state))

        # Add random noise for exploration
        noise_scale = self.std_schedule.sample()  # You can adjust the scale of the noise
        noise = noise_scale * np.random.randn(self.action_dim)

        # Clip the action to ensure it's within the valid range
        action = np.clip(action + noise, -1, 1)  # Assuming action range is between -1 and 1
        return action
  



    def update(self, replay_buffer, batch_size, writer, episode, timestep):
        # Sample PRNG keys for randomness in this update
        rng_critic, rng_actor, rng_soft_update = jax.random.split(jax.random.PRNGKey(0), 3)

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
       
        # Convert to JAX arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)
        def critic_loss_fn(critic_params: hk.Params,
                           
                            target_critic_params: hk.Params,
                            
                            target_actor_params: hk.Params,
                            state: np.ndarray,
                            action: np.ndarray,
                            next_state: np.ndarray,
                            reward: np.ndarray,
                            not_done: np.ndarray,
                            rng: jnp.ndarray,
            ):
            noise = (
                jax.random.normal(rng, shape=action.shape) * 0.2
            ).clip(-0.5, 0.5)

            next_action = (
                    self.actor.apply(target_actor_params, self.rng_key_actor ,next_state) + noise
            ).clip(-1, 1)

            next_q = self.critic.apply(target_critic_params, 1, next_state, next_action).reshape(-1)
            target_q = jax.lax.stop_gradient(reward.reshape(-1) + self.gamma * next_q * not_done.reshape(-1))
            q = self.critic.apply(critic_params,1, state, action).reshape(-1)
            critic_loss= jnp.mean((q - target_q)**2)

            

            return critic_loss 
        def critic2_loss_fn(
                           critic2_params:hk.Params,
                            
                            target_critic2_params: hk.Params,
                            target_actor_params: hk.Params,
                            state: np.ndarray,
                            action: np.ndarray,
                            next_state: np.ndarray,
                            reward: np.ndarray,
                            not_done: np.ndarray,
                            rng: jnp.ndarray,
            ):
            noise = (
                jax.random.normal(rng, shape=action.shape) * 0.2
            ).clip(-0.5, 0.5)

            next_action = (
                    self.actor.apply(target_actor_params, self.rng_key_actor ,next_state) + noise
            ).clip(-1, 1)

            

            # Critic 2 loss computation
            next_q2 = self.critic2.apply(target_critic2_params, rng, next_state, next_action).reshape(-1)
            target_q2 = jax.lax.stop_gradient(reward.reshape(-1) + self.gamma * next_q2 * not_done.reshape(-1))
            q2 = self.critic2.apply(critic2_params, rng, state, action).reshape(-1)
            critic2_loss = jnp.mean((q2 - target_q2) ** 2)

            return critic2_loss


        



  
        def actor_loss_fn(actor_params, critic_params ,states):
            action = self.actor.apply(actor_params, 1, states)
            values = self.critic.apply(critic_params, 1, states, action).reshape(-1)
            return -jnp.mean(values)

           
        not_done = ~dones
        critic_loss, critic_gradient = jax.value_and_grad(critic_loss_fn)(self.critic_params,self.target_critic_params,self.target_actor_params,
                                                           states, actions, next_states, rewards, not_done, rng_critic)
        
        critic2_loss,critic2_gradient = jax.value_and_grad(critic2_loss_fn)(
                                                            self.critic2_params, self.target_critic2_params, self.target_actor_params,
                                                            states, actions, next_states, rewards, not_done, rng_critic
                                                        )
       

        
        writer.add_scalar("critic_loss", critic_loss, episode)
        writer.add_scalar("critic2_loss", critic2_loss, episode)

        noise = (jax.random.normal(rng_critic, shape=(batch_size, self.action_dim)) * 0.2
                ).clip(-0.5, 0.5)


       
        next_action = (
                    self.actor.apply(self.target_actor_params, self.rng_key_actor ,next_states) + noise
            ).clip(-1, 1)
       



        # TD3: Add noise to the next action for Target Policy Smoothing
        noise = jax.random.normal(rng_critic, shape=next_action.shape) * self.policy_noise
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
        next_action = (next_action + noise).clip(-self.max_action, self.max_action)

        # Compute the minimum of the Q-values from the two target critics
        target_q1 = self.target_critic.apply(self.target_critic_params, rng_critic, next_states, next_action)
        target_q2 = self.target_critic2.apply(self.target_critic2_params, rng_critic, next_states, next_action)
        target_q = rewards + (1 - dones) * self.gamma * jnp.minimum(target_q1, target_q2)


        critic_updates, self.critic_opt_state = self.critic_opt.update(critic_gradient, self.critic_opt_state)
        critic2_updates, self.critic_opt_state = self.critic_opt.update(critic2_gradient, self.critic2_opt_state)
        self.critic_params = optax.apply_updates(self.critic_params, critic_updates)
        self.critic2_params = optax.apply_updates(self.critic2_params, critic2_updates)
    

    
        

        self.std_schedule.update(timestep)

       

        rng_soft_update, subkey = jax.random.split(rng_soft_update)

         # TD3: Delayed policy updates
        self.policy_update_counter += 1
        if self.policy_update_counter % self.policy_delay == 0:
            # # Actor update
            actor_loss, gradient = jax.value_and_grad(actor_loss_fn)(self.actor_params, self.critic_params, states)
            updates, self.actor_opt_state = self.actor_opt.update(gradient, self.actor_opt_state)
            self.actor_params = optax.apply_updates(self.actor_params, updates)
            

            # Soft update of target networks
            self.target_actor_params = soft_update(self.target_actor_params, self.actor_params, self.tau)
            self.target_critic_params = soft_update(self.target_critic_params, self.critic_params, self.tau)
            self.target_critic2_params = soft_update(self.target_critic2_params, self.critic2_params, self.tau)

            writer.add_scalar("actor_loss", actor_loss, episode)
            writer.add_scalar("std", self.std_schedule.std, timestep)
       
@jax.jit
def soft_update(target_params: hk.Params, online_params: hk.Params, tau: float = 0.005) -> hk.Params:
    return jax.tree_map(lambda x, y: (1 - tau) * x + tau * y, target_params, online_params)
