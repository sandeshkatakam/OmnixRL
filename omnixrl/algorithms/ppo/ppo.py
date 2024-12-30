class PolicyGradient(BaseAlgorithm):
    def __init__(self, model: BaseModel, input_shape, optimizer, rng):
        super().__init__(model, input_shape, optimizer, rng)

    def select_action(self, state):
        logits, self.rng = self.model.forward(self.params, state, self.rng)
        action = jax.random.categorical(self.rng, logits)
        return action

    def train_step(self, states, actions, rewards, loss_fn):
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_grad_fn(self.params, states, actions, rewards)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss
        
    def update(self, states, actions, rewards):
        """
        Update policy parameters using the collected experience.
        
        Parameters:
            states: Array of states from collected experience
            actions: Array of actions taken
            rewards: Array of rewards received
            
        Returns:
            loss: The training loss for this update
        """
        def loss_fn(params, states, actions, rewards):
            logits, _ = self.model.forward(params, states, self.rng)
            # Example loss computation - should be implemented based on specific algorithm
            return -jnp.mean(rewards)  # Placeholder loss
            
        return self.train_step(states, actions, rewards, loss_fn)