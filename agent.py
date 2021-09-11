import numpy as np
from replay_buffer import ReplayBuffer
from RLGlue.agent import BaseAgent
from Adam import Adam


class Agent(BaseAgent):
    def __init__(self):
        self.name = "expected_sarsa_agent"

    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            optimizer_config: dictionary : {
                "num_states",
                "num_hidden_layer",
                "num_hidden_units",
                "step_size",
                "beta_m",
                "beta_v",
                "epsilon"},
            replay_buffer_size: integer,
            minibatch_sz: integer,
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))

        # replay_buffer
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'], agent_config.get("seed"))

        # optimizer
        self.optimizer = Adam()
        self.optimizer.optimizer_init(agent_config["optimizer_config"])
        # number of replays
        self.num_replay = agent_config['num_replay_updates_per_step']
        # discount
        self.discount = agent_config['gamma']
        # tau
        self.tau = agent_config['tau']
        # number of states
        self.num_states = agent_config.get("num_states")
        # number of hidden layers : NN
        self.num_hidden_layer = agent_config.get("num_hidden_layer")
        # number of hidden units : NN
        self.num_hidden_units = agent_config.get("num_hidden_units")

        # layer size : NN
        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize the neural network's parameter
        self.weights = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            ins, outs = self.layer_size[i], self.layer_size[i + 1]
            self.weights[i]['W'] = self.rand_generator.normal(0, np.sqrt(2 / ins), (ins, outs))
            self.weights[i]['b'] = self.rand_generator.normal(0, np.sqrt(2 / ins), (1, outs))

        self.last_observation = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0


    def policy(self, observation):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action.
        """
        
        # DONE!
        
        action_values = self.get_action_values(observation)
        probs_batch = self.softmax(action_values, self.tau)
        index = self.rand_generator.choice(len(observation[1]), p=probs_batch[:, 1].squeeze())
        return action_values[index, 0]


    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    # Work Required: Yes. Fill in the action selection, replay-buffer update,
    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        self.sum_rewards += reward
        self.episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array([state])

        # Select action
        # your code here
        action = self.policy(state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        # your code here

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network (~1 Line)
                # your code here
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)

        # Update the last state and last action.
        ### START CODE HERE (~2 Lines)
        self.last_state = state
        self.last_action = action
        ### END CODE HERE
        # your code here

        return action

    # Work Required: Yes. Fill in the replay-buffer update and
    # update of the weights using optimize_network (~2 lines).
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        # Note: look at the replay_buffer append function for the order of arguments
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        # your code here

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network
                # your code here
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
        
        
        
        
    def my_matmul(self, x1, x2):
        """
        Given matrices x1 and x2, return the multiplication of them
        """
        
        # DONE
        
        result = np.zeros((x1.shape[0], x2.shape[1]))
        x1_non_zero_indices = x1.nonzero()
        if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
            result = x2[x1_non_zero_indices[1], :]
        elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
            result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
        else:
            result = np.matmul(x1, x2)
        return result
    
    def get_value(self, s, weights):
        """
        Compute value of input s given the weights of a neural network
        """
        
        # DONE!
        
        psi = self.my_matmul(s, weights[0]["W"]) + weights[0]["b"]
        x = np.maximum(psi, 0)
        v = self.my_matmul(x, weights[1]["W"]) + weights[1]["b"]
        
        return v
    
    def get_gradient(self, s, weights):
        """
        Given inputs s and weights, return the gradient of v with respect to the weights
        """
        
        # DONE!
        
        grads = [dict() for i in range(len(weights))]
        x = np.maximum(self.my_matmul(s, weights[0]["W"]) + weights[0]["b"], 0)
        grads[0]["W"] = self.my_matmul(s.T, (weights[1]["W"].T * (x > 0)))
        grads[0]["b"] = weights[1]["W"].T * (x > 0)
        grads[1]["W"] = x.T
        grads[1]["b"] = 1

        return grads
    
    def one_hot(self, state, num_states):
        """
        Given num_state and a state, return the one-hot encoding of the state
        """

        # DONE!
        
        one_hot_vector = np.zeros((1, num_states))
        one_hot_vector[0, int((state - 1))] = 1
        
        return one_hot_vector
    
    
    def softmax(self, action_values, tau=1.0):
        """
        Args:
            action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                        The action-values computed by an action-value network.              
            tau (float): The temperature parameter scalar.
        Returns:
            A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
            the actions representing the policy.
        """
        
        # DONE!
        
        # Compute the preferences by dividing the action-values by the temperature parameter tau
        preferences = action_values[:, 1] / tau
        # Compute the maximum preference across the actions
        max_preference = np.max(preferences)
        
        # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting 
        # when subtracting the maximum preference from the preference of each action.
        reshaped_max_preference = max_preference.reshape((-1, 1))
        
        # Compute the numerator, i.e., the exponential of the preference - the max preference.
        exp_preferences = np.exp(preferences - reshaped_max_preference)
        # Compute the denominator, i.e., the sum over the numerator along the actions axis.
        sum_of_exp_preferences = np.sum(exp_preferences)
        
        # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting 
        # when dividing the numerator by the denominator.
        reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

        # Compute the action probabilities according to the equation in the previous cell.
        probs = exp_preferences / reshaped_sum_of_exp_preferences
        
        # squeeze() removes any singleton dimensions. It is used here because this function is used in the 
        # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in 
        # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
        probs = probs.squeeze()

        action_probs = np.zeros((len(action_values), 2))
        for i in range(len(action_probs)):
            action_probs[i, 0] = action_values[i, 0]
            action_probs[i, 1] = probs[i]
        return action_probs
    
    
    def get_action_values(self, observation):
        # find values of each next state
        state = observation[0]
        possible_actions = observation[1]
        
        q_values = np.zeros((len(possible_actions), 2))
        
        for i in range(len(possible_actions)):
            temp = state
            temp[possible_actions[i][0], possible_actions[i][1]] = 1
            state_vec = self.one_hot(self.generate_hash(temp), self.num_states)
            v_s = self.get_value(state_vec, self.weights)
            action = 3 * possible_actions[i][0] + possible_actions[i][1]
            
            # POSSIBLE FAULT    
            q_values[i] = (action, -1 + self.discount * v_s)
        
        return q_values 
            
    def generate_hash(self, m):
        my_hash = 0
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                my_hash = my_hash * 3 + int(m[i][j])
        return my_hash
        