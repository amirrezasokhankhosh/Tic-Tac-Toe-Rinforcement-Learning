import numpy as np
from replay_buffer import ReplayBuffer
from RLGlue.agent import BaseAgent
from state_value_network import StateValueNetwork
from Adam import Adam
from copy import deepcopy

class Agent(BaseAgent):
    def __init__(self):
        self.name = "expected_sarsa_agent"

    def agent_init(self, agent_config):
        
        # DONE!
        
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
        # network
        self.network = StateValueNetwork(agent_config["network_config"])
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
        
        # DONE!
        
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action.
        """
        action_values = self.network.get_action_values(observation)
        probs_batch = self.softmax(action_values, self.tau)
        index = self.rand_generator.choice(len(observation[1]), p=probs_batch[:, 1].squeeze())
        action_num = action_values[index, 0]
        y = action_num % 3
        x = action_num // 3
        return (x, y)


    def agent_start(self, observation):
        
        # DONE!
        
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
        self.last_observation = observation
        self.last_action = self.policy(self.last_observation)
        return self.last_action

    # Work Required: Yes. Fill in the action selection, replay-buffer update,
    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def agent_step(self, reward, observation):
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


        # Select action
        action = self.policy(observation)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_observation, self.last_action, reward, 0, observation)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_v = deepcopy(self.network)
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network (~1 Line)
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_v, self.tau)

        # Update the last state and last action.
        self.last_observation = observation
        self.last_action = action

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
        
        observation = (np.zeros_like(self.last_observation[0]), [])

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_observation, self.last_action, reward, 1, observation)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_v = deepcopy(self.network)
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network
                # your code here
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_v, self.tau)

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
        
    
    
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
    
     
    
        