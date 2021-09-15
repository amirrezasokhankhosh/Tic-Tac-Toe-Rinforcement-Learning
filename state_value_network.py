from copy import deepcopy
import numpy as np


class StateValueNetwork:
    def __init__(self, network_config):
        # number of states
        self.num_states = network_config.get("num_states")
        # number of hidden layers
        self.num_hidden_layer = network_config.get("num_hidden_layer")
        # number of hidden units
        self.num_hidden_units = network_config.get("num_hidden_units")
        # discount
        self.discount = network_config['gamma']

        self.rand_generator = np.random.RandomState(network_config.get("seed"))

        # layer size : NN
        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize the neural network's parameter
        self.weights = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            ins, outs = self.layer_size[i], self.layer_size[i + 1]
            self.weights[i]['W'] = self.rand_generator.normal(0, np.sqrt(2 / ins), (ins, outs))
            self.weights[i]['b'] = self.rand_generator.normal(0, np.sqrt(2 / ins), (1, outs))

    def get_value(self, s):
        """
        Compute value of input s given the weights of a neural network
        """

        # DONE!

        psi = self.my_matmul(s, self.weights[0]["W"]) + self.weights[0]["b"]
        x = np.maximum(psi, 0)
        v = self.my_matmul(x, self.weights[1]["W"]) + self.weights[1]["b"]

        return v

    def get_gradient(self, s):
        """
        Given inputs s and weights, return the gradient of v with respect to the weights
        """

        # DONE!

        grads = [dict() for i in range(len(self.weights))]
        x = np.maximum(self.my_matmul(s, self.weights[0]["W"]) + self.weights[0]["b"], 0)
        grads[0]["W"] = self.my_matmul(s.T, (self.weights[1]["W"].T * (x > 0)))
        grads[0]["b"] = self.weights[1]["W"].T * (x > 0)
        grads[1]["W"] = x.T
        grads[1]["b"] = 1

        return grads

    def get_action_values(self, observation):
        # find values of each next state
        state = observation[0]
        possible_actions = observation[1]

        q_values = np.zeros((len(possible_actions), 2))

        for i in range(len(possible_actions)):
            temp = state
            temp[possible_actions[i][0], possible_actions[i][1]] = 1
            state_vec = self.one_hot(self.generate_hash(temp), self.num_states)
            v_s = self.get_value(state_vec)
            action = 3 * possible_actions[i][0] + possible_actions[i][1]

            # POSSIBLE FAULT
            q_values[i] = (action, -1 + self.discount * v_s)

        return q_values

    def one_hot(self, state, num_states):
        """
        Given num_state and a state, return the one-hot encoding of the state
        """

        # DONE!

        one_hot_vector = np.zeros((1, num_states))
        one_hot_vector[0, int((state - 1))] = 1

        return one_hot_vector

    def generate_hash(self, m):
        my_hash = 0
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                my_hash = my_hash * 3 + int(m[i][j])
        return my_hash

    def my_matmul(self, x1, x2):
        """
        Given matrices x1 and x2, return the multiplication of them
        """

        result = np.zeros((x1.shape[0], x2.shape[1]))
        x1_non_zero_indices = x1.nonzero()
        if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
            result = x2[x1_non_zero_indices[1], :]
        elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
            result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
        else:
            result = np.matmul(x1, x2)
        return result

    def get_weights(self):
        """
        Returns:
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)

    def set_weights(self, weights):
        """
        Args: 
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)