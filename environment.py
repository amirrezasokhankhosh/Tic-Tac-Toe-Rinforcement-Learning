#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from RLGlue import BaseEnvironment
import numpy as np
import random


class TicTacToeEnvironment(BaseEnvironment):
    def __init__(self):
        pass

    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        # initialize board
        self.board = np.zeros((3, 3))

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        reward = 0.0
        self.board = np.zeros((3, 3))
        state = np.zeros((3, 3))
        actions = self.find_actions(state)
        observation = (state, actions)
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        # return first state observation from the environment
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        self.take_action(action)
        reward, is_terminal = self.get_reward()
        if not is_terminal:
            self.take_action_opp()
            state = np.copy(self.board)
            actions = self.find_actions(state)
            observation = (state, actions)
        else:
            observation = (0, 0)
        reward, is_terminal = self.get_reward()
        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term

    def find_actions(self, state):
        actions = []
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                if state[i, j] == 0:
                    actions.append((i, j))
        return actions

    def get_board(self):
        return self.board

    def set_board(self, board):
        self.board = board

    def take_action(self, action):
        """
            0 represents none chosen grid.
            1 represents agent action.
            2 represents opponent action.
        """
        self.board[int(action[0]), int(action[1])] = 1
        
    def take_action_opp(self):
        possible_actions = self.find_actions(np.copy(self.board))
        action = random.choice(possible_actions)
        self.board[int(action[0]), int(action[1])] = 2

    def get_reward(self):
        """
            Win : 10
            Draw : 0
            Loose : -10
            Each time step : -1
        """
        reward = -1
        is_terminal = False
        if np.count_nonzero(self.board) == 9:
            is_terminal = True
        for x in self.board:
            if np.array_equal(x, np.ones(3)):
                reward += 10
                is_terminal = True
                return reward, is_terminal
            elif np.array_equal(x, np.full(3, 2)):
                reward += -10
                is_terminal = True
                return reward, is_terminal
        for x in self.board.T:
            if np.array_equal(x, np.ones(3)):
                reward += 10
                is_terminal = True
                return reward, is_terminal
            elif np.array_equal(x, np.full(3, 2)):
                reward += -10
                is_terminal = True
                return reward, is_terminal
        if np.array_equal(np.diagonal(self.board), np.full(3, 1)) or np.array_equal(np.fliplr(self.board).diagonal(),
                                                                                    np.full(3, 1)):
            reward += 10
            is_terminal = True
            return reward, is_terminal
        elif np.array_equal(np.diagonal(self.board), np.full(3, 2)) or np.array_equal(np.fliplr(self.board).diagonal(),
                                                                                      np.full(3, 2)):
            reward += -10
            is_terminal = True
            return reward, is_terminal
        return reward, is_terminal
