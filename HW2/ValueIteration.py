#!/usr/bin/env python
""" ValueIteration.py

This program performs Value Iteration

Class: STAT 232C - Cognitive Artificial Intelligence
Project 2: Value Iteration
Name: Zachary Lacey
Date: April 14th, 2021

"""

__author__ = 'Zachary Lacey'

import numpy as np


class ValueIteration(object):
    """ A ValueIteration Class """

    def __init__(self, transitionTable, rewardTable, valueTable, convergenceTolerance, gamma=1, use_softmax=False,
                 use_noise=False, noise_beta=0):
        """
        Args:
            transitionTable (datatype: dict): This is a nested dictionary of the state-action-nextstate combinations
            for transitions that have a non-zero probability. The transitionTable has the following structure:

                transitionTable = {state: {action: {nextstate: {probability}}}}  == P(s' | s, a)

                Example:
                {(0, 0): {(1, 0): {(1, 0): 0.7, (0, 1): 0.2, (0, 0): 0.1}}
                {state (0, 0) : action (1, 0):
                    nextstate (1, 0) [move right]: probability 0.7
                    nextstate (0, 1) [move up]: probability 0.2
                    nextstate (0, 0) [no movement]: probability 0.1}

            rewardTable (datatype: dict): This is a nested dictionary of form state-action-nextstate combinations for
            each corresponding deterministic reward function. The rewardTable has the following structure:

                rewardTable = {state: {action: {nextstate: {reward}}}} == R(s', s, a)

            valueTable (datatype: dict): This is a nested dictionary of form state-value that specifies the initialized
            values of each state. All state values are initialized to zero. The valueTable has the following structure:

                valueTable = {state: {value}} == V(s)

            convergenceTolerance (datatype: float): A small scalar threshold determining how much error is tolerated
            in the estimation.

            gamma (datatype: float): Discount factor

            use_softmax (datatype: boolean): This is a boolean flag that the user can specify if they want to solve the
            Policy via the Boltzmann Policy (Softmax) rather than by using the Argmax approach.

        """
        self.transitionTable = transitionTable
        self.rewardTable = rewardTable
        self.valueTable = valueTable
        self.convergenceTolerance = convergenceTolerance
        self.gamma = gamma

        self.use_softmax = use_softmax
        self.use_noise = use_noise
        self.noise_beta = noise_beta

    def __call__(self):
        """
        Returns:
            state_values (datatype: dict): This is a nested dictionary of form state-value that specifies the
            state-values of the optimal policy. All state values are initialized to zero. The state_values has the
            following structure:

                state_values = {state: {value}} == V(s)

            policyTable (datatype: dict): This is a nested dictionary of form state-action-probability giving the
            approximate optimal policy of an agent.

                policyTable = {state: {action: {probability}}} == pi(a | s)

        """
        iteration = 0
        state_values_init = self.valueTable.copy()

        stateValues, policyTable, iteration = ValueIteration.value_iteration(self, state_values_init, iteration)

        return ([stateValues, policyTable])

    def value_iteration(self, state_values, iteration):
        """
        Value Iteration Method

        This method performs the Value Iteration Algorithm for a 2D Grid World.

        Args:
            state_values (datatype: dict): This is a nested dictionary of form state-value that specifies the
            state-values of the optimal policy. All state values are initialized to zero. This input is the initialized
            state-values structure with initialized zeros. The state_values has the following structure:

                state_values = {state: {value}} == V(s)

            iteration (datatype: integer): This specifies the number of iteration that Value Iteration performs.

        Returns:
            state_values (datatype: dict): This is a nested dictionary of form state-value that specifies the
            state-values of the optimal policy. The state_values has the following structure:

                state_values = {state: {value}} == V(s)

            policy_table (datatype: dict): This is a nested dictionary of form state-action-probability giving the
            approximate optimal policy of an agent.

                policy_table = {state: {action: {probability}}} == pi(a | s)

            iteration (datatype: integer): This specifies the number of iteration that Value Iteration performs.

        """
        while True:
            delta = 0.0

            # Initialize a the current and previous state-value tables as copies of the initial state value table.
            state_values_curr = state_values.copy()
            state_values_prev = state_values.copy()

            # Dictionary to store possible state-action values
            state_action_value_mem = dict()

            states = list(self.transitionTable)
            num_states = len(states)

            # For each state
            for s in range(num_states):

                value_scalar = state_values_prev[states[s]]

                actions = list(self.transitionTable[states[s]])
                num_actions = len(actions)

                state_value_sum_per_action = np.zeros((num_actions, 1))
                state_value_sum_per_action_mem = dict()

                # For each action
                for a in range(num_actions):
                    state_prime = list(self.transitionTable[states[s]][actions[a]])
                    num_states_prime = len(state_prime)

                    state_values_curr[states[s]] = np.sum([
                        self.transitionTable[states[s]][actions[a]][state_prime[sp]] * (
                                self.rewardTable[states[s]][actions[a]][state_prime[sp]] +
                                self.gamma * state_values_prev[state_prime[sp]]
                        ) for sp in range(num_states_prime)])

                    state_value_sum_per_action[a] = state_values_curr[states[s]]
                    state_value_sum_per_action_mem[actions[a]] = state_values_curr[states[s]]

                state_action_value_mem[states[s]] = state_value_sum_per_action_mem
                max_state_value = np.max(state_value_sum_per_action)

                # Update the state-value
                state_values[states[s]] = max_state_value

                delta = max(delta, abs(value_scalar - state_values[states[s]]))

            iteration += 1

            if delta < self.convergenceTolerance:
                break

        policy_table = ValueIteration.optimal_policy(self, state_action_value_mem)

        state_values = ValueIteration.optimal_value(self, policy_table, state_action_value_mem)

        return [state_values, policy_table, iteration]

    def optimal_policy(self, state_action_value_mem):
        """ Optimal Policy Method

        Args:
            state_action_value_mem (datatype: dict): This is a nested dictionary of form state-action-value which
            contains the stored state-action values (aka, Q-Values) for the optimal policy. The state_action_value_mem
            has the following structure:

                state_action_value_mem = {state: {action: {value}}} == Q(s, a)

        Returns:
            policy_table (datatype: dict): This is a nested dictionary of form state-action-probability giving the
            approximate optimal policy of an agent.

                policy_table = {state: {action: {probability}}} == pi(a | s)

        """
        policy_table = dict()

        states = list(self.transitionTable)
        num_states = len(states)

        state_values = dict()

        for s in range(num_states):
            actions = list(self.transitionTable[states[s]])
            num_actions = len(actions)

            state_value_sum_per_action = np.zeros((num_actions, 1))

            for a in range(num_actions):
                state_value_sum_per_action[a] = state_action_value_mem[states[s]][actions[a]]

            if self.use_softmax:
                softmax_policy_actions = dict()
                policy_actions_table = np.zeros((num_actions, 1))
                for a in range(num_actions):
                    policy_actions_table[a] = np.exp(self.noise_beta * state_action_value_mem[states[s]][actions[a]])
                    softmax_policy_actions[actions[a]] = policy_actions_table[a][0]

                policy_actions_sum = np.sum(policy_actions_table)

                for a in range(num_actions):
                    softmax_policy_actions[actions[a]] /= policy_actions_sum

                policy_table[states[s]] = softmax_policy_actions

            else:
                max_state_value = np.max(state_value_sum_per_action)
                max_state_idx = np.argwhere(state_value_sum_per_action.flatten() == max_state_value)
                num_max_values = len(max_state_idx)

                max_actions = dict()
                for a in range(num_actions):
                    if a in max_state_idx:
                        max_actions[actions[a]] = 1.0 / num_max_values

                policy_table[states[s]] = max_actions

        return policy_table

    def optimal_value(self, optimal_policy, state_action_value_mem):
        """ Optimal Value Method

        This method calculates the optimal state-value based on the optimal policy.

        Args:
            optimal_policy (datatype: dict): This is a nested dictionary of form state-action-probability giving the
            approximate optimal policy of an agent.

                optimal_policy = {state: {action: {probability}}} == pi(a | s)

            state_action_value_mem (datatype: dict): This is a nested dictionary of form state-action-value which
            contains the stored state-action values (aka, Q-Values) for the optimal policy. The state_action_value_mem
            has the following structure:

                state_action_value_mem = {state: {action: {value}}} == Q(s, a)

        Returns:
            state_values (datatype: dict): This is a nested dictionary of form state-value that specifies the
            state-values of the optimal policy. The state_values has the following structure:

                state_values = {state: {value}} == V(s)

        """
        states = list(self.transitionTable)
        num_states = len(states)

        state_values = dict()

        # For each state
        for s in range(num_states):
            actions = list(optimal_policy[states[s]])
            num_actions = len(actions)

            state_values[states[s]] = np.sum([optimal_policy[states[s]][actions[a]]
                                              * state_action_value_mem[states[s]][actions[a]]
                                              for a in range(num_actions)])
        return state_values


def viewDictionaryStructure(d, levels, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": " + str(key))
        if isinstance(value, dict):
            viewDictionaryStructure(value, levels, indent + 1)
        else:
            print('\t' * (indent + 1) + str(levels[indent + 1]) + ": " + str(value))


def main():
    """
    Example 1: Deterministic Transition
    When transitions are deterministic, the optimal policy is always to take the action or actions that move you closer
    to the goal state while avoiding the trap.
    """

    transitionTableDet = {
        (0, 0): {(1, 0): {(1, 0): 1}, (0, 1): {(0, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(0, 0): 1}},
        (0, 1): {(1, 0): {(1, 1): 1}, (0, 1): {(0, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(0, 0): 1}},
        (0, 2): {(1, 0): {(1, 2): 1}, (0, 1): {(0, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(0, 1): 1}},
        (0, 3): {(1, 0): {(1, 3): 1}, (0, 1): {(0, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(0, 2): 1}},
        (0, 4): {(1, 0): {(1, 4): 1}, (0, 1): {(0, 4): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(0, 3): 1}},
        (1, 0): {(1, 0): {(2, 0): 1}, (0, 1): {(1, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(1, 0): 1}},
        (1, 1): {(1, 0): {(2, 1): 1}, (0, 1): {(1, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(1, 0): 1}},
        (1, 2): {(1, 0): {(2, 2): 1}, (0, 1): {(1, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(1, 1): 1}},
        (1, 3): {(1, 0): {(2, 3): 1}, (0, 1): {(1, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(1, 2): 1}},
        (1, 4): {(1, 0): {(2, 4): 1}, (0, 1): {(1, 4): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(1, 3): 1}},
        (2, 0): {(1, 0): {(2, 0): 1}, (0, 1): {(2, 1): 1}, (-1, 0): {(1, 0): 1}, (0, -1): {(2, 0): 1}},
        (2, 1): {(1, 0): {(2, 1): 1}, (0, 1): {(2, 2): 1}, (-1, 0): {(1, 1): 1}, (0, -1): {(2, 0): 1}},
        (2, 2): {(1, 0): {(2, 2): 1}, (0, 1): {(2, 3): 1}, (-1, 0): {(1, 2): 1}, (0, -1): {(2, 1): 1}},
        (2, 3): {(1, 0): {(2, 3): 1}, (0, 1): {(2, 4): 1}, (-1, 0): {(1, 3): 1}, (0, -1): {(2, 2): 1}},
        (2, 4): {(1, 0): {(2, 4): 1}, (0, 1): {(2, 4): 1}, (-1, 0): {(1, 4): 1}, (0, -1): {(2, 3): 1}}
    }

    rewardTableDet = {
        (0, 0): {(1, 0): {(1, 0): -1}, (0, 1): {(0, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(0, 0): -1}},
        (0, 1): {(1, 0): {(1, 1): -1}, (0, 1): {(0, 2): -1}, (-1, 0): {(0, 1): -1}, (0, -1): {(0, 0): -1}},
        (0, 2): {(1, 0): {(1, 2): -1}, (0, 1): {(0, 3): -1}, (-1, 0): {(0, 2): -1}, (0, -1): {(0, 1): -1}},
        (0, 3): {(1, 0): {(1, 3): -1}, (0, 1): {(0, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(0, 2): -1}},
        (0, 4): {(1, 0): {(1, 4): -1}, (0, 1): {(0, 4): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(0, 3): -1}},
        (1, 0): {(1, 0): {(2, 0): -1}, (0, 1): {(1, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(1, 0): -1}},
        (1, 1): {(1, 0): {(2, 1): 10}, (0, 1): {(1, 2): 10}, (-1, 0): {(0, 1): 10}, (0, -1): {(1, 0): 10}},
        (1, 2): {(1, 0): {(2, 2): -100}, (0, 1): {(1, 3): -100}, (-1, 0): {(0, 2): -100}, (0, -1): {(1, 1): -100}},
        (1, 3): {(1, 0): {(2, 3): -1}, (0, 1): {(1, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(1, 2): -1}},
        (1, 4): {(1, 0): {(2, 4): -1}, (0, 1): {(1, 4): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(1, 3): -1}},
        (2, 0): {(1, 0): {(2, 0): -1}, (0, 1): {(2, 1): -1}, (-1, 0): {(1, 0): -1}, (0, -1): {(2, 0): -1}},
        (2, 1): {(1, 0): {(2, 1): -1}, (0, 1): {(2, 2): -1}, (-1, 0): {(1, 1): -1}, (0, -1): {(2, 0): -1}},
        (2, 2): {(1, 0): {(2, 2): -1}, (0, 1): {(2, 3): -1}, (-1, 0): {(1, 2): -1}, (0, -1): {(2, 1): -1}},
        (2, 3): {(1, 0): {(2, 3): -1}, (0, 1): {(2, 4): -1}, (-1, 0): {(1, 3): -1}, (0, -1): {(2, 2): -1}},
        (2, 4): {(1, 0): {(2, 4): -1}, (0, 1): {(2, 4): -1}, (-1, 0): {(1, 4): -1}, (0, -1): {(2, 3): -1}}
    }

    valueTableDet = {
        (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0,
        (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0,
        (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 0}

    convergenceTolerance = 10e-7
    gamma = .9

    """
    Example 2: Probabilistic Transition
    """
    transitionTable = {
        (0, 0): {
            (1, 0): {(1, 0): 0.7, (0, 1): 0.2, (0, 0): 0.1},
            (0, 1): {(0, 1): 0.7999999999999999, (1, 0): 0.2},
            (-1, 0): {(0, 0): 0.7, (1, 0): 0.2, (0, 1): 0.1},
            (0, -1): {(0, 0): 0.7, (1, 0): 0.1, (0, 1): 0.2}},
        (0, 1): {
            (1, 0): {(1, 1): 0.7999999999999999, (0, 1): 0.1, (0, 2): 0.1},
            (0, 1): {(0, 2): 0.7999999999999999, (0, 0): 0.2},
            (-1, 0): {(0, 1): 0.8999999999999999, (0, 0): 0.1},
            (0, -1): {(0, 0): 0.7999999999999999, (0, 2): 0.1, (0, 1): 0.1}},
        (0, 2): {
            (1, 0): {(1, 2): 0.7999999999999999, (0, 1): 0.2},
            (0, 1): {(0, 3): 0.7999999999999999, (0, 1): 0.1, (1, 2): 0.1},
            (-1, 0): {(0, 2): 0.7, (0, 1): 0.1, (1, 2): 0.1, (0, 3): 0.1},
            (0, -1): {(0, 1): 0.8999999999999999, (0, 3): 0.1}},
        (0, 3): {
            (1, 0): {(1, 3): 0.8999999999999999, (0, 2): 0.1},
            (0, 1): {(0, 3): 0.9999999999999999},
            (-1, 0): {(0, 3): 0.7999999999999999, (0, 2): 0.1, (1, 3): 0.1},
            (0, -1): {(0, 2): 0.7999999999999999, (0, 3): 0.2}},
        (1, 0): {
            (1, 0): {(2, 0): 0.8999999999999999, (1, 1): 0.1},
            (0, 1): {(1, 1): 0.8999999999999999, (1, 0): 0.1},
            (-1, 0): {(0, 0): 0.7, (1, 1): 0.2, (2, 0): 0.1},
            (0, -1): {(1, 0): 0.7999999999999999, (0, 0): 0.2}},
        (1, 1): {
            (1, 0): {(2, 1): 0.7999999999999999, (1, 0): 0.1, (0, 1): 0.1},
            (0, 1): {(1, 2): 0.7, (2, 1): 0.30000000000000004},
            (-1, 0): {(0, 1): 0.7, (2, 1): 0.1, (1, 0): 0.2},
            (0, -1): {(1, 0): 0.7999999999999999, (0, 1): 0.1, (2, 1): 0.1}},
        (1, 2): {
            (1, 0): {(2, 2): 0.7999999999999999, (1, 3): 0.1, (1, 1): 0.1},
            (0, 1): {(1, 3): 0.8999999999999999, (2, 2): 0.1},
            (-1, 0): {(0, 2): 0.8999999999999999, (1, 1): 0.1},
            (0, -1): {(1, 1): 0.7999999999999999, (2, 2): 0.1, (0, 2): 0.1}},
        (1, 3): {
            (1, 0): {(2, 3): 0.7999999999999999, (1, 3): 0.2},
            (0, 1): {(1, 3): 0.7999999999999999, (2, 3): 0.1, (0, 3): 0.1},
            (-1, 0): {(0, 3): 0.7, (2, 3): 0.1, (1, 2): 0.2},
            (0, -1): {(1, 2): 0.7999999999999999, (0, 3): 0.2}},
        (2, 0): {
            (1, 0): {(3, 0): 0.8999999999999999, (2, 0): 0.1},
            (0, 1): {(2, 1): 0.7999999999999999, (3, 0): 0.1, (1, 0): 0.1},
            (-1, 0): {(1, 0): 0.7, (2, 0): 0.2, (2, 1): 0.1},
            (0, -1): {(2, 0): 0.7, (2, 1): 0.2, (1, 0): 0.1}},
        (2, 1): {
            (1, 0): {(3, 1): 0.7999999999999999, (1, 1): 0.2},
            (0, 1): {(2, 2): 0.7, (1, 1): 0.1, (3, 1): 0.2},
            (-1, 0): {(1, 1): 0.7, (2, 0): 0.1, (2, 2): 0.1, (3, 1): 0.1},
            (0, -1): {(2, 0): 0.7, (1, 1): 0.2, (3, 1): 0.1}},
        (2, 2): {
            (1, 0): {(3, 2): 0.7, (1, 2): 0.1, (2, 1): 0.2},
            (0, 1): {(2, 3): 0.7999999999999999, (2, 1): 0.2},
            (-1, 0): {(1, 2): 0.7999999999999999, (2, 1): 0.1, (3, 2): 0.1},
            (0, -1): {(2, 1): 0.7999999999999999, (1, 2): 0.1, (3, 2): 0.1}},
        (2, 3): {
            (1, 0): {(3, 3): 0.7, (2, 3): 0.2, (2, 2): 0.1},
            (0, 1): {(2, 3): 0.7999999999999999, (2, 2): 0.1, (3, 3): 0.1},
            (-1, 0): {(1, 3): 0.8999999999999999, (2, 3): 0.1},
            (0, -1): {(2, 2): 0.7, (3, 3): 0.1, (1, 3): 0.1, (2, 3): 0.1}},
        (3, 0): {
            (1, 0): {(3, 0): 0.7, (3, 1): 0.1, (2, 0): 0.2},
            (0, 1): {(3, 1): 0.7999999999999999, (2, 0): 0.2},
            (-1, 0): {(2, 0): 0.7999999999999999, (3, 0): 0.2},
            (0, -1): {(3, 0): 0.7999999999999999, (2, 0): 0.1, (3, 1): 0.1}},
        (3, 1): {
            (1, 0): {(3, 1): 0.8999999999999999, (3, 2): 0.1},
            (0, 1): {(3, 2): 0.7, (2, 1): 0.2, (3, 0): 0.1},
            (-1, 0): {(2, 1): 0.7999999999999999, (3, 0): 0.1, (3, 1): 0.1},
            (0, -1): {(3, 0): 0.7999999999999999, (2, 1): 0.2}},
        (3, 2): {
            (1, 0): {(3, 2): 0.7999999999999999, (3, 1): 0.1, (2, 2): 0.1},
            (0, 1): {(3, 3): 0.7, (3, 2): 0.2, (2, 2): 0.1},
            (-1, 0): {(2, 2): 0.9999999999999999},
            (0, -1): {(3, 1): 0.7999999999999999, (3, 3): 0.1, (3, 2): 0.1}},
        (3, 3): {
            (1, 0): {(3, 3): 0.7999999999999999, (3, 2): 0.2},
            (0, 1): {(3, 3): 0.7999999999999999, (3, 2): 0.2},
            (-1, 0): {(2, 3): 0.7999999999999999, (3, 2): 0.1, (3, 3): 0.1},
            (0, -1): {(3, 2): 0.7999999999999999, (2, 3): 0.2}}}

    rewardTable = {
        (0, 0): {
            (1, 0): {(1, 0): -1, (0, 1): -1, (0, 0): -1}, (0, 1): {(0, 1): -1, (1, 0): -1},
            (-1, 0): {(0, 0): -1, (1, 0): -1, (0, 1): -1}, (0, -1): {(0, 0): -1, (1, 0): -1, (0, 1): -1}},
        (0, 1): {
            (1, 0): {(1, 1): -1, (0, 1): -1, (0, 2): -1}, (0, 1): {(0, 2): -1, (0, 0): -1},
            (-1, 0): {(0, 1): -1, (0, 0): -1}, (0, -1): {(0, 0): -1, (0, 2): -1, (0, 1): -1}},
        (0, 2): {
            (1, 0): {(1, 2): -1, (0, 1): -1}, (0, 1): {(0, 3): -1, (0, 1): -1, (1, 2): -1},
            (-1, 0): {(0, 2): -1, (0, 1): -1, (1, 2): -1, (0, 3): -1}, (0, -1): {(0, 1): -1, (0, 3): -1}},
        (0, 3): {
            (1, 0): {(1, 3): -1, (0, 2): -1}, (0, 1): {(0, 3): -1},
            (-1, 0): {(0, 3): -1, (0, 2): -1, (1, 3): -1}, (0, -1): {(0, 2): -1, (0, 3): -1}
        },
        (1, 0): {
            (1, 0): {(2, 0): -1, (1, 1): -1}, (0, 1): {(1, 1): -1, (1, 0): -1},
            (-1, 0): {(0, 0): -1, (1, 1): -1, (2, 0): -1}, (0, -1): {(1, 0): -1, (0, 0): -1}
        },
        (1, 1): {
            (1, 0): {(2, 1): -100, (1, 0): -100, (0, 1): -100}, (0, 1): {(1, 2): -100, (2, 1): -100},
            (-1, 0): {(0, 1): -100, (2, 1): -100, (1, 0): -100}, (0, -1): {(1, 0): -100, (0, 1): -100, (2, 1): -100}
        },
        (1, 2): {
            (1, 0): {(2, 2): -1, (1, 3): -1, (1, 1): -1}, (0, 1): {(1, 3): -1, (2, 2): -1},
            (-1, 0): {(0, 2): -1, (1, 1): -1}, (0, -1): {(1, 1): -1, (2, 2): -1, (0, 2): -1}
        },
        (1, 3): {
            (1, 0): {(2, 3): -1, (1, 3): -1}, (0, 1): {(1, 3): -1, (2, 3): -1, (0, 3): -1},
            (-1, 0): {(0, 3): -1, (2, 3): -1, (1, 2): -1}, (0, -1): {(1, 2): -1, (0, 3): -1}
        },
        (2, 0): {
            (1, 0): {(3, 0): -1, (2, 0): -1}, (0, 1): {(2, 1): -1, (3, 0): -1, (1, 0): -1},
            (-1, 0): {(1, 0): -1, (2, 0): -1, (2, 1): -1}, (0, -1): {(2, 0): -1, (2, 1): -1, (1, 0): -1}
        },
        (2, 1): {
            (1, 0): {(3, 1): -1, (1, 1): -1}, (0, 1): {(2, 2): -1, (1, 1): -1, (3, 1): -1},
            (-1, 0): {(1, 1): -1, (2, 0): -1, (2, 2): -1, (3, 1): -1}, (0, -1): {(2, 0): -1, (1, 1): -1, (3, 1): -1}
        },
        (2, 2): {
            (1, 0): {(3, 2): -1, (1, 2): -1, (2, 1): -1}, (0, 1): {(2, 3): -1, (2, 1): -1},
            (-1, 0): {(1, 2): -1, (2, 1): -1, (3, 2): -1}, (0, -1): {(2, 1): -1, (1, 2): -1, (3, 2): -1}
        },
        (2, 3): {
            (1, 0): {(3, 3): -1, (2, 3): -1, (2, 2): -1}, (0, 1): {(2, 3): -1, (2, 2): -1, (3, 3): -1},
            (-1, 0): {(1, 3): -1, (2, 3): -1}, (0, -1): {(2, 2): -1, (3, 3): -1, (1, 3): -1, (2, 3): -1}
        },
        (3, 0): {
            (1, 0): {(3, 0): -1, (3, 1): -1, (2, 0): -1}, (0, 1): {(3, 1): -1, (2, 0): -1},
            (-1, 0): {(2, 0): -1, (3, 0): -1}, (0, -1): {(3, 0): -1, (2, 0): -1, (3, 1): -1}
        },
        (3, 1): {
            (1, 0): {(3, 1): -1, (3, 2): 10}, (0, 1): {(3, 2): 10, (2, 1): 10, (3, 0): 10},
            (-1, 0): {(2, 1): 10, (3, 0): 10, (3, 1): -1}, (0, -1): {(3, 0): 10, (2, 1): 10}
        },
        (3, 2): {
            (1, 0): {(3, 2): -1, (3, 1): -1, (2, 2): -1}, (0, 1): {(3, 3): -1, (3, 2): -1, (2, 2): -1},
            (-1, 0): {(2, 2): -1}, (0, -1): {(3, 1): -1, (3, 3): -1, (3, 2): -1}},
        (3, 3): {
            (1, 0): {(3, 3): -1, (3, 2): -1}, (0, 1): {(3, 3): -1, (3, 2): -1},
            (-1, 0): {(2, 3): -1, (3, 2): -1, (3, 3): -1}, (0, -1): {(3, 2): -1, (2, 3): -1}
        }
    }

    valueTable = {
        (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0,
        (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0,
        (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0,
        (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0}

    convergenceTolerance = 10e-7
    gamma = .9

    """
        Uncomment to view transition or reward structure in a readable format

        levelsReward  = ["state", "action", "next state", "reward"]
        levelsTransition  = ["state", "action", "next state", "probability"]

        viewDictionaryStructure(transition, levelsTransition)
        viewDictionaryStructure(reward, levelsReward)
        """

    performValueIteration = ValueIteration(transitionTableDet, rewardTableDet, valueTableDet, convergenceTolerance,
                                           gamma)
    optimalValuesDeterminsitic, policyTableDet = performValueIteration()
    print(optimalValuesDeterminsitic)
    print(policyTableDet)

    performValueIteration = ValueIteration(transitionTable, rewardTable, valueTable, convergenceTolerance, gamma)
    optimalValuesDeterminsitic, policyTable = performValueIteration()
    print(optimalValuesDeterminsitic)
    print(policyTable)


if __name__ == '__main__':
    main()
