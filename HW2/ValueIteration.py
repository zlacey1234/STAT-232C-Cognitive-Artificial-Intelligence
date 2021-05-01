#!/usr/bin/env python
""" ValueIteration.py

This program performs Value Iteration

Course: STAT 232C: Cognitive Artificial Intelligence
Author: Zachary Lacey
Date: April 14th, 2021
"""

__author__ = 'Zachary Lacey'

import numpy as np


class ValueIteration(object):
	"""
	A ValueIteration Class which

	valueIteration = (transti

	"""
	def __init__(self, transitionTable, rewardTable, valueTable, convergenceTolerance, gamma):
		"""
		Args:
			transitionTable: This is a nested dictionary of the state-action-nextstate combinations for transitions that
			have a non-zero probability. The transitionTable has the following structure:

				transitionTable = {state: {action: {nextstate: {probability}}}}  == P(s' | s, a)

				Example:
				{(0, 0): {(1, 0): {(1, 0): 0.7, (0, 1): 0.2, (0, 0): 0.1}}
				{state (0, 0) : action (1, 0):
					nextstate (1, 0) [move right]: probability 0.7
					nextstate (0, 1) [move up]: probability 0.2
					nextstate (0, 0) [no movement]: probability 0.1}

			rewardTable: This is a nested dictionary of form state-action-nextstate combinations for each corresponding
			deterministic reward function. The rewardTable has the following structure:

				rewardTable = {state: {action: {nextstate: {reward}}}} == R(s', s, a)

			valueTable: This is a nested dictionary of form state-value that specifies the initialized values of each
			state. All state values are initialized to zero. The valueTable has the following structure:

				valueTable = {state: {value}} == V(s)


			convergenceTolerance:
			gamma:
		"""
		self.transitionTable = transitionTable
		self.rewardTable = rewardTable
		self.valueTable = valueTable
		self.convergenceTolerance = convergenceTolerance
		self.gamma = gamma

	def __call__(self):
		"""
		Returns:
			stateValues (datatype: dict): This is a nested dictionary of form state-value that specifies the
			state-values of the optimal policy. All state values are initialized to zero. The stateValues has the
			following structure:

				stateValues = {state: {value}} == V(s)

			policyTable (datatype: dict): This is a nested dictionary of form state-action-probability giving the
			approximate optimal policy of an agent.

				policyTable = {state: {action: {probability}}} == pi(a | s)
		"""
		iteration = 0
		stateValues = self.valueTable.copy()

		policyTable = self.valueTable.copy()

		# terminalStateTable = {(3, 2), (3, 1)}

		stateValues, policyTable, iteration = ValueIteration.value_iteration(
			self, stateValues, policyTable, iteration)

		print('iteration')
		print(iteration)

		return [stateValues, policyTable]

	def value_iteration(
			self, state_values, policy_table, iteration, use_terminal_state_mode=False, terminal_states_table=None,
			max_iter=-1):
		"""
		Value Iteration Method

		This method performs the Value Iteration Algorithm for a 2D Grid World.

		Args:
			state_values (datatype: dict): This is a nested dictionary of form state-value that specifies the
			state-values of the optimal policy. All state values are initialized to zero. The state_values has the
			following structure:

				state_values = {state: {value}} == V(s)

			policy_table (datatype: dict): This is a nested dictionary of form state-action-probability giving the
			approximate optimal policy of an agent.

				policy_table = {state: {action: {probability}}} == pi(a | s)

			iteration (datatype: integer): This specifies the number of iteration that Value Iteration performs.

			use_terminal_state_mode (datatype: boolean): This allows the user to specify is there are terminal states.
			This is useful if terminal states exist since the Value of Terminal States are set to zero.

			terminal_states_table (datatype: dict): This is a dictionary that contains the specified terminal states.
			The terminal_states_table has the following structure:

				terminal_states_table = {states}

			max_iter (datatype: integer): This allows the user to specify a set maximum iteration index. This is useful
			if the user wants to break out of the while loop at a specific iteration instead of until convergence.

		"""
		while True:
			delta = 0.0

			state_values_prev = state_values.copy()

			states = list(self.transitionTable)
			num_states = len(states)

			terminal_states = None

			if use_terminal_state_mode:
				terminal_states = list(terminal_states_table)

			# print('iteration')
			# print(iteration)

			for s in range(num_states):

				value_scalar = state_values_prev[states[s]]

				# print('state-value scalar')
				# print(states[s])
				# print(value_scalar)

				actions = list(self.transitionTable[states[s]])
				num_actions = len(actions)

				state_value_sum_per_action = np.zeros((num_actions, 1))

				for a in range(num_actions):
					state_prime = list(self.transitionTable[states[s]][actions[a]])
					num_states_prime = len(state_prime)

					if use_terminal_state_mode:
						if states[s] in terminal_states:
							for sp in range(num_states_prime):
								print('Yes')
								state_values_prev[state_prime[sp]] = 0.0

					# print('a')
					# print(actions[a])

					state_values[states[s]] = np.sum([
						self.transitionTable[states[s]][actions[a]][state_prime[sp]] * (
								self.rewardTable[states[s]][actions[a]][state_prime[sp]] +
								self.gamma * state_values_prev[state_prime[sp]]
						) for sp in range(num_states_prime)])

					state_value_sum_per_action[a] = state_values[states[s]]
					# print(states[s])
					# print(state_values[states[s]])
					# print('---------------------')
					# print(state_value_sum_per_action[a])

				max_state_value = np.max(state_value_sum_per_action)
				state_values[states[s]] = max_state_value
				max_state_idx = np.argwhere(state_value_sum_per_action.flatten() == max_state_value)
				num_max_values = len(max_state_idx)

				max_actions = dict()
				for a in range(num_actions):
					if a in max_state_idx:
						# print('Max Action')
						max_actions[actions[a]] = 1.0 / num_max_values
						# print(actions[a])

				policy_table[states[s]] = max_actions

				# print('======================')
				# print(state_value_sum_per_action)
				# print(max_state_value)
				# print(max_state_idx)
				# print(num_max_values)
				# print(max_actions)
				# print(state_values)
				#
				# print('\n\n')
				# print(states[s])
				# print('Policy')
				# print(policy_table[states[s]])
				#
				# print(policy_table)

				delta = max(delta, abs(value_scalar - state_values[states[s]]))
				# print('delta')
				# print(delta)
				# print('\n\n')

			iteration += 1

			if max_iter != -1:
				if iteration >= max_iter:
					break

			if delta < self.convergenceTolerance:
				break

		# print('---------------------')
		# print(state_value_sum_per_action)
		# policy_table = ValueIteration.optimal_policy(self, state_values, policy_table)

		return [state_values, policy_table, iteration]


def viewDictionaryStructure(d, levels, indent=0):
	for key, value in d.items():
		print('\t' * indent + str(levels[indent]) + ": "+ str(key))
		if isinstance(value, dict):
			viewDictionaryStructure(value, levels, indent+1)
		else:
			print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))


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
			(0, -1): {(0, 0): 0.7, (1, 0): 0.1, (0, 1): 0.2}
		},
		(0, 1): {
			(1, 0): {(1, 1): 0.7999999999999999, (0, 1): 0.1, (0, 2): 0.1},
			(0, 1): {(0, 2): 0.7999999999999999, (0, 0): 0.2},
			(-1, 0): {(0, 1): 0.8999999999999999, (0, 0): 0.1},
			(0, -1): {(0, 0): 0.7999999999999999, (0, 2): 0.1, (0, 1): 0.1}
		},
		(0, 2): {
			(1, 0): {(1, 2): 0.7999999999999999, (0, 1): 0.2},
			(0, 1): {(0, 3): 0.7999999999999999, (0, 1): 0.1, (1, 2): 0.1},
			(-1, 0): {(0, 2): 0.7, (0, 1): 0.1, (1, 2): 0.1, (0, 3): 0.1},
			(0, -1): {(0, 1): 0.8999999999999999, (0, 3): 0.1}
		},
		(0, 3): {
			(1, 0): {(1, 3): 0.8999999999999999, (0, 2): 0.1},
			(0, 1): {(0, 3): 0.9999999999999999},
			(-1, 0): {(0, 3): 0.7999999999999999, (0, 2): 0.1, (1, 3): 0.1},
			(0, -1): {(0, 2): 0.7999999999999999, (0, 3): 0.2}
		},
		(1, 0): {
			(1, 0): {(2, 0): 0.8999999999999999, (1, 1): 0.1},
			(0, 1): {(1, 1): 0.8999999999999999, (1, 0): 0.1},
			(-1, 0): {(0, 0): 0.7, (1, 1): 0.2, (2, 0): 0.1},
			(0, -1): {(1, 0): 0.7999999999999999, (0, 0): 0.2}
		},
		(1, 1): {
			(1, 0): {(2, 1): 0.7999999999999999, (1, 0): 0.1, (0, 1): 0.1},
			(0, 1): {(1, 2): 0.7, (2, 1): 0.30000000000000004},
			(-1, 0): {(0, 1): 0.7, (2, 1): 0.1, (1, 0): 0.2},
			(0, -1): {(1, 0): 0.7999999999999999, (0, 1): 0.1, (2, 1): 0.1}
		},
		(1, 2): {
			(1, 0): {(2, 2): 0.7999999999999999, (1, 3): 0.1, (1, 1): 0.1},
			(0, 1): {(1, 3): 0.8999999999999999, (2, 2): 0.1},
			(-1, 0): {(0, 2): 0.8999999999999999, (1, 1): 0.1},
			(0, -1): {(1, 1): 0.7999999999999999, (2, 2): 0.1, (0, 2): 0.1}
		},
		(1, 3): {
			(1, 0): {(2, 3): 0.7999999999999999, (1, 3): 0.2},
			(0, 1): {(1, 3): 0.7999999999999999, (2, 3): 0.1, (0, 3): 0.1},
			(-1, 0): {(0, 3): 0.7, (2, 3): 0.1, (1, 2): 0.2},
			(0, -1): {(1, 2): 0.7999999999999999, (0, 3): 0.2}
		},
		(2, 0): {
			(1, 0): {(3, 0): 0.8999999999999999, (2, 0): 0.1},
			(0, 1): {(2, 1): 0.7999999999999999, (3, 0): 0.1, (1, 0): 0.1},
			(-1, 0): {(1, 0): 0.7, (2, 0): 0.2, (2, 1): 0.1},
			(0, -1): {(2, 0): 0.7, (2, 1): 0.2, (1, 0): 0.1}
		},
		(2, 1): {
			(1, 0): {(3, 1): 0.7999999999999999, (1, 1): 0.2},
			(0, 1): {(2, 2): 0.7, (1, 1): 0.1, (3, 1): 0.2},
			(-1, 0): {(1, 1): 0.7, (2, 0): 0.1, (2, 2): 0.1, (3, 1): 0.1},
			(0, -1): {(2, 0): 0.7, (1, 1): 0.2, (3, 1): 0.1}
		},
		(2, 2): {
			(1, 0): {(3, 2): 0.7, (1, 2): 0.1, (2, 1): 0.2},
			(0, 1): {(2, 3): 0.7999999999999999, (2, 1): 0.2},
			(-1, 0): {(1, 2): 0.7999999999999999, (2, 1): 0.1, (3, 2): 0.1},
			(0, -1): {(2, 1): 0.7999999999999999, (1, 2): 0.1, (3, 2): 0.1}},
		(2, 3): {
			(1, 0): {(3, 3): 0.7, (2, 3): 0.2, (2, 2): 0.1},
			(0, 1): {(2, 3): 0.7999999999999999, (2, 2): 0.1, (3, 3): 0.1},
			(-1, 0): {(1, 3): 0.8999999999999999, (2, 3): 0.1},
			(0, -1): {(2, 2): 0.7, (3, 3): 0.1, (1, 3): 0.1, (2, 3): 0.1}
		},
		(3, 0): {
			(1, 0): {(3, 0): 0.7, (3, 1): 0.1, (2, 0): 0.2},
			(0, 1): {(3, 1): 0.7999999999999999, (2, 0): 0.2},
			(-1, 0): {(2, 0): 0.7999999999999999, (3, 0): 0.2},
			(0, -1): {(3, 0): 0.7999999999999999, (2, 0): 0.1, (3, 1): 0.1}
		},
		(3, 1): {
			(1, 0): {(3, 1): 0.8999999999999999, (3, 2): 0.1},
			(0, 1): {(3, 2): 0.7, (2, 1): 0.2, (3, 0): 0.1},
			(-1, 0): {(2, 1): 0.7999999999999999, (3, 0): 0.1, (3, 1): 0.1},
			(0, -1): {(3, 0): 0.7999999999999999, (2, 1): 0.2}
		},
		(3, 2): {
			(1, 0): {(3, 2): 0.7999999999999999, (3, 1): 0.1, (2, 2): 0.1},
			(0, 1): {(3, 3): 0.7, (3, 2): 0.2, (2, 2): 0.1},
			(-1, 0): {(2, 2): 0.9999999999999999},
			(0, -1): {(3, 1): 0.7999999999999999, (3, 3): 0.1, (3, 2): 0.1}
		},
		(3, 3): {
			(1, 0): {(3, 3): 0.7999999999999999, (3, 2): 0.2},
			(0, 1): {(3, 3): 0.7999999999999999, (3, 2): 0.2},
			(-1, 0): {(2, 3): 0.7999999999999999, (3, 2): 0.1, (3, 3): 0.1},
			(0, -1): {(3, 2): 0.7999999999999999, (2, 3): 0.2}
		}
	}

	rewardTable = {
		(0, 0): {
			(1, 0): {(1, 0): -1, (0, 1): -1, (0, 0): -1}, (0, 1): {(0, 1): -1, (1, 0): -1},
			(-1, 0): {(0, 0): -1, (1, 0): -1, (0, 1): -1}, (0, -1): {(0, 0): -1, (1, 0): -1, (0, 1): -1}
		},
		(0, 1): {
			(1, 0): {(1, 1): -1, (0, 1): -1, (0, 2): -1}, (0, 1): {(0, 2): -1, (0, 0): -1},
			(-1, 0): {(0, 1): -1, (0, 0): -1}, (0, -1): {(0, 0): -1, (0, 2): -1, (0, 1): -1}
		},
		(0, 2): {
			(1, 0): {(1, 2): -1, (0, 1): -1}, (0, 1): {(0, 3): -1, (0, 1): -1, (1, 2): -1},
			(-1, 0): {(0, 2): -1, (0, 1): -1, (1, 2): -1, (0, 3): -1}, (0, -1): {(0, 1): -1, (0, 3): -1}
		},
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
	Book Example
	"""

	transitionTableBook = {
		(0, 0): {
			(1, 0): {(1, 0): 0.8, (0, 1): 0.1, (0, 0): 0.1},
			(0, 1): {(0, 1): 0.8, (1, 0): 0.1, (0, 0): 0.1},
			(-1, 0): {(0, 0): 0.9, (0, 1): 0.1},
			(0, -1): {(0, 0): 0.9, (1, 0): 0.1}},
		(0, 1): {
			(1, 0): {(1, 0): 0.8, (0, 0): 0.1, (0, 2): 0.1},
			(0, 1): {(0, 2): 0.8, (1, 0): 0.2},
			(-1, 0): {(1, 0): 0.8, (0, 0): 0.1, (0, 2): 0.1},
			(0, -1): {(0, 0): 0.8, (1, 0): 0.2}},
		(0, 2): {
			(1, 0): {(1, 2): 0.8, (0, 1): 0.1, (0, 2): 0.1},
			(0, 1): {(0, 2): 0.9, (1, 2): 0.1},
			(-1, 0): {(0, 2): 0.9, (0, 1): 0.1},
			(0, -1): {(0, 1): 0.8, (0, 2): 0.1, (1, 2): 0.1}},
		(1, 0): {
			(1, 0): {(2, 0): 0.8, (1, 0): 0.2},
			(0, 1): {(1, 0): 0.8, (0, 0): 0.1, (2, 0): 0.1},
			(-1, 0): {(0, 0): 0.8, (1, 0): 0.2},
			(0, -1): {(1, 0): 0.8, (0, 0): 0.1, (2, 0): 0.1}},
		(1, 2): {
			(1, 0): {(2, 2): 0.8, (1, 2): 0.2},
			(0, 1): {(1, 2): 0.8, (0, 2): 0.1, (2, 2): 0.1},
			(-1, 0): {(0, 0): 0.8, (1, 0): 0.2},
			(0, -1): {(1, 2): 0.8, (0, 2): 0.1, (2, 2): 0.1}},

		(2, 0): {
			(1, 0): {(3, 0): 0.8, (2, 1): 0.1, (2, 0): 0.1},
			(0, 1): {(2, 1): 0.8, (3, 0): 0.1, (1, 0): 0.1},
			(-1, 0): {(1, 0): 0.8, (2, 1): 0.1, (2, 0): 0.1},
			(0, -1): {(2, 0): 0.8, (1, 0): 0.1, (3, 0): 0.1}},
		(2, 1): {
			(1, 0): {(3, 1): 0.8, (2, 0): 0.1, (2, 2): 0.1},
			(0, 1): {(2, 2): 0.8, (3, 1): 0.1, (2, 1): 0.1},
			(-1, 0): {(2, 1): 0.8, (2, 0): 0.1, (2, 2): 0.1},
			(0, -1): {(2, 0): 0.8, (3, 1): 0.1, (2, 1): 0.1}},
		(2, 2): {
			(1, 0): {(3, 2): 0.8, (2, 1): 0.1, (2, 2): 0.1},
			(0, 1): {(2, 2): 0.8, (1, 2): 0.1, (3, 2): 0.1},
			(-1, 0): {(1, 2): 0.8, (2, 2): 0.1, (2, 1): 0.1},
			(0, -1): {(2, 1): 0.8, (1, 2): 0.1, (3, 2): 0.1}},

		(3, 0): {
			(1, 0): {(3, 0): 0.9, (3, 1): 0.1},
			(0, 1): {(3, 1): 0.8, (3, 0): 0.1, (2, 0): 0.1},
			(-1, 0): {(2, 0): 0.8, (3, 1): 0.1, (3, 0): 0.1},
			(0, -1): {(3, 0): 0.9, (2, 0): 0.1}},
		(3, 1): {
			(1, 0): {(3, 1): 0.8, (3, 0): 0.1, (3, 2): 0.1},
			(0, 1): {(3, 2): 0.8, (3, 1): 0.1, (2, 1): 0.1},
			(-1, 0): {(2, 1): 0.8, (3, 0): 0.1, (3, 2): 0.1},
			(0, -1): {(3, 0): 0.8, (2, 1): 0.1, (3, 1): 0.1}},
		(3, 2): {
			(1, 0): {(3, 2): 0.9, (3, 1): 0.1},
			(0, 1): {(3, 2): 0.9, (2, 2): 0.1},
			(-1, 0): {(2, 2): 0.8, (3, 1): 0.1, (3, 2): 0.1},
			(0, -1): {(3, 1): 0.8, (2, 2): 0.1, (3, 2): 0.1}}
	}

	rewardTableBook = {
		(0, 0): {
			(1, 0): {(1, 0): 0, (0, 1): 0, (0, 0): 0},
			(0, 1): {(0, 1): 0, (1, 0): 0, (0, 0): 0},
			(-1, 0): {(0, 0): 0, (0, 1): 0},
			(0, -1): {(0, 0): 0, (1, 0): 0}},
		(0, 1): {
			(1, 0): {(1, 0): 0, (0, 0): 0, (0, 2): 0},
			(0, 1): {(0, 2): 0, (1, 0): 0},
			(-1, 0): {(1, 0): 0, (0, 0): 0, (0, 2): 0},
			(0, -1): {(0, 0): 0, (1, 0): 0}},
		(0, 2): {
			(1, 0): {(1, 2): 0, (0, 1): 0, (0, 2): 0},
			(0, 1): {(0, 2): 0, (1, 2): 0},
			(-1, 0): {(0, 2): 0, (0, 1): 0},
			(0, -1): {(0, 1): 0, (0, 2): 0, (1, 2): 0}},
		(1, 0): {
			(1, 0): {(2, 0): 0, (1, 0): 0},
			(0, 1): {(1, 0): 0, (0, 0): 0, (2, 0): 0},
			(-1, 0): {(0, 0): 0, (1, 0): 0},
			(0, -1): {(1, 0): 0, (0, 0): 0, (2, 0): 0}},
		(1, 2): {
			(1, 0): {(2, 2): 0, (1, 2): 0},
			(0, 1): {(1, 2): 0, (0, 2): 0, (2, 2): 0},
			(-1, 0): {(0, 0): 0, (1, 0): 0},
			(0, -1): {(1, 2): 0, (0, 2): 0, (2, 2): 0}},

		(2, 0): {
			(1, 0): {(3, 0): 0, (2, 1): 0, (2, 0): 0},
			(0, 1): {(2, 1): 0, (3, 0): 0, (1, 0): 0},
			(-1, 0): {(1, 0): 0, (2, 1): 0, (2, 0): 0},
			(0, -1): {(2, 0): 0, (1, 0): 0, (3, 0): 0}},
		(2, 1): {
			(1, 0): {(3, 1): 0, (2, 0): 0, (2, 2): 0},
			(0, 1): {(2, 2): 0, (3, 1): 0, (2, 1): 0},
			(-1, 0): {(2, 1): 0, (2, 0): 0, (2, 2): 0},
			(0, -1): {(2, 0): 0, (3, 1): 0, (2, 1): 0}},
		(2, 2): {
			(1, 0): {(3, 2): 0, (2, 1): 0, (2, 2): 0},
			(0, 1): {(2, 2): 0, (1, 2): 0, (3, 2): 0},
			(-1, 0): {(1, 2): 0, (2, 2): 0, (2, 1): 0},
			(0, -1): {(2, 1): 0, (1, 2): 0, (3, 2): 0}},

		(3, 0): {
			(1, 0): {(3, 0): 0, (3, 1): 0},
			(0, 1): {(3, 1): 0, (3, 0): 0, (2, 0): 0},
			(-1, 0): {(2, 0): 0, (3, 1): 0, (3, 0): 0},
			(0, -1): {(3, 0): 0, (2, 0): 0}},
		(3, 1): {
			(1, 0): {(3, 1): -1, (3, 0): -1, (3, 2): -1},
			(0, 1): {(3, 2): -1, (3, 1): -1, (2, 1): -1},
			(-1, 0): {(2, 1): -1, (3, 0): -1, (3, 2): -1},
			(0, -1): {(3, 0): -1, (2, 1): -1, (3, 1): -1}},
		(3, 2): {
			(1, 0): {(3, 2): 1, (3, 1): 1},
			(0, 1): {(3, 2): 1, (2, 2): 1},
			(-1, 0): {(2, 2): 1, (3, 1): 1, (3, 2): 1},
			(0, -1): {(3, 1): 1, (2, 2): 1, (3, 2): 1}}
	}

	valueTableBook = {
		(0, 0): 0, (0, 1): 0, (0, 2): 0,
		(1, 0): 0, (1, 1): -np.inf, (1, 2): 0,
		(2, 0): 0, (2, 1): 0, (2, 2): 0,
		(3, 0): 0, (3, 1): 0, (3, 2): 0}

	convergenceTolerance = 10e-7
	gamma = .9

	# """
	# Uncomment to view transition or reward structure in a readable format
	# """
	# levelsReward  = ["state", "action", "next state", "reward"]
	# levelsTransition  = ["state", "action", "next state", "probability"]
	#
	# viewDictionaryStructure(transitionTableDet, levelsTransition)
	# print('================================')
	# viewDictionaryStructure(transitionTable, levelsTransition)
	# # viewDictionaryStructure(rewardTableDet, levelsReward)
	#

	performValueIteration = ValueIteration(
		transitionTableDet, rewardTableDet, valueTableDet, convergenceTolerance, gamma)
	optimalValuesDeterminsitic, policyTableDet = performValueIteration()
	print(optimalValuesDeterminsitic)
	print(policyTableDet)

	performValueIteration = ValueIteration(transitionTable, rewardTable, valueTable, convergenceTolerance, gamma)
	optimalValuesDeterminsitic, policyTable = performValueIteration()
	print(optimalValuesDeterminsitic)
	print(policyTable)

	# performValueIteration = ValueIteration(transitionTableBook, rewardTableBook, valueTableBook, convergenceTolerance, gamma)
	# optimalValuesDeterminsitic, policyTable = performValueIteration()
	# print(optimalValuesDeterminsitic)
	#
	# levels_values = ["state", "values"]
	#
	# viewDictionaryStructure(optimalValuesDeterminsitic, levels_values)
	# print(policyTable)
	

if __name__ == '__main__':
	main()