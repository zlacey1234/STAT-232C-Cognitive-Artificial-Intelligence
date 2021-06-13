""" GetLikelihoodReward.py

Class: STAT 232C - Cognitive Artificial Intelligence
Project 4: Signaling Policy
Name: Zachary Lacey
Date: May 22nd, 2021

Goal:   return the new reward function
        get a policy that reflects signaling to an observer for each possible true goal in the environment.
        visualize the policy as before by providing graphs of the value table and policy
        write a brief commentary on what has changed between the original goal policies and the new signaling policies.
"""


from HW3.GoalInference import *


class GetLikelihoodReward(object):
    """ Get Likelihood Reward Class """
    
    def __init__(self, transition_table, goal_policies):
        """
        Args:
            transition_table (datatype: dict): This is a nested dictionary of the state-action-nextstate combinations
            for transitions that have a non-zero probability. The transition_table has the following structure:

                transition_table = {state: {action: {nextstate: {probability}}}}  == P(s' | s, a)

                Example:
                {(0, 0): {(1, 0): {(1, 0): 0.7, (0, 1): 0.2, (0, 0): 0.1}}
                {state (0, 0) : action (1, 0):
                    nextstate (1, 0) [move right]: probability 0.7
                    nextstate (0, 1) [move up]: probability 0.2
                    nextstate (0, 0) [no movement]: probability 0.1}
            
            goal_policies (datatype: dict): This is 
        """
        self.transition_table = transition_table
        self.goal_policies = goal_policies 
        # can be dictionary of form goal:goal policy or a list, but true_goal should link to the correct policy here

    def __call__(self, true_goal, original_reward, alpha):
        """
        Args:
            true_goal (datatype: state): This is an State value that corresponds to the true goal state of the
            trajectory. For our case, we have three goal trajectories:

                true_goal:
                    Goal A: (6, 1)

                    Goal B: (6, 4)

                    Goal C: (1, 5)

            original_reward (datatype: dict): This is a nested dictionary of form state-action-nextstate combinations
            for the original expected reward. The original_reward has the following structure:

                original_reward = {state: {action: {nextstate: {reward}}}} == R(s' | s, a)

            alpha:

        Returns:
            new_reward (datatype: dict): This is a nested dictionary of form state-action-nextstate combinations
            for the new signaling expected reward. The new_reward has the following structure:

                new_reward = {state: {action: {nextstate: {reward}}}} == R(s' | s, a)

        """
        # Possible Goal States
        goal_states = list(self.goal_policies)
        num_goals = len(goal_states)

        marginal_probability_next_state_tables = dict()

        # For each goal
        for g in range(num_goals):
            marginal_probability_next_state_tables[goal_states[g]] = get_probability_of_individual_state_transitions(
                self.transition_table, self.goal_policies[goal_states[g]])

        # Likelihood Ratio of the True Goal
        likelihood_ratio_true_goal = GetLikelihoodReward.get_likelihood_ratio(
            true_goal, marginal_probability_next_state_tables)

        # New Reward Calculations
        new_reward = dict()

        states = list(self.transition_table)
        num_states = len(states)

        # For each state
        for s in range(num_states):
            actions = list(self.transition_table[states[s]])
            num_actions = len(actions)

            new_reward_actions = dict()

            # For each action
            for a in range(num_actions):
                state_prime = list(self.transition_table[states[s]][actions[a]])
                num_states_prime = len(state_prime)

                new_reward_state_primes = dict()

                for sp in range(num_states_prime):
                    o_r = original_reward[states[s]][actions[a]][state_prime[sp]]

                    n_r = o_r + alpha*(likelihood_ratio_true_goal[states[s]][state_prime[sp]])

                    new_reward_state_primes[state_prime[sp]] = n_r

                new_reward_actions[actions[a]] = new_reward_state_primes

            new_reward[states[s]] = new_reward_actions

        return new_reward

    @staticmethod
    def get_likelihood_ratio(true_goal, marginal_probability_next_state_tables):
        """
        Get Likelihood Ratio Method

        This method

        Args:
            true_goal (datatype: state): This is an State value that corresponds to the true goal state of the
            trajectory. For our case, we have three goal trajectories:

                true_goal:
                    Goal A: (6, 1)

                    Goal B: (6, 4)

                    Goal C: (1, 5)

            marginal_probability_next_state_tables

        """
        goal_states = list(marginal_probability_next_state_tables)
        num_goals = len(goal_states)

        # for g in range(num_goals):
        states = list(marginal_probability_next_state_tables[true_goal])
        num_states = len(states)

        sum_of_marginal_probabilities_states_mem = dict()
        likelihood_ratio = dict()
        for s in range(num_states):
            state_prime = list(marginal_probability_next_state_tables[true_goal][states[s]])
            num_states_prime = len(state_prime)
            sum_of_marginal_probabilities_states_primes_mem = dict()
            likelihood_ratio_state_prime_mem = dict()
            for sp in range(num_states_prime):
                sum_of_marginal_probabilities_states_primes_mem[state_prime[sp]] = np.sum(
                    [marginal_probability_next_state_tables[goal_states[goals]][states[s]][state_prime[sp]] for
                        goals in range(num_goals)])

                likelihood_ratio_state_prime_mem[state_prime[sp]] = \
                    marginal_probability_next_state_tables[true_goal][states[s]][state_prime[sp]] / \
                    sum_of_marginal_probabilities_states_primes_mem[state_prime[sp]]

            sum_of_marginal_probabilities_states_mem[states[s]] = sum_of_marginal_probabilities_states_primes_mem
            likelihood_ratio[states[s]] = likelihood_ratio_state_prime_mem

        return likelihood_ratio


def main():
    # Parameters across all goals and environments
    convergence_threshold = 10e-7
    gamma = .9
    beta = 2
    alpha = 5
    
    # Environment specifications
    grid_width = 7
    grid_height = 6
    all_actions = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]
    trap_states = [(3, 0), (3, 1), (3, 3)]
    goal_a = (6, 1)
    goal_b = (6, 4)
    goal_c = (1, 5)

    # Transitions
    transition = {
        (0, 0): {
            (1, 0): {(1, 0): 1}, (0, 1): {(0, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(0, 0): 1}, (0, 0): {(0, 0): 1}},
        (0, 1): {
            (1, 0): {(1, 1): 1}, (0, 1): {(0, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(0, 0): 1}, (0, 0): {(0, 1): 1}},
        (0, 2): {
            (1, 0): {(1, 2): 1}, (0, 1): {(0, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(0, 1): 1}, (0, 0): {(0, 2): 1}},
        (0, 3): {
            (1, 0): {(1, 3): 1}, (0, 1): {(0, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(0, 2): 1}, (0, 0): {(0, 3): 1}},
        (0, 4): {
            (1, 0): {(1, 4): 1}, (0, 1): {(0, 5): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(0, 3): 1}, (0, 0): {(0, 4): 1}},
        (0, 5): {
            (1, 0): {(1, 5): 1}, (0, 1): {(0, 5): 1}, (-1, 0): {(0, 5): 1}, (0, -1): {(0, 4): 1}, (0, 0): {(0, 5): 1}},
        (1, 0): {
            (1, 0): {(2, 0): 1}, (0, 1): {(1, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(1, 0): 1}, (0, 0): {(1, 0): 1}},
        (1, 1): {
            (1, 0): {(2, 1): 1}, (0, 1): {(1, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(1, 0): 1}, (0, 0): {(1, 1): 1}},
        (1, 2): {
            (1, 0): {(2, 2): 1}, (0, 1): {(1, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(1, 1): 1}, (0, 0): {(1, 2): 1}},
        (1, 3): {
            (1, 0): {(2, 3): 1}, (0, 1): {(1, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(1, 2): 1}, (0, 0): {(1, 3): 1}},
        (1, 4): {
            (1, 0): {(2, 4): 1}, (0, 1): {(1, 5): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(1, 3): 1}, (0, 0): {(1, 4): 1}},
        (1, 5): {
            (1, 0): {(2, 5): 1}, (0, 1): {(1, 5): 1}, (-1, 0): {(0, 5): 1}, (0, -1): {(1, 4): 1}, (0, 0): {(1, 5): 1}},
        (2, 0): {
            (1, 0): {(3, 0): 1}, (0, 1): {(2, 1): 1}, (-1, 0): {(1, 0): 1}, (0, -1): {(2, 0): 1}, (0, 0): {(2, 0): 1}},
        (2, 1): {
            (1, 0): {(3, 1): 1}, (0, 1): {(2, 2): 1}, (-1, 0): {(1, 1): 1}, (0, -1): {(2, 0): 1}, (0, 0): {(2, 1): 1}},
        (2, 2): {
            (1, 0): {(3, 2): 1}, (0, 1): {(2, 3): 1}, (-1, 0): {(1, 2): 1}, (0, -1): {(2, 1): 1}, (0, 0): {(2, 2): 1}},
        (2, 3): {
            (1, 0): {(3, 3): 1}, (0, 1): {(2, 4): 1}, (-1, 0): {(1, 3): 1}, (0, -1): {(2, 2): 1}, (0, 0): {(2, 3): 1}},
        (2, 4): {
            (1, 0): {(3, 4): 1}, (0, 1): {(2, 5): 1}, (-1, 0): {(1, 4): 1}, (0, -1): {(2, 3): 1}, (0, 0): {(2, 4): 1}},
        (2, 5): {
            (1, 0): {(3, 5): 1}, (0, 1): {(2, 5): 1}, (-1, 0): {(1, 5): 1}, (0, -1): {(2, 4): 1}, (0, 0): {(2, 5): 1}},
        (3, 0): {
            (1, 0): {(4, 0): 1}, (0, 1): {(3, 1): 1}, (-1, 0): {(2, 0): 1}, (0, -1): {(3, 0): 1}, (0, 0): {(3, 0): 1}},
        (3, 1): {
            (1, 0): {(4, 1): 1}, (0, 1): {(3, 2): 1}, (-1, 0): {(2, 1): 1}, (0, -1): {(3, 0): 1}, (0, 0): {(3, 1): 1}},
        (3, 2): {
            (1, 0): {(4, 2): 1}, (0, 1): {(3, 3): 1}, (-1, 0): {(2, 2): 1}, (0, -1): {(3, 1): 1}, (0, 0): {(3, 2): 1}},
        (3, 3): {
            (1, 0): {(4, 3): 1}, (0, 1): {(3, 4): 1}, (-1, 0): {(2, 3): 1}, (0, -1): {(3, 2): 1}, (0, 0): {(3, 3): 1}},
        (3, 4): {
            (1, 0): {(4, 4): 1}, (0, 1): {(3, 5): 1}, (-1, 0): {(2, 4): 1}, (0, -1): {(3, 3): 1}, (0, 0): {(3, 4): 1}},
        (3, 5): {
            (1, 0): {(4, 5): 1}, (0, 1): {(3, 5): 1}, (-1, 0): {(2, 5): 1}, (0, -1): {(3, 4): 1}, (0, 0): {(3, 5): 1}},
        (4, 0): {
            (1, 0): {(5, 0): 1}, (0, 1): {(4, 1): 1}, (-1, 0): {(3, 0): 1}, (0, -1): {(4, 0): 1}, (0, 0): {(4, 0): 1}},
        (4, 1): {
            (1, 0): {(5, 1): 1}, (0, 1): {(4, 2): 1}, (-1, 0): {(3, 1): 1}, (0, -1): {(4, 0): 1}, (0, 0): {(4, 1): 1}},
        (4, 2): {
            (1, 0): {(5, 2): 1}, (0, 1): {(4, 3): 1}, (-1, 0): {(3, 2): 1}, (0, -1): {(4, 1): 1}, (0, 0): {(4, 2): 1}},
        (4, 3): {
            (1, 0): {(5, 3): 1}, (0, 1): {(4, 4): 1}, (-1, 0): {(3, 3): 1}, (0, -1): {(4, 2): 1}, (0, 0): {(4, 3): 1}},
        (4, 4): {
            (1, 0): {(5, 4): 1}, (0, 1): {(4, 5): 1}, (-1, 0): {(3, 4): 1}, (0, -1): {(4, 3): 1}, (0, 0): {(4, 4): 1}},
        (4, 5): {
            (1, 0): {(5, 5): 1}, (0, 1): {(4, 5): 1}, (-1, 0): {(3, 5): 1}, (0, -1): {(4, 4): 1}, (0, 0): {(4, 5): 1}},
        (5, 0): {
            (1, 0): {(6, 0): 1}, (0, 1): {(5, 1): 1}, (-1, 0): {(4, 0): 1}, (0, -1): {(5, 0): 1}, (0, 0): {(5, 0): 1}},
        (5, 1): {
            (1, 0): {(6, 1): 1}, (0, 1): {(5, 2): 1}, (-1, 0): {(4, 1): 1}, (0, -1): {(5, 0): 1}, (0, 0): {(5, 1): 1}},
        (5, 2): {
            (1, 0): {(6, 2): 1}, (0, 1): {(5, 3): 1}, (-1, 0): {(4, 2): 1}, (0, -1): {(5, 1): 1}, (0, 0): {(5, 2): 1}},
        (5, 3): {
            (1, 0): {(6, 3): 1}, (0, 1): {(5, 4): 1}, (-1, 0): {(4, 3): 1}, (0, -1): {(5, 2): 1}, (0, 0): {(5, 3): 1}},
        (5, 4): {
            (1, 0): {(6, 4): 1}, (0, 1): {(5, 5): 1}, (-1, 0): {(4, 4): 1}, (0, -1): {(5, 3): 1}, (0, 0): {(5, 4): 1}},
        (5, 5): {
            (1, 0): {(6, 5): 1}, (0, 1): {(5, 5): 1}, (-1, 0): {(4, 5): 1}, (0, -1): {(5, 4): 1}, (0, 0): {(5, 5): 1}},
        (6, 0): {
            (1, 0): {(6, 0): 1}, (0, 1): {(6, 1): 1}, (-1, 0): {(5, 0): 1}, (0, -1): {(6, 0): 1}, (0, 0): {(6, 0): 1}},
        (6, 1): {
            (1, 0): {(6, 1): 1}, (0, 1): {(6, 2): 1}, (-1, 0): {(5, 1): 1}, (0, -1): {(6, 0): 1}, (0, 0): {(6, 1): 1}},
        (6, 2): {
            (1, 0): {(6, 2): 1}, (0, 1): {(6, 3): 1}, (-1, 0): {(5, 2): 1}, (0, -1): {(6, 1): 1}, (0, 0): {(6, 2): 1}},
        (6, 3): {
            (1, 0): {(6, 3): 1}, (0, 1): {(6, 4): 1}, (-1, 0): {(5, 3): 1}, (0, -1): {(6, 2): 1}, (0, 0): {(6, 3): 1}},
        (6, 4): {
            (1, 0): {(6, 4): 1}, (0, 1): {(6, 5): 1}, (-1, 0): {(5, 4): 1}, (0, -1): {(6, 3): 1}, (0, 0): {(6, 4): 1}},
        (6, 5): {
            (1, 0): {(6, 5): 1}, (0, 1): {(6, 5): 1}, (-1, 0): {(5, 5): 1}, (0, -1): {(6, 4): 1}, (0, 0): {(6, 5): 1}}}

    # Rewards
    reward_for_goal_a = dict()
    reward_for_goal_b = dict()
    reward_for_goal_c = dict()

    common_rewards = {
        (0, 0): {
            (1, 0): {(1, 0): -1}, (0, 1): {(0, 1): -1},
            (-1, 0): {(0, 0): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 0): -0.1}},
        (0, 1): {
            (1, 0): {(1, 1): -1}, (0, 1): {(0, 2): -1},
            (-1, 0): {(0, 1): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 1): -0.1}},
        (0, 2): {
            (1, 0): {(1, 2): -1}, (0, 1): {(0, 3): -1},
            (-1, 0): {(0, 2): -1}, (0, -1): {(0, 1): -1}, (0, 0): {(0, 2): -0.1}},
        (0, 3): {
            (1, 0): {(1, 3): -1}, (0, 1): {(0, 4): -1},
            (-1, 0): {(0, 3): -1}, (0, -1): {(0, 2): -1}, (0, 0): {(0, 3): -0.1}},
        (0, 4): {
            (1, 0): {(1, 4): -1}, (0, 1): {(0, 5): -1},
            (-1, 0): {(0, 4): -1}, (0, -1): {(0, 3): -1}, (0, 0): {(0, 4): -0.1}},
        (0, 5): {
            (1, 0): {(1, 5): -1}, (0, 1): {(0, 5): -1},
            (-1, 0): {(0, 5): -1}, (0, -1): {(0, 4): -1}, (0, 0): {(0, 5): -0.1}},
        (1, 0): {
            (1, 0): {(2, 0): -1}, (0, 1): {(1, 1): -1},
            (-1, 0): {(0, 0): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 0): -0.1}},
        (1, 1): {
            (1, 0): {(2, 1): -1}, (0, 1): {(1, 2): -1},
            (-1, 0): {(0, 1): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 1): -0.1}},
        (1, 2): {
            (1, 0): {(2, 2): -1}, (0, 1): {(1, 3): -1},
            (-1, 0): {(0, 2): -1}, (0, -1): {(1, 1): -1}, (0, 0): {(1, 2): -0.1}},
        (1, 3): {
            (1, 0): {(2, 3): -1}, (0, 1): {(1, 4): -1},
            (-1, 0): {(0, 3): -1}, (0, -1): {(1, 2): -1}, (0, 0): {(1, 3): -0.1}},
        (1, 4): {
            (1, 0): {(2, 4): -1}, (0, 1): {(1, 5): -1},
            (-1, 0): {(0, 4): -1}, (0, -1): {(1, 3): -1}, (0, 0): {(1, 4): -0.1}},
        (2, 0): {
            (1, 0): {(3, 0): -1}, (0, 1): {(2, 1): -1},
            (-1, 0): {(1, 0): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 0): -0.1}},
        (2, 1): {
            (1, 0): {(3, 1): -1}, (0, 1): {(2, 2): -1},
            (-1, 0): {(1, 1): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 1): -0.1}},
        (2, 2): {
            (1, 0): {(3, 2): -1}, (0, 1): {(2, 3): -1},
            (-1, 0): {(1, 2): -1}, (0, -1): {(2, 1): -1}, (0, 0): {(2, 2): -0.1}},
        (2, 3): {
            (1, 0): {(3, 3): -1}, (0, 1): {(2, 4): -1},
            (-1, 0): {(1, 3): -1}, (0, -1): {(2, 2): -1}, (0, 0): {(2, 3): -0.1}},
        (2, 4): {
            (1, 0): {(3, 4): -1}, (0, 1): {(2, 5): -1},
            (-1, 0): {(1, 4): -1}, (0, -1): {(2, 3): -1}, (0, 0): {(2, 4): -0.1}},
        (2, 5): {
            (1, 0): {(3, 5): -1}, (0, 1): {(2, 5): -1},
            (-1, 0): {(1, 5): -1}, (0, -1): {(2, 4): -1}, (0, 0): {(2, 5): -0.1}},
        (3, 0): {
            (1, 0): {(4, 0): -100}, (0, 1): {(3, 1): -100},
            (-1, 0): {(2, 0): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 0): -100}},
        (3, 1): {
            (1, 0): {(4, 1): -100}, (0, 1): {(3, 2): -100},
            (-1, 0): {(2, 1): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 1): -100}},
        (3, 2): {
            (1, 0): {(4, 2): -1}, (0, 1): {(3, 3): -1},
            (-1, 0): {(2, 2): -1}, (0, -1): {(3, 1): -1}, (0, 0): {(3, 2): -0.1}},
        (3, 3): {
            (1, 0): {(4, 3): -100}, (0, 1): {(3, 4): -100},
            (-1, 0): {(2, 3): -100}, (0, -1): {(3, 2): -100}, (0, 0): {(3, 3): -100}},
        (3, 4): {
            (1, 0): {(4, 4): -1}, (0, 1): {(3, 5): -1},
            (-1, 0): {(2, 4): -1}, (0, -1): {(3, 3): -1}, (0, 0): {(3, 4): -0.1}},
        (3, 5): {
            (1, 0): {(4, 5): -1}, (0, 1): {(3, 5): -1},
            (-1, 0): {(2, 5): -1}, (0, -1): {(3, 4): -1}, (0, 0): {(3, 5): -0.1}},
        (4, 0): {
            (1, 0): {(5, 0): -1}, (0, 1): {(4, 1): -1},
            (-1, 0): {(3, 0): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 0): -0.1}},
        (4, 1): {
            (1, 0): {(5, 1): -1}, (0, 1): {(4, 2): -1},
            (-1, 0): {(3, 1): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 1): -0.1}},
        (4, 2): {
            (1, 0): {(5, 2): -1}, (0, 1): {(4, 3): -1},
            (-1, 0): {(3, 2): -1}, (0, -1): {(4, 1): -1}, (0, 0): {(4, 2): -0.1}},
        (4, 3): {
            (1, 0): {(5, 3): -1}, (0, 1): {(4, 4): -1},
            (-1, 0): {(3, 3): -1}, (0, -1): {(4, 2): -1}, (0, 0): {(4, 3): -0.1}},
        (4, 4): {
            (1, 0): {(5, 4): -1}, (0, 1): {(4, 5): -1},
            (-1, 0): {(3, 4): -1}, (0, -1): {(4, 3): -1}, (0, 0): {(4, 4): -0.1}},
        (4, 5): {
            (1, 0): {(5, 5): -1}, (0, 1): {(4, 5): -1},
            (-1, 0): {(3, 5): -1}, (0, -1): {(4, 4): -1}, (0, 0): {(4, 5): -0.1}},
        (5, 0): {
            (1, 0): {(6, 0): -1}, (0, 1): {(5, 1): -1},
            (-1, 0): {(4, 0): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 0): -0.1}},
        (5, 1): {
            (1, 0): {(6, 1): -1}, (0, 1): {(5, 2): -1},
            (-1, 0): {(4, 1): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 1): -0.1}},
        (5, 2): {
            (1, 0): {(6, 2): -1}, (0, 1): {(5, 3): -1},
            (-1, 0): {(4, 2): -1}, (0, -1): {(5, 1): -1}, (0, 0): {(5, 2): -0.1}},
        (5, 3): {
            (1, 0): {(6, 3): -1}, (0, 1): {(5, 4): -1},
            (-1, 0): {(4, 3): -1}, (0, -1): {(5, 2): -1}, (0, 0): {(5, 3): -0.1}},
        (5, 4): {
            (1, 0): {(6, 4): -1}, (0, 1): {(5, 5): -1},
            (-1, 0): {(4, 4): -1}, (0, -1): {(5, 3): -1}, (0, 0): {(5, 4): -0.1}},
        (5, 5): {
            (1, 0): {(6, 5): -1}, (0, 1): {(5, 5): -1},
            (-1, 0): {(4, 5): -1}, (0, -1): {(5, 4): -1}, (0, 0): {(5, 5): -0.1}},
        (6, 0): {
            (1, 0): {(6, 0): -1}, (0, 1): {(6, 1): -1},
            (-1, 0): {(5, 0): -1}, (0, -1): {(6, 0): -1}, (0, 0): {(6, 0): -0.1}},
        (6, 2): {
            (1, 0): {(6, 2): -1}, (0, 1): {(6, 3): -1},
            (-1, 0): {(5, 2): -1}, (0, -1): {(6, 1): -1}, (0, 0): {(6, 2): -0.1}},
        (6, 3): {
            (1, 0): {(6, 3): -1}, (0, 1): {(6, 4): -1},
            (-1, 0): {(5, 3): -1}, (0, -1): {(6, 2): -1}, (0, 0): {(6, 3): -0.1}},
        (6, 5): {
            (1, 0): {(6, 5): -1}, (0, 1): {(6, 5): -1},
            (-1, 0): {(5, 5): -1}, (0, -1): {(6, 4): -1}, (0, 0): {(6, 5): -0.1}}
    }

    reward_for_1_5_goal_a_b = {
        (1, 5): {
            (1, 0): {(2, 5): -1}, (0, 1): {(1, 5): -1},
            (-1, 0): {(0, 5): -1}, (0, -1): {(1, 4): -1}, (0, 0): {(1, 5): -0.1}}}

    reward_for_1_5_goal_c = {
        (1, 5): {
            (1, 0): {(2, 5): 9}, (0, 1): {(1, 5): 9},
            (-1, 0): {(0, 5): 9}, (0, -1): {(1, 4): 9}, (0, 0): {(1, 5): 9.9}}}

    reward_for_6_1_goal_a = {
        (6, 1): {
            (1, 0): {(6, 1): 9}, (0, 1): {(6, 2): 9},
            (-1, 0): {(5, 1): 9}, (0, -1): {(6, 0): 9}, (0, 0): {(6, 1): 9.9}}}

    reward_for_6_1_goal__b_c = {
        (6, 1): {
            (1, 0): {(6, 1): -1}, (0, 1): {(6, 2): -1},
            (-1, 0): {(5, 1): -1}, (0, -1): {(6, 0): -1}, (0, 0): {(6, 1): -0.1}}}

    reward_for_6_4_goal_a_c = {
        (6, 4): {
            (1, 0): {(6, 4): -1}, (0, 1): {(6, 5): -1},
            (-1, 0): {(5, 4): -1}, (0, -1): {(6, 3): -1}, (0, 0): {(6, 4): -0.1}}}

    reward_for_6_4_goal_b = {
        (6, 4): {
            (1, 0): {(6, 4): 9}, (0, 1): {(6, 5): 9},
            (-1, 0): {(5, 4): 9}, (0, -1): {(6, 3): 9}, (0, 0): {(6, 4): 9.9}}}

    reward_for_goal_a.update(common_rewards)
    reward_for_goal_a.update(reward_for_1_5_goal_a_b)
    reward_for_goal_a.update(reward_for_6_1_goal_a)
    reward_for_goal_a.update(reward_for_6_4_goal_a_c)

    reward_for_goal_b.update(common_rewards)
    reward_for_goal_b.update(reward_for_1_5_goal_a_b)
    reward_for_goal_b.update(reward_for_6_1_goal__b_c)
    reward_for_goal_b.update(reward_for_6_4_goal_b)

    reward_for_goal_c.update(common_rewards)
    reward_for_goal_c.update(reward_for_1_5_goal_c)
    reward_for_goal_c.update(reward_for_6_1_goal__b_c)
    reward_for_goal_c.update(reward_for_6_4_goal_a_c)

    # Goal A
    perform_value_iteration_goal_a_original = ValueIteration(transition, reward_for_goal_a, value_table_initial,
                                                             convergence_threshold, gamma, use_softmax=True,
                                                             use_noise=True, noise_beta=beta)
    optimal_value_table_a_original, optimal_policy_table_a_original = perform_value_iteration_goal_a_original()

    visualize_value_table(grid_width, grid_height, goal_a, trap_states, optimal_value_table_a_original)
    visualize_policy(grid_width, grid_height, goal_a, trap_states, optimal_policy_table_a_original)

    # Goal B
    perform_value_iteration_goal_b_original = ValueIteration(transition, reward_for_goal_b, value_table_initial,
                                                             convergence_threshold, gamma, use_softmax=True,
                                                             use_noise=True, noise_beta=beta)
    optimal_value_table_b_original, optimal_policy_table_b_original = perform_value_iteration_goal_b_original()

    visualize_value_table(grid_width, grid_height, goal_b, trap_states, optimal_value_table_b_original)
    visualize_policy(grid_width, grid_height, goal_b, trap_states, optimal_policy_table_b_original)

    # Goal C
    perform_value_iteration_goal_c_original = ValueIteration(transition, reward_for_goal_c, value_table_initial,
                                                             convergence_threshold, gamma, use_softmax=True,
                                                             use_noise=True, noise_beta=beta)
    optimal_value_table_c_original, optimal_policy_table_c_original = perform_value_iteration_goal_c_original()

    visualize_value_table(grid_width, grid_height, goal_c, trap_states, optimal_value_table_c_original)
    visualize_policy(grid_width, grid_height, goal_c, trap_states, optimal_policy_table_c_original)

    # Goal Policies
    goal_policies = dict()

    goal_policies[goal_a] = optimal_policy_table_a_original
    goal_policies[goal_b] = optimal_policy_table_b_original
    goal_policies[goal_c] = optimal_policy_table_c_original

    print('Goals')
    print(goal_policies)

    perform_get_likelihood_reward = GetLikelihoodReward(transition, goal_policies)

    # Goal A
    new_reward_a = perform_get_likelihood_reward(goal_a, reward_for_goal_a, alpha)

    print(new_reward_a)

    perform_value_iteration_goal_a_new = ValueIteration(transition, new_reward_a, value_table_initial,
                                                        convergence_threshold, gamma, use_softmax=True,
                                                        use_noise=True, noise_beta=beta)
    optimal_value_table_a_new, optimal_policy_table_a_new = perform_value_iteration_goal_a_new()

    visualize_value_table(grid_width, grid_height, goal_a, trap_states, optimal_value_table_a_new)
    visualize_policy(grid_width, grid_height, goal_a, trap_states, optimal_policy_table_a_new)

    # Goal B
    new_reward_b = perform_get_likelihood_reward(goal_b, reward_for_goal_b, alpha)

    print(new_reward_b)

    perform_value_iteration_goal_b_new = ValueIteration(transition, new_reward_b, value_table_initial,
                                                        convergence_threshold, gamma, use_softmax=True,
                                                        use_noise=True, noise_beta=beta)
    optimal_value_table_b_new, optimal_policy_table_b_new = perform_value_iteration_goal_b_new()

    visualize_value_table(grid_width, grid_height, goal_b, trap_states, optimal_value_table_b_new)
    visualize_policy(grid_width, grid_height, goal_b, trap_states, optimal_policy_table_b_new)

    # Goal C
    new_reward_c = perform_get_likelihood_reward(goal_c, reward_for_goal_c, alpha)

    print(new_reward_c)

    perform_value_iteration_goal_c_new = ValueIteration(transition, new_reward_c, value_table_initial,
                                                        convergence_threshold, gamma, use_softmax=True,
                                                        use_noise=True, noise_beta=beta)
    optimal_value_table_c_new, optimal_policy_table_c_new = perform_value_iteration_goal_c_new()

    visualize_value_table(grid_width, grid_height, goal_c, trap_states, optimal_value_table_c_new)
    visualize_policy(grid_width, grid_height, goal_c, trap_states, optimal_policy_table_c_new)


if __name__ == "__main__":
    main()
