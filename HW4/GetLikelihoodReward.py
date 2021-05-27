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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from HW3.GoalInference import *


class GetLikelihoodReward(object):
    """ Get Likelihood Reward Class """
    
    def __init__(self, transitionTable, goalPolicies):
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
            
            goalPolicies (datatype: dict): This is 
        """
        self.transitionTable = transitionTable
        self.goalPolicies = goalPolicies 
        # can be dictionary of form goal:goal policy or a list, but trueGoal should link to the correct policy here

    def __call__(self, trueGoal, originalReward, alpha):
        """
        Args:
            trueGoal (datatype: state): This is an State value that corresponds to the true goal state of the
            trajectory. For our case, we have three goal trajectories:

                trueGoal:
                    Goal A: (6, 1)

                    Goal B: (6, 4)

                    Goal C: (1, 5)

            originalReward:
            alpha:

        Returns:
            newReward

        """
        # Possible Goal States
        goal_states = list(self.goalPolicies)
        num_goals = len(goal_states)

        marginal_probability_next_state_tables = dict()

        # For each goal
        for g in range(num_goals):
            marginal_probability_next_state_tables[goal_states[g]] = get_probability_of_individual_state_transitions(
                self.transitionTable, self.goalPolicies[goal_states[g]])

        # Likelihood Ratio of the True Goal
        likelihood_ratio_true_goal = GetLikelihoodReward.get_likelihood_ratio(
            trueGoal, marginal_probability_next_state_tables)

        # New Reward Calculations
        newReward = dict()

        states = list(self.transitionTable)
        num_states = len(states)

        # For each state
        for s in range(num_states):
            actions = list(self.transitionTable[states[s]])
            num_actions = len(actions)

            new_reward_actions = dict()

            # For each action
            for a in range(num_actions):
                state_prime = list(self.transitionTable[states[s]][actions[a]])
                num_states_prime = len(state_prime)

                new_reward_state_primes = dict()

                for sp in range(num_states_prime):
                    o_r = originalReward[states[s]][actions[a]][state_prime[sp]]
                    print('original reward')
                    print(o_r)

                    print('ratio')
                    print(likelihood_ratio_true_goal[states[s]][state_prime[sp]])

                    print('new reward')

                    n_r = o_r + alpha*(likelihood_ratio_true_goal[states[s]][state_prime[sp]])
                    print(n_r)
                    new_reward_state_primes[state_prime[sp]] = n_r
                    print('\n')

                print(new_reward_state_primes)
                new_reward_actions[actions[a]] = new_reward_state_primes

            print('new reward (actions)')
            print(new_reward_actions)
            newReward[states[s]] = new_reward_actions

        
        return(newReward)

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

        sum_of_marginal_probabilities_mem = dict()

        # for g in range(num_goals):
        states = list(marginal_probability_next_state_tables[true_goal])
        num_states = len(states)

        sum_of_marginal_probabilities_states_mem = dict()
        likelihood_ratio = dict()
        for s in range(num_states):
            print(states[s])
            state_prime = list(marginal_probability_next_state_tables[true_goal][states[s]])
            num_states_prime = len(state_prime)
            print(state_prime)
            sum_of_marginal_probabilities_states_primes_mem = dict()
            likelihood_ratio_state_prime_mem = dict()
            for sp in range(num_states_prime):
                sum_of_marginal_probabilities_states_primes_mem[state_prime[sp]] = np.sum(
                    [marginal_probability_next_state_tables[goal_states[goals]][states[s]][state_prime[sp]] for
                        goals in range(num_goals)])

                likelihood_ratio_state_prime_mem[state_prime[sp]] = \
                    marginal_probability_next_state_tables[true_goal][states[s]][state_prime[sp]] / \
                    sum_of_marginal_probabilities_states_primes_mem[state_prime[sp]]

            print('state prime sum')
            print(sum_of_marginal_probabilities_states_primes_mem)
            print(likelihood_ratio_state_prime_mem)

            sum_of_marginal_probabilities_states_mem[states[s]] = sum_of_marginal_probabilities_states_primes_mem
            likelihood_ratio[states[s]] = likelihood_ratio_state_prime_mem

        print('state sum')
        print(sum_of_marginal_probabilities_states_mem)
        print(likelihood_ratio)
        print('\n\n')

        return likelihood_ratio





def visualizeValueTable(gridWidth, gridHeight, goalState, trapStates, valueTable):
    gridAdjust = .5
    gridScale = 1.5
    
    xs = np.linspace(-gridAdjust, gridWidth-gridAdjust, gridWidth+1)
    ys = np.linspace(-gridAdjust, gridHeight-gridAdjust, gridHeight+1)
    
    plt.rcParams["figure.figsize"] = [gridWidth*gridScale,gridHeight*gridScale]
    ax = plt.gca(frameon=False, xticks = range(gridWidth), yticks = range(gridHeight))

    #goal and trap coloring 
    ax.add_patch(Rectangle((goalState[0]-gridAdjust, goalState[1]-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
    
    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='black', alpha=.1))
    
    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color = "black")
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color = "black")

    #labeled values
    for (statex, statey), val in valueTable.items():
        plt.text(statex-.2, statey, str(round(val, 3)))    

    plt.show()


def visualizePolicy(gridWidth, gridHeight, goalState, trapStates, policy):
    #grid height/width
    gridAdjust = .5
    gridScale = 1.5
    arrowScale = .5
    
    xs = np.linspace(-gridAdjust, gridWidth-gridAdjust, gridWidth+1)
    ys = np.linspace(-gridAdjust, gridHeight-gridAdjust, gridHeight+1)
    
    plt.rcParams["figure.figsize"] = [gridWidth*gridScale,gridHeight*gridScale]
    ax = plt.gca(frameon=False, xticks = range(gridWidth), yticks = range(gridHeight))

    #goal and trap coloring 
    ax.add_patch(Rectangle((goalState[0]-gridAdjust, goalState[1]-gridAdjust), 1, 1, fill=True, color='green', alpha=.1))
    
    for (trapx, trapy) in trapStates:
        ax.add_patch(Rectangle((trapx-gridAdjust, trapy-gridAdjust), 1, 1, fill=True, color='black', alpha=.1))

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color = "black")
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color = "black")

    #labeled values
    for (statex, statey), actionDict in policy.items():
        for (optimalActionX, optimalActionY), actionProb in actionDict.items():
            plt.arrow(statex, statey, optimalActionX*actionProb*arrowScale, optimalActionY*actionProb*arrowScale, head_width=0.05*actionProb, head_length=0.1*actionProb)    

    plt.show()


def main():
    # Parameters across all goals and environments
    convergence_threshold = 10e-7
    gamma = .9
    beta = 2
    alpha = 5
    
    # Environment specifications
    gridWidth = 7
    gridHeight = 6
    allActions = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]
    trapStates = [(3, 0), (3, 1), (3, 3)]
    goalA = (6, 1)
    goalB = (6, 4)
    goalC = (1, 5)

    # Transitions
    transition = {(0, 0): {(1, 0): {(1, 0): 1}, (0, 1): {(0, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(0, 0): 1}, (0, 0): {(0, 0): 1}}, (0, 1): {(1, 0): {(1, 1): 1}, (0, 1): {(0, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(0, 0): 1}, (0, 0): {(0, 1): 1}}, (0, 2): {(1, 0): {(1, 2): 1}, (0, 1): {(0, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(0, 1): 1}, (0, 0): {(0, 2): 1}}, (0, 3): {(1, 0): {(1, 3): 1}, (0, 1): {(0, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(0, 2): 1}, (0, 0): {(0, 3): 1}}, (0, 4): {(1, 0): {(1, 4): 1}, (0, 1): {(0, 5): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(0, 3): 1}, (0, 0): {(0, 4): 1}}, (0, 5): {(1, 0): {(1, 5): 1}, (0, 1): {(0, 5): 1}, (-1, 0): {(0, 5): 1}, (0, -1): {(0, 4): 1}, (0, 0): {(0, 5): 1}}, (1, 0): {(1, 0): {(2, 0): 1}, (0, 1): {(1, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(1, 0): 1}, (0, 0): {(1, 0): 1}}, (1, 1): {(1, 0): {(2, 1): 1}, (0, 1): {(1, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(1, 0): 1}, (0, 0): {(1, 1): 1}}, (1, 2): {(1, 0): {(2, 2): 1}, (0, 1): {(1, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(1, 1): 1}, (0, 0): {(1, 2): 1}}, (1, 3): {(1, 0): {(2, 3): 1}, (0, 1): {(1, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(1, 2): 1}, (0, 0): {(1, 3): 1}}, (1, 4): {(1, 0): {(2, 4): 1}, (0, 1): {(1, 5): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(1, 3): 1}, (0, 0): {(1, 4): 1}}, (1, 5): {(1, 0): {(2, 5): 1}, (0, 1): {(1, 5): 1}, (-1, 0): {(0, 5): 1}, (0, -1): {(1, 4): 1}, (0, 0): {(1, 5): 1}}, (2, 0): {(1, 0): {(3, 0): 1}, (0, 1): {(2, 1): 1}, (-1, 0): {(1, 0): 1}, (0, -1): {(2, 0): 1}, (0, 0): {(2, 0): 1}}, (2, 1): {(1, 0): {(3, 1): 1}, (0, 1): {(2, 2): 1}, (-1, 0): {(1, 1): 1}, (0, -1): {(2, 0): 1}, (0, 0): {(2, 1): 1}}, (2, 2): {(1, 0): {(3, 2): 1}, (0, 1): {(2, 3): 1}, (-1, 0): {(1, 2): 1}, (0, -1): {(2, 1): 1}, (0, 0): {(2, 2): 1}}, (2, 3): {(1, 0): {(3, 3): 1}, (0, 1): {(2, 4): 1}, (-1, 0): {(1, 3): 1}, (0, -1): {(2, 2): 1}, (0, 0): {(2, 3): 1}}, (2, 4): {(1, 0): {(3, 4): 1}, (0, 1): {(2, 5): 1}, (-1, 0): {(1, 4): 1}, (0, -1): {(2, 3): 1}, (0, 0): {(2, 4): 1}}, (2, 5): {(1, 0): {(3, 5): 1}, (0, 1): {(2, 5): 1}, (-1, 0): {(1, 5): 1}, (0, -1): {(2, 4): 1}, (0, 0): {(2, 5): 1}}, (3, 0): {(1, 0): {(4, 0): 1}, (0, 1): {(3, 1): 1}, (-1, 0): {(2, 0): 1}, (0, -1): {(3, 0): 1}, (0, 0): {(3, 0): 1}}, (3, 1): {(1, 0): {(4, 1): 1}, (0, 1): {(3, 2): 1}, (-1, 0): {(2, 1): 1}, (0, -1): {(3, 0): 1}, (0, 0): {(3, 1): 1}}, (3, 2): {(1, 0): {(4, 2): 1}, (0, 1): {(3, 3): 1}, (-1, 0): {(2, 2): 1}, (0, -1): {(3, 1): 1}, (0, 0): {(3, 2): 1}}, (3, 3): {(1, 0): {(4, 3): 1}, (0, 1): {(3, 4): 1}, (-1, 0): {(2, 3): 1}, (0, -1): {(3, 2): 1}, (0, 0): {(3, 3): 1}}, (3, 4): {(1, 0): {(4, 4): 1}, (0, 1): {(3, 5): 1}, (-1, 0): {(2, 4): 1}, (0, -1): {(3, 3): 1}, (0, 0): {(3, 4): 1}}, (3, 5): {(1, 0): {(4, 5): 1}, (0, 1): {(3, 5): 1}, (-1, 0): {(2, 5): 1}, (0, -1): {(3, 4): 1}, (0, 0): {(3, 5): 1}}, (4, 0): {(1, 0): {(5, 0): 1}, (0, 1): {(4, 1): 1}, (-1, 0): {(3, 0): 1}, (0, -1): {(4, 0): 1}, (0, 0): {(4, 0): 1}}, (4, 1): {(1, 0): {(5, 1): 1}, (0, 1): {(4, 2): 1}, (-1, 0): {(3, 1): 1}, (0, -1): {(4, 0): 1}, (0, 0): {(4, 1): 1}}, (4, 2): {(1, 0): {(5, 2): 1}, (0, 1): {(4, 3): 1}, (-1, 0): {(3, 2): 1}, (0, -1): {(4, 1): 1}, (0, 0): {(4, 2): 1}}, (4, 3): {(1, 0): {(5, 3): 1}, (0, 1): {(4, 4): 1}, (-1, 0): {(3, 3): 1}, (0, -1): {(4, 2): 1}, (0, 0): {(4, 3): 1}}, (4, 4): {(1, 0): {(5, 4): 1}, (0, 1): {(4, 5): 1}, (-1, 0): {(3, 4): 1}, (0, -1): {(4, 3): 1}, (0, 0): {(4, 4): 1}}, (4, 5): {(1, 0): {(5, 5): 1}, (0, 1): {(4, 5): 1}, (-1, 0): {(3, 5): 1}, (0, -1): {(4, 4): 1}, (0, 0): {(4, 5): 1}}, (5, 0): {(1, 0): {(6, 0): 1}, (0, 1): {(5, 1): 1}, (-1, 0): {(4, 0): 1}, (0, -1): {(5, 0): 1}, (0, 0): {(5, 0): 1}}, (5, 1): {(1, 0): {(6, 1): 1}, (0, 1): {(5, 2): 1}, (-1, 0): {(4, 1): 1}, (0, -1): {(5, 0): 1}, (0, 0): {(5, 1): 1}}, (5, 2): {(1, 0): {(6, 2): 1}, (0, 1): {(5, 3): 1}, (-1, 0): {(4, 2): 1}, (0, -1): {(5, 1): 1}, (0, 0): {(5, 2): 1}}, (5, 3): {(1, 0): {(6, 3): 1}, (0, 1): {(5, 4): 1}, (-1, 0): {(4, 3): 1}, (0, -1): {(5, 2): 1}, (0, 0): {(5, 3): 1}}, (5, 4): {(1, 0): {(6, 4): 1}, (0, 1): {(5, 5): 1}, (-1, 0): {(4, 4): 1}, (0, -1): {(5, 3): 1}, (0, 0): {(5, 4): 1}}, (5, 5): {(1, 0): {(6, 5): 1}, (0, 1): {(5, 5): 1}, (-1, 0): {(4, 5): 1}, (0, -1): {(5, 4): 1}, (0, 0): {(5, 5): 1}}, (6, 0): {(1, 0): {(6, 0): 1}, (0, 1): {(6, 1): 1}, (-1, 0): {(5, 0): 1}, (0, -1): {(6, 0): 1}, (0, 0): {(6, 0): 1}}, (6, 1): {(1, 0): {(6, 1): 1}, (0, 1): {(6, 2): 1}, (-1, 0): {(5, 1): 1}, (0, -1): {(6, 0): 1}, (0, 0): {(6, 1): 1}}, (6, 2): {(1, 0): {(6, 2): 1}, (0, 1): {(6, 3): 1}, (-1, 0): {(5, 2): 1}, (0, -1): {(6, 1): 1}, (0, 0): {(6, 2): 1}}, (6, 3): {(1, 0): {(6, 3): 1}, (0, 1): {(6, 4): 1}, (-1, 0): {(5, 3): 1}, (0, -1): {(6, 2): 1}, (0, 0): {(6, 3): 1}}, (6, 4): {(1, 0): {(6, 4): 1}, (0, 1): {(6, 5): 1}, (-1, 0): {(5, 4): 1}, (0, -1): {(6, 3): 1}, (0, 0): {(6, 4): 1}}, (6, 5): {(1, 0): {(6, 5): 1}, (0, 1): {(6, 5): 1}, (-1, 0): {(5, 5): 1}, (0, -1): {(6, 4): 1}, (0, 0): {(6, 5): 1}}}

    # Rewards
    rewardForGoalA = {(0, 0): {(1, 0): {(1, 0): -1}, (0, 1): {(0, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 0): -0.1}}, (0, 1): {(1, 0): {(1, 1): -1}, (0, 1): {(0, 2): -1}, (-1, 0): {(0, 1): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 1): -0.1}}, (0, 2): {(1, 0): {(1, 2): -1}, (0, 1): {(0, 3): -1}, (-1, 0): {(0, 2): -1}, (0, -1): {(0, 1): -1}, (0, 0): {(0, 2): -0.1}}, (0, 3): {(1, 0): {(1, 3): -1}, (0, 1): {(0, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(0, 2): -1}, (0, 0): {(0, 3): -0.1}}, (0, 4): {(1, 0): {(1, 4): -1}, (0, 1): {(0, 5): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(0, 3): -1}, (0, 0): {(0, 4): -0.1}}, (0, 5): {(1, 0): {(1, 5): -1}, (0, 1): {(0, 5): -1}, (-1, 0): {(0, 5): -1}, (0, -1): {(0, 4): -1}, (0, 0): {(0, 5): -0.1}}, (1, 0): {(1, 0): {(2, 0): -1}, (0, 1): {(1, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 0): -0.1}}, (1, 1): {(1, 0): {(2, 1): -1}, (0, 1): {(1, 2): -1}, (-1, 0): {(0, 1): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 1): -0.1}}, (1, 2): {(1, 0): {(2, 2): -1}, (0, 1): {(1, 3): -1}, (-1, 0): {(0, 2): -1}, (0, -1): {(1, 1): -1}, (0, 0): {(1, 2): -0.1}}, (1, 3): {(1, 0): {(2, 3): -1}, (0, 1): {(1, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(1, 2): -1}, (0, 0): {(1, 3): -0.1}}, (1, 4): {(1, 0): {(2, 4): -1}, (0, 1): {(1, 5): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(1, 3): -1}, (0, 0): {(1, 4): -0.1}}, (1, 5): {(1, 0): {(2, 5): -1}, (0, 1): {(1, 5): -1}, (-1, 0): {(0, 5): -1}, (0, -1): {(1, 4): -1}, (0, 0): {(1, 5): -0.1}}, (2, 0): {(1, 0): {(3, 0): -1}, (0, 1): {(2, 1): -1}, (-1, 0): {(1, 0): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 0): -0.1}}, (2, 1): {(1, 0): {(3, 1): -1}, (0, 1): {(2, 2): -1}, (-1, 0): {(1, 1): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 1): -0.1}}, (2, 2): {(1, 0): {(3, 2): -1}, (0, 1): {(2, 3): -1}, (-1, 0): {(1, 2): -1}, (0, -1): {(2, 1): -1}, (0, 0): {(2, 2): -0.1}}, (2, 3): {(1, 0): {(3, 3): -1}, (0, 1): {(2, 4): -1}, (-1, 0): {(1, 3): -1}, (0, -1): {(2, 2): -1}, (0, 0): {(2, 3): -0.1}}, (2, 4): {(1, 0): {(3, 4): -1}, (0, 1): {(2, 5): -1}, (-1, 0): {(1, 4): -1}, (0, -1): {(2, 3): -1}, (0, 0): {(2, 4): -0.1}}, (2, 5): {(1, 0): {(3, 5): -1}, (0, 1): {(2, 5): -1}, (-1, 0): {(1, 5): -1}, (0, -1): {(2, 4): -1}, (0, 0): {(2, 5): -0.1}}, (3, 0): {(1, 0): {(4, 0): -100}, (0, 1): {(3, 1): -100}, (-1, 0): {(2, 0): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 0): -100}}, (3, 1): {(1, 0): {(4, 1): -100}, (0, 1): {(3, 2): -100}, (-1, 0): {(2, 1): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 1): -100}}, (3, 2): {(1, 0): {(4, 2): -1}, (0, 1): {(3, 3): -1}, (-1, 0): {(2, 2): -1}, (0, -1): {(3, 1): -1}, (0, 0): {(3, 2): -0.1}}, (3, 3): {(1, 0): {(4, 3): -100}, (0, 1): {(3, 4): -100}, (-1, 0): {(2, 3): -100}, (0, -1): {(3, 2): -100}, (0, 0): {(3, 3): -100}}, (3, 4): {(1, 0): {(4, 4): -1}, (0, 1): {(3, 5): -1}, (-1, 0): {(2, 4): -1}, (0, -1): {(3, 3): -1}, (0, 0): {(3, 4): -0.1}}, (3, 5): {(1, 0): {(4, 5): -1}, (0, 1): {(3, 5): -1}, (-1, 0): {(2, 5): -1}, (0, -1): {(3, 4): -1}, (0, 0): {(3, 5): -0.1}}, (4, 0): {(1, 0): {(5, 0): -1}, (0, 1): {(4, 1): -1}, (-1, 0): {(3, 0): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 0): -0.1}}, (4, 1): {(1, 0): {(5, 1): -1}, (0, 1): {(4, 2): -1}, (-1, 0): {(3, 1): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 1): -0.1}}, (4, 2): {(1, 0): {(5, 2): -1}, (0, 1): {(4, 3): -1}, (-1, 0): {(3, 2): -1}, (0, -1): {(4, 1): -1}, (0, 0): {(4, 2): -0.1}}, (4, 3): {(1, 0): {(5, 3): -1}, (0, 1): {(4, 4): -1}, (-1, 0): {(3, 3): -1}, (0, -1): {(4, 2): -1}, (0, 0): {(4, 3): -0.1}}, (4, 4): {(1, 0): {(5, 4): -1}, (0, 1): {(4, 5): -1}, (-1, 0): {(3, 4): -1}, (0, -1): {(4, 3): -1}, (0, 0): {(4, 4): -0.1}}, (4, 5): {(1, 0): {(5, 5): -1}, (0, 1): {(4, 5): -1}, (-1, 0): {(3, 5): -1}, (0, -1): {(4, 4): -1}, (0, 0): {(4, 5): -0.1}}, (5, 0): {(1, 0): {(6, 0): -1}, (0, 1): {(5, 1): -1}, (-1, 0): {(4, 0): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 0): -0.1}}, (5, 1): {(1, 0): {(6, 1): -1}, (0, 1): {(5, 2): -1}, (-1, 0): {(4, 1): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 1): -0.1}}, (5, 2): {(1, 0): {(6, 2): -1}, (0, 1): {(5, 3): -1}, (-1, 0): {(4, 2): -1}, (0, -1): {(5, 1): -1}, (0, 0): {(5, 2): -0.1}}, (5, 3): {(1, 0): {(6, 3): -1}, (0, 1): {(5, 4): -1}, (-1, 0): {(4, 3): -1}, (0, -1): {(5, 2): -1}, (0, 0): {(5, 3): -0.1}}, (5, 4): {(1, 0): {(6, 4): -1}, (0, 1): {(5, 5): -1}, (-1, 0): {(4, 4): -1}, (0, -1): {(5, 3): -1}, (0, 0): {(5, 4): -0.1}}, (5, 5): {(1, 0): {(6, 5): -1}, (0, 1): {(5, 5): -1}, (-1, 0): {(4, 5): -1}, (0, -1): {(5, 4): -1}, (0, 0): {(5, 5): -0.1}}, (6, 0): {(1, 0): {(6, 0): -1}, (0, 1): {(6, 1): -1}, (-1, 0): {(5, 0): -1}, (0, -1): {(6, 0): -1}, (0, 0): {(6, 0): -0.1}}, (6, 1): {(1, 0): {(6, 1): 9}, (0, 1): {(6, 2): 9}, (-1, 0): {(5, 1): 9}, (0, -1): {(6, 0): 9}, (0, 0): {(6, 1): 9.9}}, (6, 2): {(1, 0): {(6, 2): -1}, (0, 1): {(6, 3): -1}, (-1, 0): {(5, 2): -1}, (0, -1): {(6, 1): -1}, (0, 0): {(6, 2): -0.1}}, (6, 3): {(1, 0): {(6, 3): -1}, (0, 1): {(6, 4): -1}, (-1, 0): {(5, 3): -1}, (0, -1): {(6, 2): -1}, (0, 0): {(6, 3): -0.1}}, (6, 4): {(1, 0): {(6, 4): -1}, (0, 1): {(6, 5): -1}, (-1, 0): {(5, 4): -1}, (0, -1): {(6, 3): -1}, (0, 0): {(6, 4): -0.1}}, (6, 5): {(1, 0): {(6, 5): -1}, (0, 1): {(6, 5): -1}, (-1, 0): {(5, 5): -1}, (0, -1): {(6, 4): -1}, (0, 0): {(6, 5): -0.1}}}
    rewardForGoalB = {(0, 0): {(1, 0): {(1, 0): -1}, (0, 1): {(0, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 0): -0.1}}, (0, 1): {(1, 0): {(1, 1): -1}, (0, 1): {(0, 2): -1}, (-1, 0): {(0, 1): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 1): -0.1}}, (0, 2): {(1, 0): {(1, 2): -1}, (0, 1): {(0, 3): -1}, (-1, 0): {(0, 2): -1}, (0, -1): {(0, 1): -1}, (0, 0): {(0, 2): -0.1}}, (0, 3): {(1, 0): {(1, 3): -1}, (0, 1): {(0, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(0, 2): -1}, (0, 0): {(0, 3): -0.1}}, (0, 4): {(1, 0): {(1, 4): -1}, (0, 1): {(0, 5): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(0, 3): -1}, (0, 0): {(0, 4): -0.1}}, (0, 5): {(1, 0): {(1, 5): -1}, (0, 1): {(0, 5): -1}, (-1, 0): {(0, 5): -1}, (0, -1): {(0, 4): -1}, (0, 0): {(0, 5): -0.1}}, (1, 0): {(1, 0): {(2, 0): -1}, (0, 1): {(1, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 0): -0.1}}, (1, 1): {(1, 0): {(2, 1): -1}, (0, 1): {(1, 2): -1}, (-1, 0): {(0, 1): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 1): -0.1}}, (1, 2): {(1, 0): {(2, 2): -1}, (0, 1): {(1, 3): -1}, (-1, 0): {(0, 2): -1}, (0, -1): {(1, 1): -1}, (0, 0): {(1, 2): -0.1}}, (1, 3): {(1, 0): {(2, 3): -1}, (0, 1): {(1, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(1, 2): -1}, (0, 0): {(1, 3): -0.1}}, (1, 4): {(1, 0): {(2, 4): -1}, (0, 1): {(1, 5): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(1, 3): -1}, (0, 0): {(1, 4): -0.1}}, (1, 5): {(1, 0): {(2, 5): -1}, (0, 1): {(1, 5): -1}, (-1, 0): {(0, 5): -1}, (0, -1): {(1, 4): -1}, (0, 0): {(1, 5): -0.1}}, (2, 0): {(1, 0): {(3, 0): -1}, (0, 1): {(2, 1): -1}, (-1, 0): {(1, 0): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 0): -0.1}}, (2, 1): {(1, 0): {(3, 1): -1}, (0, 1): {(2, 2): -1}, (-1, 0): {(1, 1): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 1): -0.1}}, (2, 2): {(1, 0): {(3, 2): -1}, (0, 1): {(2, 3): -1}, (-1, 0): {(1, 2): -1}, (0, -1): {(2, 1): -1}, (0, 0): {(2, 2): -0.1}}, (2, 3): {(1, 0): {(3, 3): -1}, (0, 1): {(2, 4): -1}, (-1, 0): {(1, 3): -1}, (0, -1): {(2, 2): -1}, (0, 0): {(2, 3): -0.1}}, (2, 4): {(1, 0): {(3, 4): -1}, (0, 1): {(2, 5): -1}, (-1, 0): {(1, 4): -1}, (0, -1): {(2, 3): -1}, (0, 0): {(2, 4): -0.1}}, (2, 5): {(1, 0): {(3, 5): -1}, (0, 1): {(2, 5): -1}, (-1, 0): {(1, 5): -1}, (0, -1): {(2, 4): -1}, (0, 0): {(2, 5): -0.1}}, (3, 0): {(1, 0): {(4, 0): -100}, (0, 1): {(3, 1): -100}, (-1, 0): {(2, 0): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 0): -100}}, (3, 1): {(1, 0): {(4, 1): -100}, (0, 1): {(3, 2): -100}, (-1, 0): {(2, 1): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 1): -100}}, (3, 2): {(1, 0): {(4, 2): -1}, (0, 1): {(3, 3): -1}, (-1, 0): {(2, 2): -1}, (0, -1): {(3, 1): -1}, (0, 0): {(3, 2): -0.1}}, (3, 3): {(1, 0): {(4, 3): -100}, (0, 1): {(3, 4): -100}, (-1, 0): {(2, 3): -100}, (0, -1): {(3, 2): -100}, (0, 0): {(3, 3): -100}}, (3, 4): {(1, 0): {(4, 4): -1}, (0, 1): {(3, 5): -1}, (-1, 0): {(2, 4): -1}, (0, -1): {(3, 3): -1}, (0, 0): {(3, 4): -0.1}}, (3, 5): {(1, 0): {(4, 5): -1}, (0, 1): {(3, 5): -1}, (-1, 0): {(2, 5): -1}, (0, -1): {(3, 4): -1}, (0, 0): {(3, 5): -0.1}}, (4, 0): {(1, 0): {(5, 0): -1}, (0, 1): {(4, 1): -1}, (-1, 0): {(3, 0): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 0): -0.1}}, (4, 1): {(1, 0): {(5, 1): -1}, (0, 1): {(4, 2): -1}, (-1, 0): {(3, 1): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 1): -0.1}}, (4, 2): {(1, 0): {(5, 2): -1}, (0, 1): {(4, 3): -1}, (-1, 0): {(3, 2): -1}, (0, -1): {(4, 1): -1}, (0, 0): {(4, 2): -0.1}}, (4, 3): {(1, 0): {(5, 3): -1}, (0, 1): {(4, 4): -1}, (-1, 0): {(3, 3): -1}, (0, -1): {(4, 2): -1}, (0, 0): {(4, 3): -0.1}}, (4, 4): {(1, 0): {(5, 4): -1}, (0, 1): {(4, 5): -1}, (-1, 0): {(3, 4): -1}, (0, -1): {(4, 3): -1}, (0, 0): {(4, 4): -0.1}}, (4, 5): {(1, 0): {(5, 5): -1}, (0, 1): {(4, 5): -1}, (-1, 0): {(3, 5): -1}, (0, -1): {(4, 4): -1}, (0, 0): {(4, 5): -0.1}}, (5, 0): {(1, 0): {(6, 0): -1}, (0, 1): {(5, 1): -1}, (-1, 0): {(4, 0): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 0): -0.1}}, (5, 1): {(1, 0): {(6, 1): -1}, (0, 1): {(5, 2): -1}, (-1, 0): {(4, 1): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 1): -0.1}}, (5, 2): {(1, 0): {(6, 2): -1}, (0, 1): {(5, 3): -1}, (-1, 0): {(4, 2): -1}, (0, -1): {(5, 1): -1}, (0, 0): {(5, 2): -0.1}}, (5, 3): {(1, 0): {(6, 3): -1}, (0, 1): {(5, 4): -1}, (-1, 0): {(4, 3): -1}, (0, -1): {(5, 2): -1}, (0, 0): {(5, 3): -0.1}}, (5, 4): {(1, 0): {(6, 4): -1}, (0, 1): {(5, 5): -1}, (-1, 0): {(4, 4): -1}, (0, -1): {(5, 3): -1}, (0, 0): {(5, 4): -0.1}}, (5, 5): {(1, 0): {(6, 5): -1}, (0, 1): {(5, 5): -1}, (-1, 0): {(4, 5): -1}, (0, -1): {(5, 4): -1}, (0, 0): {(5, 5): -0.1}}, (6, 0): {(1, 0): {(6, 0): -1}, (0, 1): {(6, 1): -1}, (-1, 0): {(5, 0): -1}, (0, -1): {(6, 0): -1}, (0, 0): {(6, 0): -0.1}}, (6, 1): {(1, 0): {(6, 1): -1}, (0, 1): {(6, 2): -1}, (-1, 0): {(5, 1): -1}, (0, -1): {(6, 0): -1}, (0, 0): {(6, 1): -0.1}}, (6, 2): {(1, 0): {(6, 2): -1}, (0, 1): {(6, 3): -1}, (-1, 0): {(5, 2): -1}, (0, -1): {(6, 1): -1}, (0, 0): {(6, 2): -0.1}}, (6, 3): {(1, 0): {(6, 3): -1}, (0, 1): {(6, 4): -1}, (-1, 0): {(5, 3): -1}, (0, -1): {(6, 2): -1}, (0, 0): {(6, 3): -0.1}}, (6, 4): {(1, 0): {(6, 4): 9}, (0, 1): {(6, 5): 9}, (-1, 0): {(5, 4): 9}, (0, -1): {(6, 3): 9}, (0, 0): {(6, 4): 9.9}}, (6, 5): {(1, 0): {(6, 5): -1}, (0, 1): {(6, 5): -1}, (-1, 0): {(5, 5): -1}, (0, -1): {(6, 4): -1}, (0, 0): {(6, 5): -0.1}}}
    rewardForGoalC = {(0, 0): {(1, 0): {(1, 0): -1}, (0, 1): {(0, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 0): -0.1}}, (0, 1): {(1, 0): {(1, 1): -1}, (0, 1): {(0, 2): -1}, (-1, 0): {(0, 1): -1}, (0, -1): {(0, 0): -1}, (0, 0): {(0, 1): -0.1}}, (0, 2): {(1, 0): {(1, 2): -1}, (0, 1): {(0, 3): -1}, (-1, 0): {(0, 2): -1}, (0, -1): {(0, 1): -1}, (0, 0): {(0, 2): -0.1}}, (0, 3): {(1, 0): {(1, 3): -1}, (0, 1): {(0, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(0, 2): -1}, (0, 0): {(0, 3): -0.1}}, (0, 4): {(1, 0): {(1, 4): -1}, (0, 1): {(0, 5): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(0, 3): -1}, (0, 0): {(0, 4): -0.1}}, (0, 5): {(1, 0): {(1, 5): -1}, (0, 1): {(0, 5): -1}, (-1, 0): {(0, 5): -1}, (0, -1): {(0, 4): -1}, (0, 0): {(0, 5): -0.1}}, (1, 0): {(1, 0): {(2, 0): -1}, (0, 1): {(1, 1): -1}, (-1, 0): {(0, 0): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 0): -0.1}}, (1, 1): {(1, 0): {(2, 1): -1}, (0, 1): {(1, 2): -1}, (-1, 0): {(0, 1): -1}, (0, -1): {(1, 0): -1}, (0, 0): {(1, 1): -0.1}}, (1, 2): {(1, 0): {(2, 2): -1}, (0, 1): {(1, 3): -1}, (-1, 0): {(0, 2): -1}, (0, -1): {(1, 1): -1}, (0, 0): {(1, 2): -0.1}}, (1, 3): {(1, 0): {(2, 3): -1}, (0, 1): {(1, 4): -1}, (-1, 0): {(0, 3): -1}, (0, -1): {(1, 2): -1}, (0, 0): {(1, 3): -0.1}}, (1, 4): {(1, 0): {(2, 4): -1}, (0, 1): {(1, 5): -1}, (-1, 0): {(0, 4): -1}, (0, -1): {(1, 3): -1}, (0, 0): {(1, 4): -0.1}}, (1, 5): {(1, 0): {(2, 5): 9}, (0, 1): {(1, 5): 9}, (-1, 0): {(0, 5): 9}, (0, -1): {(1, 4): 9}, (0, 0): {(1, 5): 9.9}}, (2, 0): {(1, 0): {(3, 0): -1}, (0, 1): {(2, 1): -1}, (-1, 0): {(1, 0): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 0): -0.1}}, (2, 1): {(1, 0): {(3, 1): -1}, (0, 1): {(2, 2): -1}, (-1, 0): {(1, 1): -1}, (0, -1): {(2, 0): -1}, (0, 0): {(2, 1): -0.1}}, (2, 2): {(1, 0): {(3, 2): -1}, (0, 1): {(2, 3): -1}, (-1, 0): {(1, 2): -1}, (0, -1): {(2, 1): -1}, (0, 0): {(2, 2): -0.1}}, (2, 3): {(1, 0): {(3, 3): -1}, (0, 1): {(2, 4): -1}, (-1, 0): {(1, 3): -1}, (0, -1): {(2, 2): -1}, (0, 0): {(2, 3): -0.1}}, (2, 4): {(1, 0): {(3, 4): -1}, (0, 1): {(2, 5): -1}, (-1, 0): {(1, 4): -1}, (0, -1): {(2, 3): -1}, (0, 0): {(2, 4): -0.1}}, (2, 5): {(1, 0): {(3, 5): -1}, (0, 1): {(2, 5): -1}, (-1, 0): {(1, 5): -1}, (0, -1): {(2, 4): -1}, (0, 0): {(2, 5): -0.1}}, (3, 0): {(1, 0): {(4, 0): -100}, (0, 1): {(3, 1): -100}, (-1, 0): {(2, 0): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 0): -100}}, (3, 1): {(1, 0): {(4, 1): -100}, (0, 1): {(3, 2): -100}, (-1, 0): {(2, 1): -100}, (0, -1): {(3, 0): -100}, (0, 0): {(3, 1): -100}}, (3, 2): {(1, 0): {(4, 2): -1}, (0, 1): {(3, 3): -1}, (-1, 0): {(2, 2): -1}, (0, -1): {(3, 1): -1}, (0, 0): {(3, 2): -0.1}}, (3, 3): {(1, 0): {(4, 3): -100}, (0, 1): {(3, 4): -100}, (-1, 0): {(2, 3): -100}, (0, -1): {(3, 2): -100}, (0, 0): {(3, 3): -100}}, (3, 4): {(1, 0): {(4, 4): -1}, (0, 1): {(3, 5): -1}, (-1, 0): {(2, 4): -1}, (0, -1): {(3, 3): -1}, (0, 0): {(3, 4): -0.1}}, (3, 5): {(1, 0): {(4, 5): -1}, (0, 1): {(3, 5): -1}, (-1, 0): {(2, 5): -1}, (0, -1): {(3, 4): -1}, (0, 0): {(3, 5): -0.1}}, (4, 0): {(1, 0): {(5, 0): -1}, (0, 1): {(4, 1): -1}, (-1, 0): {(3, 0): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 0): -0.1}}, (4, 1): {(1, 0): {(5, 1): -1}, (0, 1): {(4, 2): -1}, (-1, 0): {(3, 1): -1}, (0, -1): {(4, 0): -1}, (0, 0): {(4, 1): -0.1}}, (4, 2): {(1, 0): {(5, 2): -1}, (0, 1): {(4, 3): -1}, (-1, 0): {(3, 2): -1}, (0, -1): {(4, 1): -1}, (0, 0): {(4, 2): -0.1}}, (4, 3): {(1, 0): {(5, 3): -1}, (0, 1): {(4, 4): -1}, (-1, 0): {(3, 3): -1}, (0, -1): {(4, 2): -1}, (0, 0): {(4, 3): -0.1}}, (4, 4): {(1, 0): {(5, 4): -1}, (0, 1): {(4, 5): -1}, (-1, 0): {(3, 4): -1}, (0, -1): {(4, 3): -1}, (0, 0): {(4, 4): -0.1}}, (4, 5): {(1, 0): {(5, 5): -1}, (0, 1): {(4, 5): -1}, (-1, 0): {(3, 5): -1}, (0, -1): {(4, 4): -1}, (0, 0): {(4, 5): -0.1}}, (5, 0): {(1, 0): {(6, 0): -1}, (0, 1): {(5, 1): -1}, (-1, 0): {(4, 0): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 0): -0.1}}, (5, 1): {(1, 0): {(6, 1): -1}, (0, 1): {(5, 2): -1}, (-1, 0): {(4, 1): -1}, (0, -1): {(5, 0): -1}, (0, 0): {(5, 1): -0.1}}, (5, 2): {(1, 0): {(6, 2): -1}, (0, 1): {(5, 3): -1}, (-1, 0): {(4, 2): -1}, (0, -1): {(5, 1): -1}, (0, 0): {(5, 2): -0.1}}, (5, 3): {(1, 0): {(6, 3): -1}, (0, 1): {(5, 4): -1}, (-1, 0): {(4, 3): -1}, (0, -1): {(5, 2): -1}, (0, 0): {(5, 3): -0.1}}, (5, 4): {(1, 0): {(6, 4): -1}, (0, 1): {(5, 5): -1}, (-1, 0): {(4, 4): -1}, (0, -1): {(5, 3): -1}, (0, 0): {(5, 4): -0.1}}, (5, 5): {(1, 0): {(6, 5): -1}, (0, 1): {(5, 5): -1}, (-1, 0): {(4, 5): -1}, (0, -1): {(5, 4): -1}, (0, 0): {(5, 5): -0.1}}, (6, 0): {(1, 0): {(6, 0): -1}, (0, 1): {(6, 1): -1}, (-1, 0): {(5, 0): -1}, (0, -1): {(6, 0): -1}, (0, 0): {(6, 0): -0.1}}, (6, 1): {(1, 0): {(6, 1): -1}, (0, 1): {(6, 2): -1}, (-1, 0): {(5, 1): -1}, (0, -1): {(6, 0): -1}, (0, 0): {(6, 1): -0.1}}, (6, 2): {(1, 0): {(6, 2): -1}, (0, 1): {(6, 3): -1}, (-1, 0): {(5, 2): -1}, (0, -1): {(6, 1): -1}, (0, 0): {(6, 2): -0.1}}, (6, 3): {(1, 0): {(6, 3): -1}, (0, 1): {(6, 4): -1}, (-1, 0): {(5, 3): -1}, (0, -1): {(6, 2): -1}, (0, 0): {(6, 3): -0.1}}, (6, 4): {(1, 0): {(6, 4): -1}, (0, 1): {(6, 5): -1}, (-1, 0): {(5, 4): -1}, (0, -1): {(6, 3): -1}, (0, 0): {(6, 4): -0.1}}, (6, 5): {(1, 0): {(6, 5): -1}, (0, 1): {(6, 5): -1}, (-1, 0): {(5, 5): -1}, (0, -1): {(6, 4): -1}, (0, 0): {(6, 5): -0.1}}}

    value_table_initial = {
        (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0, (0, 5): 0,
        (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0, (1, 5): 0,
        (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 0, (2, 5): 0,
        (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 0, (3, 5): 0,
        (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0, (4, 5): 0,
        (5, 0): 0, (5, 1): 0, (5, 2): 0, (5, 3): 0, (5, 4): 0, (5, 5): 0,
        (6, 0): 0, (6, 1): 0, (6, 2): 0, (6, 3): 0, (6, 4): 0, (6, 5): 0}

    # Goal A
    perform_value_iteration_goal_a_original = ValueIteration(transition, rewardForGoalA, value_table_initial,
                                                             convergence_threshold, gamma, use_softmax=True,
                                                             use_noise=True, noise_beta=beta)
    optimal_value_table_a_original, optimal_policy_table_a_original = perform_value_iteration_goal_a_original()

    visualizeValueTable(gridWidth, gridHeight, goalA, trapStates, optimal_value_table_a_original)
    visualizePolicy(gridWidth, gridHeight, goalA, trapStates, optimal_policy_table_a_original)

    # Goal B
    perform_value_iteration_goal_b_original = ValueIteration(transition, rewardForGoalB, value_table_initial,
                                                             convergence_threshold, gamma, use_softmax=True,
                                                             use_noise=True, noise_beta=beta)
    optimal_value_table_b_original, optimal_policy_table_b_original = perform_value_iteration_goal_b_original()

    visualizeValueTable(gridWidth, gridHeight, goalB, trapStates, optimal_value_table_b_original)
    visualizePolicy(gridWidth, gridHeight, goalB, trapStates, optimal_policy_table_b_original)

    # Goal C
    perform_value_iteration_goal_c_original = ValueIteration(transition, rewardForGoalC, value_table_initial,
                                                             convergence_threshold, gamma, use_softmax=True,
                                                             use_noise=True, noise_beta=beta)
    optimal_value_table_c_original, optimal_policy_table_c_original = perform_value_iteration_goal_c_original()

    visualizeValueTable(gridWidth, gridHeight, goalC, trapStates, optimal_value_table_c_original)
    visualizePolicy(gridWidth, gridHeight, goalC, trapStates, optimal_policy_table_c_original)

    # Goal Policies
    goal_policies = dict()

    goal_policies[goalA] = optimal_policy_table_a_original
    goal_policies[goalB] = optimal_policy_table_b_original
    goal_policies[goalC] = optimal_policy_table_c_original

    print('Goals')
    print(goal_policies)

    perform_get_likelihood_reward = GetLikelihoodReward(transition, goal_policies)

    # Goal A
    new_reward_a = perform_get_likelihood_reward(goalA, rewardForGoalA, alpha)

    print(new_reward_a)

    perform_value_iteration_goal_a_new = ValueIteration(transition, new_reward_a, value_table_initial,
                                                        convergence_threshold, gamma, use_softmax=True,
                                                        use_noise=True, noise_beta=beta)
    optimal_value_table_a_new, optimal_policy_table_a_new = perform_value_iteration_goal_a_new()

    visualizeValueTable(gridWidth, gridHeight, goalA, trapStates, optimal_value_table_a_new)
    visualizePolicy(gridWidth, gridHeight, goalA, trapStates, optimal_policy_table_a_new)

    # Goal B
    new_reward_b = perform_get_likelihood_reward(goalB, rewardForGoalB, alpha)

    print(new_reward_b)

    perform_value_iteration_goal_b_new = ValueIteration(transition, new_reward_b, value_table_initial,
                                                        convergence_threshold, gamma, use_softmax=True,
                                                        use_noise=True, noise_beta=beta)
    optimal_value_table_b_new, optimal_policy_table_b_new = perform_value_iteration_goal_b_new()

    visualizeValueTable(gridWidth, gridHeight, goalB, trapStates, optimal_value_table_b_new)
    visualizePolicy(gridWidth, gridHeight, goalB, trapStates, optimal_policy_table_b_new)

    # Goal C
    new_reward_c = perform_get_likelihood_reward(goalC, rewardForGoalC, alpha)

    print(new_reward_c)

    perform_value_iteration_goal_c_new = ValueIteration(transition, new_reward_c, value_table_initial,
                                                        convergence_threshold, gamma, use_softmax=True,
                                                        use_noise=True, noise_beta=beta)
    optimal_value_table_c_new, optimal_policy_table_c_new = perform_value_iteration_goal_c_new()

    visualizeValueTable(gridWidth, gridHeight, goalC, trapStates, optimal_value_table_c_new)
    visualizePolicy(gridWidth, gridHeight, goalC, trapStates, optimal_policy_table_c_new)

    # # Calculating the Marginal Probability Tables for Each Goal in Environment
    # marginal_probability_next_state_a_original = get_probability_of_individual_state_transitions(
    #     transition, optimal_policy_table_a_original)
    #
    # marginal_probability_next_state_b_original = get_probability_of_individual_state_transitions(
    #     transition, optimal_policy_table_b_original)
    #
    # marginal_probability_next_state_c_original = get_probability_of_individual_state_transitions(
    #     transition, optimal_policy_table_c_original)
    #
    # # Marginal Probabilities of the Next State Tables (Combined into a single array)
    # marginal_probability_next_state_tables_original = [marginal_probability_next_state_a_original,
    #                                                    marginal_probability_next_state_b_original,
    #                                                    marginal_probability_next_state_c_original]
    #
    # print('marginal original')
    # print(marginal_probability_next_state_tables_original)


if __name__=="__main__":
    main()



