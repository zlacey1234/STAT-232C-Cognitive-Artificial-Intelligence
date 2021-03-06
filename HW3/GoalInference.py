""" GoalInferenceTest.py

Class: STAT 232C - Cognitive Artificial Intelligence
Project 3: Goal Inference
Name: Zachary Lacey
Date: May 3rd, 2021

"""

__author__ = 'Zachary Lacey'

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from HW1.BayesianInference import *
from HW2.ValueIteration import ValueIteration

value_table_initial = {
        (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0, (0, 5): 0,
        (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0, (1, 5): 0,
        (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 0, (2, 5): 0,
        (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 0, (3, 5): 0,
        (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0, (4, 5): 0,
        (5, 0): 0, (5, 1): 0, (5, 2): 0, (5, 3): 0, (5, 4): 0, (5, 5): 0,
        (6, 0): 0, (6, 1): 0, (6, 2): 0, (6, 3): 0, (6, 4): 0, (6, 5): 0}


def visualize_value_table(grid_width, grid_height, goal_state, trap_states, value_table):
    grid_adjust = .5
    grid_scale = 1.5

    xs = np.linspace(-grid_adjust, grid_width - grid_adjust, grid_width + 1)
    ys = np.linspace(-grid_adjust, grid_height - grid_adjust, grid_height + 1)

    plt.rcParams["figure.figsize"] = [grid_width * grid_scale, grid_height * grid_scale]
    ax = plt.gca(frameon=False, xticks=range(grid_width), yticks=range(grid_height))

    # goal and trap coloring
    ax.add_patch(Rectangle((goal_state[0]-grid_adjust, goal_state[1]-grid_adjust), 1, 1,
                           fill=True, color='green', alpha=.1))

    for (trapx, trapy) in trap_states:
        ax.add_patch(Rectangle((trapx - grid_adjust, trapy - grid_adjust), 1, 1, fill=True, color='black', alpha=.1))

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color="black")
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color="black")

    # labeled values
    for (statex, statey), val in value_table.items():
        plt.text(statex - .2, statey, str(round(val, 3)))

    plt.show()


def visualize_policy(grid_width, grid_height, goal_state, trap_states, policy):
    # grid height/width
    grid_adjust = .5
    grid_scale = 1.5
    arrow_scale = .5

    xs = np.linspace(-grid_adjust, grid_width - grid_adjust, grid_width + 1)
    ys = np.linspace(-grid_adjust, grid_height - grid_adjust, grid_height + 1)

    plt.rcParams["figure.figsize"] = [grid_width * grid_scale, grid_height * grid_scale]
    ax = plt.gca(frameon=False, xticks=range(grid_width), yticks=range(grid_height))

    # goal and trap coloring
    ax.add_patch(Rectangle((goal_state[0]-grid_adjust, goal_state[1]-grid_adjust), 1, 1,
                           fill=True, color='green', alpha=.1))

    for (trapx, trapy) in trap_states:
        ax.add_patch(Rectangle((trapx - grid_adjust, trapy - grid_adjust), 1, 1, fill=True, color='black', alpha=.1))

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color="black")
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color="black")

    # labeled values
    for (statex, statey), actionDict in policy.items():
        for (optimalActionX, optimalActionY), actionProb in actionDict.items():
            plt.arrow(statex, statey, optimalActionX * actionProb * arrow_scale,
                      optimalActionY * actionProb * arrow_scale,
                      head_width=0.05 * actionProb, head_length=0.1 * actionProb)

    plt.show()


def get_probability_of_individual_state_transitions(transition_table, policy_table):
    """
    Get Probability of Individual State Transitions Method

    This method performs the calculations of the probabilities of the individual state transitions (current-state
    [s_{t}] to the next-state [s_{t + 1}]), given the environment and the goal.

    Args:
        transition_table (datatype: dict): This is a nested dictionary of the state-action-nextstate combinations
        for transitions that have a non-zero probability. The transition table can be dependent on the given
        environment [w].The transition_table has the following structure:

            transition_table = {state: {action: {nextstate: {probability}}}}  == P(s' | s, a) == P(s' | s, a, w)

        policy_table (datatype: dict): This is a nested dictionary of form state-action-probability giving the
        approximate optimal policy of an agent. The policy table can be dependent on the given environment [w] and
        the goal [g].

            policy_table = {state: {action: {probability}}} == pi(a | s) == P_pi(a | s, g, w)

    Returns:
        marginal_probability_next_state (datatype: dict): This is a nested dictionary of form state-nextstate
        combinations for a given goal (via it's policy table) and environment (map). This essentially the probability
        of the nextstate given the current state, goal, and environment and is calculated by marginalizing over the
        possible actions.

            P(s' | s, g, w) = sum[across a] ( P(s' | s, a, w) * P_pi(a | s, g, w) )

            marginal_probability_next_state = {state: {nextstate: {probability}}} == P(s' | s, g, w)

    """
    states = list(transition_table)
    num_states = len(states)

    # Dictionary to store Marginal Probabilities of the Next State given the Current State, Goal and Environment
    marginal_probability_next_state = dict()

    # For each state
    for s in range(num_states):
        actions = list(transition_table[states[s]])
        num_actions = len(actions)

        probability_next_state = dict()
        possible_next_states = []

        # For each action
        for a in range(num_actions):

            state_prime = list(transition_table[states[s]][actions[a]])
            num_states_prime = len(state_prime)

            # For each next-state
            for sp in range(num_states_prime):
                # If Next State is in the list of possible next states
                if state_prime[sp] in possible_next_states:
                    # Accumulation of repeated next state probabilities
                    probability_next_state[state_prime[sp]] = \
                        transition_table[states[s]][actions[a]][state_prime[sp]] * policy_table[states[s]][actions[a]] \
                        + probability_next_state[state_prime[sp]]

                else:
                    # Add this to the list of possible next states
                    possible_next_states.append(state_prime[sp])

                    probability_next_state[state_prime[sp]] = \
                        transition_table[states[s]][actions[a]][state_prime[sp]] * policy_table[states[s]][actions[a]]

        # Store in the (Marginal Probabilities of the Next State) Dictionary
        marginal_probability_next_state[states[s]] = probability_next_state

    return marginal_probability_next_state


def get_likelihood_entire_path_sequence(observed_sequence, marginal_probability_next_state):
    """
    Get Likelihood Entire Path Sequence Method

    This finds the likelihood of the entire Observed Trajectory Sequence.

    Args:
        observed_sequence (datatype: array): This is an array that contains the sequential states of trajectory. This
        is specifically referred to as the trajectory sequence [S_{1:T}]. [T] is the termination state of the
        trajectory sequence.

        marginal_probability_next_state (datatype: dict): This is a nested dictionary of form state-nextstate
        combinations for a given goal (via it's policy table) and environment (map). This essentially the probability
        of the nextstate given the current state, goal, and environment and is calculated by marginalizing over the
        possible actions. P(s' | s, g, w) = sum[across a] (P(s' | s, a, w) * P_pi(a | s, g, w))

            marginal_probability_next_state = {state: {nextstate: {probability}}} == P(s' | s, g, w)

    Returns:
        likelihood_sequence (datatype: float): This is the likelihood value that the observed trajectory sequence
        occurs given the specified goal and environment. In particular, this returns the likelihood at the
        time step [t].

        likelihood_sequence_history (datatype: array): This is an array that contains the updated likelihood values
        at each individual time steps in the sequence.

            likelihood_sequence_history = [likelihood([t=1]),
                                           likelihood([t=2]),
                                           likelihood([t=3]),
                                                         ...,
                                           likelihood([t=T-1])]

    """
    num_time_steps_trajectory = len(observed_sequence)

    likelihood_sequence, likelihood_sequence_history = get_likelihood_part_of_path_sequence(
        observed_sequence, marginal_probability_next_state, t=(num_time_steps_trajectory - 1))

    return likelihood_sequence, likelihood_sequence_history


def get_likelihood_part_of_path_sequence(observed_sequence, marginal_probability_next_state, t=2):
    """
    Get Likelihood Part of Path Sequence Method

    This method calculates the likelihood of a sequence of observed actions (observed trajectory sequence) using a
    recursive algorithm. This allows us store and return the updated likelihood for each time step.

    Args:
        observed_sequence (datatype: array): This is an array that contains the sequential states of trajectory. This
        is specifically referred to as the trajectory sequence [S_{1:T}]. [T] is the termination state of the
        trajectory sequence.

        marginal_probability_next_state (datatype: dict): This is a nested dictionary of form state-nextstate
        combinations for a given goal (via it's policy table) and environment (map). This essentially the probability
        of the nextstate given the current state, goal, and environment and is calculated by marginalizing over the
        possible actions. P(s' | s, g, w) = sum[across a] (P(s' | s, a, w) * P_pi(a | s, g, w))

            marginal_probability_next_state = {state: {nextstate: {probability}}} == P(s' | s, g, w)

        t (datatype: integer) [optional] (default value: 2): This is an integer value that specifies the number of time
        steps that we want to consider. This means, if t = 5, we calculate the likelihood considering the first five
        time steps.

    Returns:
        likelihood (datatype: float): This is the likelihood value that the observed trajectory sequence occurs given
        the specified goal and environment. In particular, this returns the likelihood at the time step [t].

        likelihood_history (datatype: array): This is an array that contains the updated likelihood values at each
        individual time steps in the sequence.

            likelhood_history = [likelihood([t=1]), likelihood([t=2]), likelihood([t=3]), ..., likelihood([t=T-1])]

    """
    # Terminating Condition in the Recursive Function
    if t == 1:
        initial_trajectory_state = observed_sequence[0]
        next_trajectory_state = observed_sequence[1]

        likelihood = marginal_probability_next_state[initial_trajectory_state][next_trajectory_state]
        likelihood_history = [likelihood]

        return likelihood, likelihood_history
    else:
        current_trajectory_state = observed_sequence[t - 1]
        next_trajectory_state = observed_sequence[t]

        likelihood_prev, likelihood_history_prev = get_likelihood_part_of_path_sequence(
            observed_sequence, marginal_probability_next_state, (t - 1))

        # Update (via multiplication of probabilities (sequences))
        likelihood = marginal_probability_next_state[current_trajectory_state][next_trajectory_state] * likelihood_prev
        likelihood_history_prev.append(likelihood)
        likelihood_history = likelihood_history_prev

        return likelihood, likelihood_history


def get_posterior_of_trajectory(observed_sequence, marginal_probability_next_state_tables):
    """
    Get Posterior of Trajectory Method

    This method calculates the Posterior of the Goals at every time step of the Observed Trajectory Sequence.

    Args:
        observed_sequence (datatype: array): This is an array that contains the sequential states of trajectory. This
        is specifically referred to as the trajectory sequence [S_{1:T}]. [T] is the termination state of the
        trajectory sequence.

        marginal_probability_next_state_tables (datatype: array): This is an array of nested dictionary of form
        state-nextstate combinations for a given goal (via it's policy table) and environment (map). This essentially
        the probability of the nextstate given the current state, goal, and environment and is calculated by
        marginalizing over the possible actions. P(s' | s, g, w) = sum[across a] (P(s' | s, a, w) * P_pi(a | s, g, w)).
        It is important to note that this array contains multiple marginal_probability_next_state dictionaries. This
        determines how many goals we consider and what the prior goal is initialized as (for our case it is uniform
        across how ever many goals we have (i.e., how many marginal_probabilities_next_state dictionaries are in the
        array).

            marginal_probability_next_state_tables = [marginal_probability_next_state[goal A],
                                                      marginal_probability_next_state[goal B],
                                                      marginal_probability_next_state[goal C],
                                                      ...]

                marginal_probability_next_state = {state: {nextstate: {probability}}} == P(s' | s, g, w)

    Returns:
        posteriors_history (datatype: dict): This is a nested dictionary of form time_step-goal combinations for a given
        observed trajectory sequence and environment. This contains the posteriors of each goal at each time step for
        the given observed trajectory sequence and environment. The posteriors_history has the following structure:

            posteriors_history = {time_step: [{goal: {goal posterior}}, {initial_state: {initial state posterior}}]}
                               == P(g | S_{1:T})

        num_possible_goals (datatype: integer): This is the number of possible Goal Points in the Environment.

        goal_strings (datatype: array): This is an array which contains the String naming convention of each Goal
        (i.e., for our case we use A, B, C as the goal names). Provided this example, the goal_strings has the
        following value:

            goal_string = ['A', 'B', 'C']

    """
    num_time_steps_trajectory = len(observed_sequence)
    num_possible_goals = len(marginal_probability_next_state_tables)

    goal_strings = []
    likelihood_history_list = []
    prior_goals = dict()

    # Set up the initial state prior. In this case, we know that the initial location in the trajectory is state: (0, 0)
    prior_initial_state = {'(0, 0)': 1}

    # Set up the goal naming convention (i.e., A, B, C, ...) and assign the initial priors of the goals (uniformly)
    for goal_idx in range(num_possible_goals):
        goal_strings.append(chr(ord('A') + goal_idx))

        # Uniform Prior Distribution of Goals. P(g)
        prior_goals[goal_strings[goal_idx]] = 1 / num_possible_goals

        # Get the Likelihood History of the Trajectory Sequence
        likelihood_final, likelihood_history = get_likelihood_entire_path_sequence(
                observed_sequence, marginal_probability_next_state_tables[goal_idx])

        likelihood_history_list.append(likelihood_history)

    likelihood_dict = dict()
    posteriors_history = dict()

    # For each time step
    for time_step in range(num_time_steps_trajectory - 1):
        # For each goal
        for goal_idx in range(num_possible_goals):
            likelihood_dict[(goal_strings[goal_idx], '(0, 0)')] = likelihood_history_list[goal_idx][time_step]

        # Get the Posterior Probability
        posteriors_history[time_step + 1] = get_posterior(prior_goals, prior_initial_state, likelihood_dict)

    print('Posteriors')
    print(posteriors_history)

    return posteriors_history, num_possible_goals, goal_strings


def plot_posterior_of_trajectory(posteriors_history, num_possible_goals, goal_strings,
                                 title_goal_string='Goal: Default', title_environment_string='Environment: Default'):
    """
    Plot Posterior of Trajectory Method

    This method plots the Posterior of the Goals at every time step of the Observed Trajectory Sequence.

    Args:
        posteriors_history (datatype: dict): This is a nested dictionary of form time_step-goal combinations for a given
        observed trajectory sequence and environment. This contains the posteriors of each goal at each time step for
        the given observed trajectory sequence and environment. The posteriors_history has the following structure:

            posteriors_history = {time_step: [{goal: {goal posterior}}, {initial_state: {initial state posterior}}]}
                               == P(g | S_{1:T})

        num_possible_goals (datatype: integer): This is the number of possible Goal Points in the Environment.

        goal_strings (datatype: array): This is an array which contains the String naming convention of each Goal
        (i.e., for our case we use A, B, C as the goal names). Provided this example, the goal_strings has the
        following value:

            goal_string = ['A', 'B', 'C']

        title_goal_string (datatype: string) [optional] (default value: 'Goal: Default'): This string allows the user
        to specify the Goal in the plot title.

        title_environment_string (datatype: string) [optional] (default value: 'Environment: Default'): This string
        allows the user to specify the Environment in the plot title.

    """
    time_step = list(posteriors_history)
    num_time_steps = len(time_step)

    posteriors_goals_matrix = np.zeros((num_possible_goals, num_time_steps))

    for t in range(num_time_steps):
        posterior_goal_dict = list(posteriors_history[time_step[t]][0])

        for goal_idx in range(num_possible_goals):
            posteriors_goals_matrix[goal_idx][t] = posteriors_history[time_step[t]][0][posterior_goal_dict[goal_idx]]

    for goal_idx in range(num_possible_goals):
        plt.plot(time_step, posteriors_goals_matrix[goal_idx][:], label='Posterior Probability of Goal: '
                                                                        + goal_strings[goal_idx])

    plt.title('Posterior of Goals Given the Observed Trajectory to ' + title_goal_string
              + ' in ' + title_environment_string)
    plt.xlabel('Time Step [ t ]')
    plt.ylabel('Posterior Probabilities of Goals [ P(Goal | Actions, Environment) ]')

    plt.legend()
    plt.show()


def main():
    # Transitions
    transition = {
        (0, 0): {
            (1, 0): {(1, 0): 1}, (0, 1): {(0, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(0, 0): 1}, 
            (-1, 1): {(0, 0): 1}, (1, -1): {(0, 0): 1}, (1, 1): {(1, 1): 1}, (-1, -1): {(0, 0): 1}}, 
        (0, 1): {
            (1, 0): {(1, 1): 1}, (0, 1): {(0, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(0, 0): 1}, 
            (-1, 1): {(0, 1): 1}, (1, -1): {(1, 0): 1}, (1, 1): {(1, 2): 1}, (-1, -1): {(0, 1): 1}}, 
        (0, 2): {
            (1, 0): {(1, 2): 1}, (0, 1): {(0, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(0, 1): 1}, 
            (-1, 1): {(0, 2): 1}, (1, -1): {(1, 1): 1}, (1, 1): {(1, 3): 1}, (-1, -1): {(0, 2): 1}}, 
        (0, 3): {
            (1, 0): {(1, 3): 1}, (0, 1): {(0, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(0, 2): 1}, 
            (-1, 1): {(0, 3): 1}, (1, -1): {(1, 2): 1}, (1, 1): {(1, 4): 1}, (-1, -1): {(0, 3): 1}}, 
        (0, 4): {
            (1, 0): {(1, 4): 1}, (0, 1): {(0, 5): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(0, 3): 1}, 
            (-1, 1): {(0, 4): 1}, (1, -1): {(1, 3): 1}, (1, 1): {(1, 5): 1}, (-1, -1): {(0, 4): 1}}, 
        (0, 5): {
            (1, 0): {(1, 5): 1}, (0, 1): {(0, 5): 1}, (-1, 0): {(0, 5): 1}, (0, -1): {(0, 4): 1}, 
            (-1, 1): {(0, 5): 1}, (1, -1): {(1, 4): 1}, (1, 1): {(0, 5): 1}, (-1, -1): {(0, 5): 1}}, 
        (1, 0): {
            (1, 0): {(2, 0): 1}, (0, 1): {(1, 1): 1}, (-1, 0): {(0, 0): 1}, (0, -1): {(1, 0): 1}, 
            (-1, 1): {(0, 1): 1}, (1, -1): {(1, 0): 1}, (1, 1): {(2, 1): 1}, (-1, -1): {(1, 0): 1}}, 
        (1, 1): {
            (1, 0): {(2, 1): 1}, (0, 1): {(1, 2): 1}, (-1, 0): {(0, 1): 1}, (0, -1): {(1, 0): 1}, 
            (-1, 1): {(0, 2): 1}, (1, -1): {(2, 0): 1}, (1, 1): {(2, 2): 1}, (-1, -1): {(0, 0): 1}}, 
        (1, 2): {
            (1, 0): {(2, 2): 1}, (0, 1): {(1, 3): 1}, (-1, 0): {(0, 2): 1}, (0, -1): {(1, 1): 1}, 
            (-1, 1): {(0, 3): 1}, (1, -1): {(2, 1): 1}, (1, 1): {(2, 3): 1}, (-1, -1): {(0, 1): 1}}, 
        (1, 3): {
            (1, 0): {(2, 3): 1}, (0, 1): {(1, 4): 1}, (-1, 0): {(0, 3): 1}, (0, -1): {(1, 2): 1}, 
            (-1, 1): {(0, 4): 1}, (1, -1): {(2, 2): 1}, (1, 1): {(2, 4): 1}, (-1, -1): {(0, 2): 1}}, 
        (1, 4): {
            (1, 0): {(2, 4): 1}, (0, 1): {(1, 5): 1}, (-1, 0): {(0, 4): 1}, (0, -1): {(1, 3): 1}, 
            (-1, 1): {(0, 5): 1}, (1, -1): {(2, 3): 1}, (1, 1): {(2, 5): 1}, (-1, -1): {(0, 3): 1}}, 
        (1, 5): {
            (1, 0): {(2, 5): 1}, (0, 1): {(1, 5): 1}, (-1, 0): {(0, 5): 1}, (0, -1): {(1, 4): 1}, 
            (-1, 1): {(1, 5): 1}, (1, -1): {(2, 4): 1}, (1, 1): {(1, 5): 1}, (-1, -1): {(0, 4): 1}}, 
        (2, 0): {
            (1, 0): {(3, 0): 1}, (0, 1): {(2, 1): 1}, (-1, 0): {(1, 0): 1}, (0, -1): {(2, 0): 1}, 
            (-1, 1): {(1, 1): 1}, (1, -1): {(2, 0): 1}, (1, 1): {(3, 1): 1}, (-1, -1): {(2, 0): 1}}, 
        (2, 1): {
            (1, 0): {(3, 1): 1}, (0, 1): {(2, 2): 1}, (-1, 0): {(1, 1): 1}, (0, -1): {(2, 0): 1}, 
            (-1, 1): {(1, 2): 1}, (1, -1): {(3, 0): 1}, (1, 1): {(3, 2): 1}, (-1, -1): {(1, 0): 1}}, 
        (2, 2): {
            (1, 0): {(3, 2): 1}, (0, 1): {(2, 3): 1}, (-1, 0): {(1, 2): 1}, (0, -1): {(2, 1): 1}, 
            (-1, 1): {(1, 3): 1}, (1, -1): {(3, 1): 1}, (1, 1): {(3, 3): 1}, (-1, -1): {(1, 1): 1}}, 
        (2, 3): {
            (1, 0): {(3, 3): 1}, (0, 1): {(2, 4): 1}, (-1, 0): {(1, 3): 1}, (0, -1): {(2, 2): 1}, 
            (-1, 1): {(1, 4): 1}, (1, -1): {(3, 2): 1}, (1, 1): {(3, 4): 1}, (-1, -1): {(1, 2): 1}}, 
        (2, 4): {
            (1, 0): {(3, 4): 1}, (0, 1): {(2, 5): 1}, (-1, 0): {(1, 4): 1}, (0, -1): {(2, 3): 1}, 
            (-1, 1): {(1, 5): 1}, (1, -1): {(3, 3): 1}, (1, 1): {(3, 5): 1}, (-1, -1): {(1, 3): 1}}, 
        (2, 5): {
            (1, 0): {(3, 5): 1}, (0, 1): {(2, 5): 1}, (-1, 0): {(1, 5): 1}, (0, -1): {(2, 4): 1}, 
            (-1, 1): {(2, 5): 1}, (1, -1): {(3, 4): 1}, (1, 1): {(2, 5): 1}, (-1, -1): {(1, 4): 1}}, 
        (3, 0): {
            (1, 0): {(4, 0): 1}, (0, 1): {(3, 1): 1}, (-1, 0): {(2, 0): 1}, (0, -1): {(3, 0): 1}, 
            (-1, 1): {(2, 1): 1}, (1, -1): {(3, 0): 1}, (1, 1): {(4, 1): 1}, (-1, -1): {(3, 0): 1}}, 
        (3, 1): {
            (1, 0): {(4, 1): 1}, (0, 1): {(3, 2): 1}, (-1, 0): {(2, 1): 1}, (0, -1): {(3, 0): 1}, 
            (-1, 1): {(2, 2): 1}, (1, -1): {(4, 0): 1}, (1, 1): {(4, 2): 1}, (-1, -1): {(2, 0): 1}}, 
        (3, 2): {
            (1, 0): {(4, 2): 1}, (0, 1): {(3, 3): 1}, (-1, 0): {(2, 2): 1}, (0, -1): {(3, 1): 1}, 
            (-1, 1): {(2, 3): 1}, (1, -1): {(4, 1): 1}, (1, 1): {(4, 3): 1}, (-1, -1): {(2, 1): 1}}, 
        (3, 3): {
            (1, 0): {(4, 3): 1}, (0, 1): {(3, 4): 1}, (-1, 0): {(2, 3): 1}, (0, -1): {(3, 2): 1}, 
            (-1, 1): {(2, 4): 1}, (1, -1): {(4, 2): 1}, (1, 1): {(4, 4): 1}, (-1, -1): {(2, 2): 1}}, 
        (3, 4): {
            (1, 0): {(4, 4): 1}, (0, 1): {(3, 5): 1}, (-1, 0): {(2, 4): 1}, (0, -1): {(3, 3): 1}, 
            (-1, 1): {(2, 5): 1}, (1, -1): {(4, 3): 1}, (1, 1): {(4, 5): 1}, (-1, -1): {(2, 3): 1}}, 
        (3, 5): {
            (1, 0): {(4, 5): 1}, (0, 1): {(3, 5): 1}, (-1, 0): {(2, 5): 1}, (0, -1): {(3, 4): 1}, 
            (-1, 1): {(3, 5): 1}, (1, -1): {(4, 4): 1}, (1, 1): {(3, 5): 1}, (-1, -1): {(2, 4): 1}}, 
        (4, 0): {
            (1, 0): {(5, 0): 1}, (0, 1): {(4, 1): 1}, (-1, 0): {(3, 0): 1}, (0, -1): {(4, 0): 1}, 
            (-1, 1): {(3, 1): 1}, (1, -1): {(4, 0): 1}, (1, 1): {(5, 1): 1}, (-1, -1): {(4, 0): 1}}, 
        (4, 1): {
            (1, 0): {(5, 1): 1}, (0, 1): {(4, 2): 1}, (-1, 0): {(3, 1): 1}, (0, -1): {(4, 0): 1}, 
            (-1, 1): {(3, 2): 1}, (1, -1): {(5, 0): 1}, (1, 1): {(5, 2): 1}, (-1, -1): {(3, 0): 1}}, 
        (4, 2): {
            (1, 0): {(5, 2): 1}, (0, 1): {(4, 3): 1}, (-1, 0): {(3, 2): 1}, (0, -1): {(4, 1): 1}, 
            (-1, 1): {(3, 3): 1}, (1, -1): {(5, 1): 1}, (1, 1): {(5, 3): 1}, (-1, -1): {(3, 1): 1}}, 
        (4, 3): {
            (1, 0): {(5, 3): 1}, (0, 1): {(4, 4): 1}, (-1, 0): {(3, 3): 1}, (0, -1): {(4, 2): 1}, 
            (-1, 1): {(3, 4): 1}, (1, -1): {(5, 2): 1}, (1, 1): {(5, 4): 1}, (-1, -1): {(3, 2): 1}}, 
        (4, 4): {
            (1, 0): {(5, 4): 1}, (0, 1): {(4, 5): 1}, (-1, 0): {(3, 4): 1}, (0, -1): {(4, 3): 1}, 
            (-1, 1): {(3, 5): 1}, (1, -1): {(5, 3): 1}, (1, 1): {(5, 5): 1}, (-1, -1): {(3, 3): 1}}, 
        (4, 5): {
            (1, 0): {(5, 5): 1}, (0, 1): {(4, 5): 1}, (-1, 0): {(3, 5): 1}, (0, -1): {(4, 4): 1}, 
            (-1, 1): {(4, 5): 1}, (1, -1): {(5, 4): 1}, (1, 1): {(4, 5): 1}, (-1, -1): {(3, 4): 1}}, 
        (5, 0): {
            (1, 0): {(6, 0): 1}, (0, 1): {(5, 1): 1}, (-1, 0): {(4, 0): 1}, (0, -1): {(5, 0): 1}, 
            (-1, 1): {(4, 1): 1}, (1, -1): {(5, 0): 1}, (1, 1): {(6, 1): 1}, (-1, -1): {(5, 0): 1}}, 
        (5, 1): {
            (1, 0): {(6, 1): 1}, (0, 1): {(5, 2): 1}, (-1, 0): {(4, 1): 1}, (0, -1): {(5, 0): 1}, 
            (-1, 1): {(4, 2): 1}, (1, -1): {(6, 0): 1}, (1, 1): {(6, 2): 1}, (-1, -1): {(4, 0): 1}}, 
        (5, 2): {
            (1, 0): {(6, 2): 1}, (0, 1): {(5, 3): 1}, (-1, 0): {(4, 2): 1}, (0, -1): {(5, 1): 1}, 
            (-1, 1): {(4, 3): 1}, (1, -1): {(6, 1): 1}, (1, 1): {(6, 3): 1}, (-1, -1): {(4, 1): 1}}, 
        (5, 3): {
            (1, 0): {(6, 3): 1}, (0, 1): {(5, 4): 1}, (-1, 0): {(4, 3): 1}, (0, -1): {(5, 2): 1}, 
            (-1, 1): {(4, 4): 1}, (1, -1): {(6, 2): 1}, (1, 1): {(6, 4): 1}, (-1, -1): {(4, 2): 1}}, 
        (5, 4): {
            (1, 0): {(6, 4): 1}, (0, 1): {(5, 5): 1}, (-1, 0): {(4, 4): 1}, (0, -1): {(5, 3): 1}, 
            (-1, 1): {(4, 5): 1}, (1, -1): {(6, 3): 1}, (1, 1): {(6, 5): 1}, (-1, -1): {(4, 3): 1}}, 
        (5, 5): {
            (1, 0): {(6, 5): 1}, (0, 1): {(5, 5): 1}, (-1, 0): {(4, 5): 1}, (0, -1): {(5, 4): 1}, 
            (-1, 1): {(5, 5): 1}, (1, -1): {(6, 4): 1}, (1, 1): {(5, 5): 1}, (-1, -1): {(4, 4): 1}}, 
        (6, 0): {
            (1, 0): {(6, 0): 1}, (0, 1): {(6, 1): 1}, (-1, 0): {(5, 0): 1}, (0, -1): {(6, 0): 1}, 
            (-1, 1): {(5, 1): 1}, (1, -1): {(6, 0): 1}, (1, 1): {(6, 0): 1}, (-1, -1): {(6, 0): 1}}, 
        (6, 1): {
            (1, 0): {(6, 1): 1}, (0, 1): {(6, 2): 1}, (-1, 0): {(5, 1): 1}, (0, -1): {(6, 0): 1}, 
            (-1, 1): {(5, 2): 1}, (1, -1): {(6, 1): 1}, (1, 1): {(6, 1): 1}, (-1, -1): {(5, 0): 1}}, 
        (6, 2): {
            (1, 0): {(6, 2): 1}, (0, 1): {(6, 3): 1}, (-1, 0): {(5, 2): 1}, (0, -1): {(6, 1): 1}, 
            (-1, 1): {(5, 3): 1}, (1, -1): {(6, 2): 1}, (1, 1): {(6, 2): 1}, (-1, -1): {(5, 1): 1}}, 
        (6, 3): {
            (1, 0): {(6, 3): 1}, (0, 1): {(6, 4): 1}, (-1, 0): {(5, 3): 1}, (0, -1): {(6, 2): 1}, 
            (-1, 1): {(5, 4): 1}, (1, -1): {(6, 3): 1}, (1, 1): {(6, 3): 1}, (-1, -1): {(5, 2): 1}}, 
        (6, 4): {
            (1, 0): {(6, 4): 1}, (0, 1): {(6, 5): 1}, (-1, 0): {(5, 4): 1}, (0, -1): {(6, 3): 1}, 
            (-1, 1): {(5, 5): 1}, (1, -1): {(6, 4): 1}, (1, 1): {(6, 4): 1}, (-1, -1): {(5, 3): 1}}, 
        (6, 5): {
            (1, 0): {(6, 5): 1}, (0, 1): {(6, 5): 1}, (-1, 0): {(5, 5): 1}, (0, -1): {(6, 4): 1}, 
            (-1, 1): {(6, 5): 1}, (1, -1): {(6, 5): 1}, (1, 1): {(6, 5): 1}, (-1, -1): {(5, 4): 1}}}

    # Observed Trajectories
    trajectory_to_goal_a = [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4), (4, 4), (5, 4), (6, 4)]
    trajectory_to_goal_b = [(0, 0), (1, 1), (2, 2), (2, 3), (3, 4), (4, 3), (5, 2), (6, 1)]
    trajectory_to_goal_c = [(0, 0), (0, 1), (1, 2), (1, 3), (1, 4), (1, 5)]

    common_rewards = {
        (0, 0): {
            (1, 0): {(1, 0): -1.0}, (0, 1): {(0, 1): -1.0},
            (-1, 0): {(0, 0): -1}, (0, -1): {(0, 0): -1},
            (-1, 1): {(0, 0): -1}, (1, -1): {(0, 0): -1},
            (1, 1): {(1, 1): -1.4142135623730951}, (-1, -1): {(0, 0): -1}},
        (0, 1): {
            (1, 0): {(1, 1): -1.0}, (0, 1): {(0, 2): -1.0},
            (-1, 0): {(0, 1): -1}, (0, -1): {(0, 0): -1.0},
            (-1, 1): {(0, 1): -1}, (1, -1): {(1, 0): -1.4142135623730951},
            (1, 1): {(1, 2): -1.4142135623730951}, (-1, -1): {(0, 1): -1}},
        (0, 2): {
            (1, 0): {(1, 2): -1.0}, (0, 1): {(0, 3): -1.0},
            (-1, 0): {(0, 2): -1}, (0, -1): {(0, 1): -1.0},
            (-1, 1): {(0, 2): -1}, (1, -1): {(1, 1): -1.4142135623730951},
            (1, 1): {(1, 3): -1.4142135623730951}, (-1, -1): {(0, 2): -1}},
        (0, 3): {
            (1, 0): {(1, 3): -1.0}, (0, 1): {(0, 4): -1.0},
            (-1, 0): {(0, 3): -1}, (0, -1): {(0, 2): -1.0},
            (-1, 1): {(0, 3): -1}, (1, -1): {(1, 2): -1.4142135623730951},
            (1, 1): {(1, 4): -1.4142135623730951}, (-1, -1): {(0, 3): -1}},
        (0, 4): {
            (1, 0): {(1, 4): -1.0}, (0, 1): {(0, 5): -1.0},
            (-1, 0): {(0, 4): -1}, (0, -1): {(0, 3): -1.0},
            (-1, 1): {(0, 4): -1}, (1, -1): {(1, 3): -1.4142135623730951},
            (1, 1): {(1, 5): -1.4142135623730951}, (-1, -1): {(0, 4): -1}},
        (0, 5): {
            (1, 0): {(1, 5): -1.0}, (0, 1): {(0, 5): -1},
            (-1, 0): {(0, 5): -1}, (0, -1): {(0, 4): -1.0},
            (-1, 1): {(0, 5): -1}, (1, -1): {(1, 4): -1.4142135623730951},
            (1, 1): {(0, 5): -1}, (-1, -1): {(0, 5): -1}},
        (1, 0): {
            (1, 0): {(2, 0): -1.0}, (0, 1): {(1, 1): -1.0},
            (-1, 0): {(0, 0): -1.0}, (0, -1): {(1, 0): -1},
            (-1, 1): {(0, 1): -1.4142135623730951}, (1, -1): {(1, 0): -1},
            (1, 1): {(2, 1): -1.4142135623730951}, (-1, -1): {(1, 0): -1}},
        (1, 1): {
            (1, 0): {(2, 1): -1.0}, (0, 1): {(1, 2): -1.0},
            (-1, 0): {(0, 1): -1.0}, (0, -1): {(1, 0): -1.0},
            (-1, 1): {(0, 2): -1.4142135623730951}, (1, -1): {(2, 0): -1.4142135623730951},
            (1, 1): {(2, 2): -1.4142135623730951}, (-1, -1): {(0, 0): -1.4142135623730951}},
        (1, 2): {
            (1, 0): {(2, 2): -1.0}, (0, 1): {(1, 3): -1.0},
            (-1, 0): {(0, 2): -1.0}, (0, -1): {(1, 1): -1.0},
            (-1, 1): {(0, 3): -1.4142135623730951}, (1, -1): {(2, 1): -1.4142135623730951},
            (1, 1): {(2, 3): -1.4142135623730951}, (-1, -1): {(0, 1): -1.4142135623730951}},
        (1, 3): {
            (1, 0): {(2, 3): -1.0}, (0, 1): {(1, 4): -1.0},
            (-1, 0): {(0, 3): -1.0}, (0, -1): {(1, 2): -1.0},
            (-1, 1): {(0, 4): -1.4142135623730951}, (1, -1): {(2, 2): -1.4142135623730951},
            (1, 1): {(2, 4): -1.4142135623730951}, (-1, -1): {(0, 2): -1.4142135623730951}},
        (1, 4): {
            (1, 0): {(2, 4): -1.0}, (0, 1): {(1, 5): -1.0},
            (-1, 0): {(0, 4): -1.0}, (0, -1): {(1, 3): -1.0},
            (-1, 1): {(0, 5): -1.4142135623730951}, (1, -1): {(2, 3): -1.4142135623730951},
            (1, 1): {(2, 5): -1.4142135623730951}, (-1, -1): {(0, 3): -1.4142135623730951}},
        (2, 0): {
            (1, 0): {(3, 0): -1.0}, (0, 1): {(2, 1): -1.0},
            (-1, 0): {(1, 0): -1.0}, (0, -1): {(2, 0): -1},
            (-1, 1): {(1, 1): -1.4142135623730951}, (1, -1): {(2, 0): -1},
            (1, 1): {(3, 1): -1.4142135623730951}, (-1, -1): {(2, 0): -1}},
        (2, 1): {
            (1, 0): {(3, 1): -1.0}, (0, 1): {(2, 2): -1.0},
            (-1, 0): {(1, 1): -1.0}, (0, -1): {(2, 0): -1.0},
            (-1, 1): {(1, 2): -1.4142135623730951}, (1, -1): {(3, 0): -1.4142135623730951},
            (1, 1): {(3, 2): -1.4142135623730951}, (-1, -1): {(1, 0): -1.4142135623730951}},
        (2, 2): {
            (1, 0): {(3, 2): -1.0}, (0, 1): {(2, 3): -1.0},
            (-1, 0): {(1, 2): -1.0}, (0, -1): {(2, 1): -1.0},
            (-1, 1): {(1, 3): -1.4142135623730951}, (1, -1): {(3, 1): -1.4142135623730951},
            (1, 1): {(3, 3): -1.4142135623730951}, (-1, -1): {(1, 1): -1.4142135623730951}},
        (2, 3): {
            (1, 0): {(3, 3): -1.0}, (0, 1): {(2, 4): -1.0},
            (-1, 0): {(1, 3): -1.0}, (0, -1): {(2, 2): -1.0},
            (-1, 1): {(1, 4): -1.4142135623730951}, (1, -1): {(3, 2): -1.4142135623730951},
            (1, 1): {(3, 4): -1.4142135623730951}, (-1, -1): {(1, 2): -1.4142135623730951}},
        (2, 4): {
            (1, 0): {(3, 4): -1.0}, (0, 1): {(2, 5): -1.0},
            (-1, 0): {(1, 4): -1.0}, (0, -1): {(2, 3): -1.0},
            (-1, 1): {(1, 5): -1.4142135623730951}, (1, -1): {(3, 3): -1.4142135623730951},
            (1, 1): {(3, 5): -1.4142135623730951}, (-1, -1): {(1, 3): -1.4142135623730951}},
        (2, 5): {
            (1, 0): {(3, 5): -1.0}, (0, 1): {(2, 5): -1},
            (-1, 0): {(1, 5): -1.0}, (0, -1): {(2, 4): -1.0},
            (-1, 1): {(2, 5): -1}, (1, -1): {(3, 4): -1.4142135623730951},
            (1, 1): {(2, 5): -1}, (-1, -1): {(1, 4): -1.4142135623730951}},
        (3, 0): {
            (1, 0): {(4, 0): -100}, (0, 1): {(3, 1): -100},
            (-1, 0): {(2, 0): -100}, (0, -1): {(3, 0): -100},
            (-1, 1): {(2, 1): -100}, (1, -1): {(3, 0): -100},
            (1, 1): {(4, 1): -100}, (-1, -1): {(3, 0): -100}},
        (3, 2): {
            (1, 0): {(4, 2): -100}, (0, 1): {(3, 3): -100},
            (-1, 0): {(2, 2): -100}, (0, -1): {(3, 1): -100},
            (-1, 1): {(2, 3): -100}, (1, -1): {(4, 1): -100},
            (1, 1): {(4, 3): -100}, (-1, -1): {(2, 1): -100}},
        (3, 3): {
            (1, 0): {(4, 3): -100}, (0, 1): {(3, 4): -100},
            (-1, 0): {(2, 3): -100}, (0, -1): {(3, 2): -100},
            (-1, 1): {(2, 4): -100}, (1, -1): {(4, 2): -100},
            (1, 1): {(4, 4): -100}, (-1, -1): {(2, 2): -100}},
        (3, 4): {
            (1, 0): {(4, 4): -1.0}, (0, 1): {(3, 5): -1.0},
            (-1, 0): {(2, 4): -1.0}, (0, -1): {(3, 3): -1.0},
            (-1, 1): {(2, 5): -1.4142135623730951}, (1, -1): {(4, 3): -1.4142135623730951},
            (1, 1): {(4, 5): -1.4142135623730951}, (-1, -1): {(2, 3): -1.4142135623730951}},
        (3, 5): {
            (1, 0): {(4, 5): -1.0}, (0, 1): {(3, 5): -1},
            (-1, 0): {(2, 5): -1.0}, (0, -1): {(3, 4): -1.0},
            (-1, 1): {(3, 5): -1}, (1, -1): {(4, 4): -1.4142135623730951},
            (1, 1): {(3, 5): -1}, (-1, -1): {(2, 4): -1.4142135623730951}},
        (4, 0): {
            (1, 0): {(5, 0): -1.0}, (0, 1): {(4, 1): -1.0},
            (-1, 0): {(3, 0): -1.0}, (0, -1): {(4, 0): -1},
            (-1, 1): {(3, 1): -1.4142135623730951}, (1, -1): {(4, 0): -1},
            (1, 1): {(5, 1): -1.4142135623730951}, (-1, -1): {(4, 0): -1}},
        (4, 1): {
            (1, 0): {(5, 1): -1.0}, (0, 1): {(4, 2): -1.0},
            (-1, 0): {(3, 1): -1.0}, (0, -1): {(4, 0): -1.0},
            (-1, 1): {(3, 2): -1.4142135623730951}, (1, -1): {(5, 0): -1.4142135623730951},
            (1, 1): {(5, 2): -1.4142135623730951}, (-1, -1): {(3, 0): -1.4142135623730951}},
        (4, 2): {
            (1, 0): {(5, 2): -1.0}, (0, 1): {(4, 3): -1.0},
            (-1, 0): {(3, 2): -1.0}, (0, -1): {(4, 1): -1.0},
            (-1, 1): {(3, 3): -1.4142135623730951}, (1, -1): {(5, 1): -1.4142135623730951},
            (1, 1): {(5, 3): -1.4142135623730951}, (-1, -1): {(3, 1): -1.4142135623730951}},
        (4, 3): {
            (1, 0): {(5, 3): -1.0}, (0, 1): {(4, 4): -1.0},
            (-1, 0): {(3, 3): -1.0}, (0, -1): {(4, 2): -1.0},
            (-1, 1): {(3, 4): -1.4142135623730951}, (1, -1): {(5, 2): -1.4142135623730951},
            (1, 1): {(5, 4): -1.4142135623730951}, (-1, -1): {(3, 2): -1.4142135623730951}},
        (4, 4): {
            (1, 0): {(5, 4): -1.0}, (0, 1): {(4, 5): -1.0},
            (-1, 0): {(3, 4): -1.0}, (0, -1): {(4, 3): -1.0},
            (-1, 1): {(3, 5): -1.4142135623730951}, (1, -1): {(5, 3): -1.4142135623730951},
            (1, 1): {(5, 5): -1.4142135623730951}, (-1, -1): {(3, 3): -1.4142135623730951}},
        (4, 5): {
            (1, 0): {(5, 5): -1.0}, (0, 1): {(4, 5): -1},
            (-1, 0): {(3, 5): -1.0}, (0, -1): {(4, 4): -1.0},
            (-1, 1): {(4, 5): -1}, (1, -1): {(5, 4): -1.4142135623730951},
            (1, 1): {(4, 5): -1}, (-1, -1): {(3, 4): -1.4142135623730951}},
        (5, 0): {
            (1, 0): {(6, 0): -1.0}, (0, 1): {(5, 1): -1.0},
            (-1, 0): {(4, 0): -1.0}, (0, -1): {(5, 0): -1},
            (-1, 1): {(4, 1): -1.4142135623730951}, (1, -1): {(5, 0): -1},
            (1, 1): {(6, 1): -1.4142135623730951}, (-1, -1): {(5, 0): -1}},
        (5, 1): {
            (1, 0): {(6, 1): -1.0}, (0, 1): {(5, 2): -1.0},
            (-1, 0): {(4, 1): -1.0}, (0, -1): {(5, 0): -1.0},
            (-1, 1): {(4, 2): -1.4142135623730951}, (1, -1): {(6, 0): -1.4142135623730951},
            (1, 1): {(6, 2): -1.4142135623730951}, (-1, -1): {(4, 0): -1.4142135623730951}},
        (5, 2): {
            (1, 0): {(6, 2): -1.0}, (0, 1): {(5, 3): -1.0},
            (-1, 0): {(4, 2): -1.0}, (0, -1): {(5, 1): -1.0},
            (-1, 1): {(4, 3): -1.4142135623730951}, (1, -1): {(6, 1): -1.4142135623730951},
            (1, 1): {(6, 3): -1.4142135623730951}, (-1, -1): {(4, 1): -1.4142135623730951}},
        (5, 3): {
            (1, 0): {(6, 3): -1.0}, (0, 1): {(5, 4): -1.0},
            (-1, 0): {(4, 3): -1.0}, (0, -1): {(5, 2): -1.0},
            (-1, 1): {(4, 4): -1.4142135623730951}, (1, -1): {(6, 2): -1.4142135623730951},
            (1, 1): {(6, 4): -1.4142135623730951}, (-1, -1): {(4, 2): -1.4142135623730951}},
        (5, 4): {
            (1, 0): {(6, 4): -1.0}, (0, 1): {(5, 5): -1.0},
            (-1, 0): {(4, 4): -1.0}, (0, -1): {(5, 3): -1.0},
            (-1, 1): {(4, 5): -1.4142135623730951}, (1, -1): {(6, 3): -1.4142135623730951},
            (1, 1): {(6, 5): -1.4142135623730951}, (-1, -1): {(4, 3): -1.4142135623730951}},
        (5, 5): {
            (1, 0): {(6, 5): -1.0}, (0, 1): {(5, 5): -1},
            (-1, 0): {(4, 5): -1.0}, (0, -1): {(5, 4): -1.0},
            (-1, 1): {(5, 5): -1}, (1, -1): {(6, 4): -1.4142135623730951},
            (1, 1): {(5, 5): -1}, (-1, -1): {(4, 4): -1.4142135623730951}},
        (6, 0): {
            (1, 0): {(6, 0): -1}, (0, 1): {(6, 1): -1.0},
            (-1, 0): {(5, 0): -1.0}, (0, -1): {(6, 0): -1},
            (-1, 1): {(5, 1): -1.4142135623730951}, (1, -1): {(6, 0): -1},
            (1, 1): {(6, 0): -1}, (-1, -1): {(6, 0): -1}},
        (6, 2): {
            (1, 0): {(6, 2): -1}, (0, 1): {(6, 3): -1.0},
            (-1, 0): {(5, 2): -1.0}, (0, -1): {(6, 1): -1.0},
            (-1, 1): {(5, 3): -1.4142135623730951}, (1, -1): {(6, 2): -1},
            (1, 1): {(6, 2): -1}, (-1, -1): {(5, 1): -1.4142135623730951}},
        (6, 3): {
            (1, 0): {(6, 3): -1}, (0, 1): {(6, 4): -1.0},
            (-1, 0): {(5, 3): -1.0}, (0, -1): {(6, 2): -1.0},
            (-1, 1): {(5, 4): -1.4142135623730951}, (1, -1): {(6, 3): -1},
            (1, 1): {(6, 3): -1}, (-1, -1): {(5, 2): -1.4142135623730951}},
        (6, 5): {
            (1, 0): {(6, 5): -1}, (0, 1): {(6, 5): -1},
            (-1, 0): {(5, 5): -1.0}, (0, -1): {(6, 4): -1.0},
            (-1, 1): {(6, 5): -1}, (1, -1): {(6, 5): -1},
            (1, 1): {(6, 5): -1}, (-1, -1): {(5, 4): -1.4142135623730951}}}

    reward_3_1_env_1 = {
        (3, 1): {
            (1, 0): {(4, 1): -100}, (0, 1): {(3, 2): -100},
            (-1, 0): {(2, 1): -100}, (0, -1): {(3, 0): -100},
            (-1, 1): {(2, 2): -100}, (1, -1): {(4, 0): -100},
            (1, 1): {(4, 2): -100}, (-1, -1): {(2, 0): -100}}}

    reward_3_1_env_2 = {
        (3, 1): {
            (1, 0): {(4, 1): -1.0}, (0, 1): {(3, 2): -1.0},
            (-1, 0): {(2, 1): -1.0}, (0, -1): {(3, 0): -1.0},
            (-1, 1): {(2, 2): -1.4142135623730951}, (1, -1): {(4, 0): -1.4142135623730951},
            (1, 1): {(4, 2): -1.4142135623730951}, (-1, -1): {(2, 0): -1.4142135623730951}}}

    reward_1_5_goal_c = {
        (1, 5): {
            (1, 0): {(2, 5): 10}, (0, 1): {(1, 5): -1},
            (-1, 0): {(0, 5): 10}, (0, -1): {(1, 4): 10},
            (-1, 1): {(1, 5): -1}, (1, -1): {(2, 4): 10},
            (1, 1): {(1, 5): -1}, (-1, -1): {(0, 4): 10}}}

    reward_1_5_goal_a_b = {
        (1, 5): {
            (1, 0): {(2, 5): -1.0}, (0, 1): {(1, 5): -1},
            (-1, 0): {(0, 5): -1.0}, (0, -1): {(1, 4): -1.0},
            (-1, 1): {(1, 5): -1}, (1, -1): {(2, 4): -1.4142135623730951},
            (1, 1): {(1, 5): -1}, (-1, -1): {(0, 4): -1.4142135623730951}}}

    reward_6_1_goal_b = {
        (6, 1): {
            (1, 0): {(6, 1): -1}, (0, 1): {(6, 2): 10},
            (-1, 0): {(5, 1): 10}, (0, -1): {(6, 0): 10},
            (-1, 1): {(5, 2): 10}, (1, -1): {(6, 1): -1},
            (1, 1): {(6, 1): -1}, (-1, -1): {(5, 0): 10}}}

    reward_6_1_goal_a_c = {
        (6, 1): {
            (1, 0): {(6, 1): -1}, (0, 1): {(6, 2): -1.0},
            (-1, 0): {(5, 1): -1.0}, (0, -1): {(6, 0): -1.0},
            (-1, 1): {(5, 2): -1.4142135623730951}, (1, -1): {(6, 1): -1},
            (1, 1): {(6, 1): -1}, (-1, -1): {(5, 0): -1.4142135623730951}}}

    reward_6_4_goal_a = {
        (6, 4): {
            (1, 0): {(6, 4): -1}, (0, 1): {(6, 5): 10},
            (-1, 0): {(5, 4): 10}, (0, -1): {(6, 3): 10},
            (-1, 1): {(5, 5): 10}, (1, -1): {(6, 4): -1},
            (1, 1): {(6, 4): -1}, (-1, -1): {(5, 3): 10}}}

    reward_6_4_goal_b_c = {
        (6, 4): {
            (1, 0): {(6, 4): -1}, (0, 1): {(6, 5): -1.0},
            (-1, 0): {(5, 4): -1.0}, (0, -1): {(6, 3): -1.0},
            (-1, 1): {(5, 5): -1.4142135623730951}, (1, -1): {(6, 4): -1},
            (1, 1): {(6, 4): -1}, (-1, -1): {(5, 3): -1.4142135623730951}}}

    common_rewards_env_1 = dict()
    common_rewards_env_2 = dict()

    common_rewards_env_1.update(common_rewards)
    common_rewards_env_1.update(reward_3_1_env_1)

    common_rewards_env_2.update(common_rewards)
    common_rewards_env_2.update(reward_3_1_env_2)

    # Environment 1: Solid  Barrier
    reward_a = dict()
    reward_b = dict()
    reward_c = dict()

    reward_a.update(common_rewards_env_1)
    reward_a.update(reward_1_5_goal_a_b)
    reward_a.update(reward_6_1_goal_a_c)
    reward_a.update(reward_6_4_goal_a)

    reward_b.update(common_rewards_env_1)
    reward_b.update(reward_1_5_goal_a_b)
    reward_b.update(reward_6_1_goal_b)
    reward_b.update(reward_6_4_goal_b_c)

    reward_c.update(common_rewards_env_1)
    reward_c.update(reward_1_5_goal_c)
    reward_c.update(reward_6_1_goal_a_c)
    reward_c.update(reward_6_4_goal_b_c)

    # Environment 2: Barrier with a Gap
    reward_a_gap = dict()
    reward_b_gap = dict()
    reward_c_gap = dict()

    reward_a_gap.update(common_rewards_env_2)
    reward_a_gap.update(reward_1_5_goal_a_b)
    reward_a_gap.update(reward_6_1_goal_a_c)
    reward_a_gap.update(reward_6_4_goal_a)

    reward_b_gap.update(common_rewards_env_2)
    reward_b_gap.update(reward_1_5_goal_a_b)
    reward_b_gap.update(reward_6_1_goal_b)
    reward_b_gap.update(reward_6_4_goal_b_c)

    reward_c_gap.update(common_rewards_env_2)
    reward_c_gap.update(reward_1_5_goal_c)
    reward_c_gap.update(reward_6_1_goal_a_c)
    reward_c_gap.update(reward_6_4_goal_b_c)

    #######################################
    # Run value iteration to get boltzmann policies
    # Can assume that at time 0 the prior over the goals is uniform
    # Plot posteriors for each of the trajectories for each environment (3 trajectories x 2 environments)
    ######################################
    goal_states = [(6, 1, "B"), (1, 5, "C"), (6, 4, "A")]
    grid_width = 7
    grid_height = 6
    trap_states = [(3, 0), (3, 1), (3, 2), (3, 3)]

    trap_states_gap = [(3, 0), (3, 2), (3, 3)]

    gamma = .95
    beta = .4
    convergence_tolerance = 10e-7

    # Environment #1
    # Goal A
    perform_value_iteration_goal_a_env1 = ValueIteration(transition, reward_a, value_table_initial,
                                                         convergence_tolerance, gamma, use_softmax=True,
                                                         use_noise=True, noise_beta=beta)
    optimal_value_table_a_env1, optimal_policy_table_a_env1 = perform_value_iteration_goal_a_env1()

    print(optimal_policy_table_a_env1)

    visualize_value_table(grid_width, grid_height, goal_states[2], trap_states, optimal_value_table_a_env1)
    visualize_policy(grid_width, grid_height, goal_states[2], trap_states, optimal_policy_table_a_env1)

    # Goal B
    perform_value_iteration_goal_b_env1 = ValueIteration(transition, reward_b, value_table_initial,
                                                         convergence_tolerance, gamma, use_softmax=True,
                                                         use_noise=True, noise_beta=beta)
    optimal_value_table_b_env1, optimal_policy_table_b_env1 = perform_value_iteration_goal_b_env1()

    visualize_value_table(grid_width, grid_height, goal_states[0], trap_states, optimal_value_table_b_env1)
    visualize_policy(grid_width, grid_height, goal_states[0], trap_states, optimal_policy_table_b_env1)

    # Goal C
    perform_value_iteration_goal_c_env1 = ValueIteration(transition, reward_c, value_table_initial,
                                                         convergence_tolerance, gamma, use_softmax=True,
                                                         use_noise=True, noise_beta=beta)
    optimal_value_table_c_env1, optimal_policy_table_c_env1 = perform_value_iteration_goal_c_env1()

    visualize_value_table(grid_width, grid_height, goal_states[1], trap_states, optimal_value_table_c_env1)
    visualize_policy(grid_width, grid_height, goal_states[1], trap_states, optimal_policy_table_c_env1)

    # Calculating the Marginal Probability Tables for Each Goal in Environment #1
    marginal_probability_next_state_a_env1 = get_probability_of_individual_state_transitions(
        transition, optimal_policy_table_a_env1)

    marginal_probability_next_state_b_env1 = get_probability_of_individual_state_transitions(
        transition, optimal_policy_table_b_env1)

    marginal_probability_next_state_c_env1 = get_probability_of_individual_state_transitions(
        transition, optimal_policy_table_c_env1)

    # Marginal Probabilities of the Next State Tables (Combined into a single array)
    marginal_probability_next_state_table_env1 = [marginal_probability_next_state_a_env1,
                                                  marginal_probability_next_state_b_env1,
                                                  marginal_probability_next_state_c_env1]

    # Calculating Posterior Probabilities
    print('Environment #1')
    print('\nTrajectory to A')
    posteriors_history_goal_a_env1, num_possible_goals, goal_strings = get_posterior_of_trajectory(
        trajectory_to_goal_a, marginal_probability_next_state_table_env1)

    print('\nTrajectory to B')
    posteriors_history_goal_b_env1, num_possible_goals, goal_strings = get_posterior_of_trajectory(
        trajectory_to_goal_b, marginal_probability_next_state_table_env1)

    print('\nTrajectory to C')
    posteriors_history_goal_c_env1, num_possible_goals, goal_strings = get_posterior_of_trajectory(
        trajectory_to_goal_c, marginal_probability_next_state_table_env1)

    # Plotting Posterior Probabilities
    plot_posterior_of_trajectory(posteriors_history_goal_a_env1, num_possible_goals, goal_strings,
                                 title_goal_string='Goal: A,', title_environment_string='Environment: #1')

    plot_posterior_of_trajectory(posteriors_history_goal_b_env1, num_possible_goals, goal_strings,
                                 title_goal_string='Goal: B,', title_environment_string='Environment: #1')

    plot_posterior_of_trajectory(posteriors_history_goal_c_env1, num_possible_goals, goal_strings,
                                 title_goal_string='Goal: C,', title_environment_string='Environment: #1')

    # ================================================================================================================
    # Environment #2
    # Goal A
    perform_value_iteration_goal_a_env2 = ValueIteration(transition, reward_a_gap, value_table_initial,
                                                         convergence_tolerance, gamma, use_softmax=True,
                                                         use_noise=True, noise_beta=beta)
    optimal_value_table_a_env2, optimal_policy_table_a_env2 = perform_value_iteration_goal_a_env2()

    visualize_value_table(grid_width, grid_height, goal_states[2], trap_states_gap, optimal_value_table_a_env2)
    visualize_policy(grid_width, grid_height, goal_states[2], trap_states_gap, optimal_policy_table_a_env2)

    # Goal B
    perform_value_iteration_goal_b_env2 = ValueIteration(transition, reward_b_gap, value_table_initial,
                                                         convergence_tolerance, gamma, use_softmax=True,
                                                         use_noise=True, noise_beta=beta)
    optimal_value_table_b_env2, optimal_policy_table_b_env2 = perform_value_iteration_goal_b_env2()

    visualize_value_table(grid_width, grid_height, goal_states[0], trap_states_gap, optimal_value_table_b_env2)
    visualize_policy(grid_width, grid_height, goal_states[0], trap_states_gap, optimal_policy_table_b_env2)

    # Goal C
    perform_value_iteration_goal_c_env2 = ValueIteration(transition, reward_c_gap, value_table_initial,
                                                         convergence_tolerance, gamma, use_softmax=True,
                                                         use_noise=True, noise_beta=beta)
    optimal_value_table_c_env2, optimal_policy_table_c_env2 = perform_value_iteration_goal_c_env2()

    visualize_value_table(grid_width, grid_height, goal_states[1], trap_states_gap, optimal_value_table_c_env2)
    visualize_policy(grid_width, grid_height, goal_states[1], trap_states_gap, optimal_policy_table_c_env2)

    # Calculating the Marginal Probability Tables for Each Goal in Environment #2
    marginal_probability_next_state_a_env2 = get_probability_of_individual_state_transitions(
        transition, optimal_policy_table_a_env2)

    marginal_probability_next_state_b_env2 = get_probability_of_individual_state_transitions(
        transition, optimal_policy_table_b_env2)

    marginal_probability_next_state_c_env2 = get_probability_of_individual_state_transitions(
        transition, optimal_policy_table_c_env2)

    # Marginal Probabilities of the Next State Tables (Combined into a single array)
    marginal_probability_next_state_table_env2 = [marginal_probability_next_state_a_env2,
                                                  marginal_probability_next_state_b_env2,
                                                  marginal_probability_next_state_c_env2]

    # Calculating Posterior Probabilities
    print('\n\nEnvironment #2')
    print('\nTrajectory to A')
    posteriors_history_goal_a_env2, num_possible_goals, goal_strings = get_posterior_of_trajectory(
        trajectory_to_goal_a, marginal_probability_next_state_table_env2)

    print('\nTrajectory to B')
    posteriors_history_goal_b_env2, num_possible_goals, goal_strings = get_posterior_of_trajectory(
        trajectory_to_goal_b, marginal_probability_next_state_table_env2)

    print('\nTrajectory to C')
    posteriors_history_goal_c_env2, num_possible_goals, goal_strings = get_posterior_of_trajectory(
        trajectory_to_goal_c, marginal_probability_next_state_table_env2)

    # Plotting Posterior Probabilities
    plot_posterior_of_trajectory(posteriors_history_goal_a_env2, num_possible_goals, goal_strings,
                                 title_goal_string='Goal: A,', title_environment_string='Environment: #2')

    plot_posterior_of_trajectory(posteriors_history_goal_b_env2, num_possible_goals, goal_strings,
                                 title_goal_string='Goal: B,', title_environment_string='Environment: #2')

    plot_posterior_of_trajectory(posteriors_history_goal_c_env2, num_possible_goals, goal_strings,
                                 title_goal_string='Goal: C,', title_environment_string='Environment: #2')


if __name__ == '__main__':
    main()
