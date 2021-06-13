#!/usr/bin/env python
""" BayesianInference.py
This program performs Bayesian Inference
Course: STAT 232C: Cognitive Artificial Intelligence
Author: Zachary Lacey
Date: April 2nd, 2021
"""

__author__ = 'Zachary Lacey'

import numpy as np


def get_posterior(prior_of_a, prior_of_b, likelihood):
    """ Get Posterior Function

    Calculates the Posterior

    Args:
        prior_of_a: This is a Probability Mass Function (PMF). This variable is a dictionary that holds all the possible

        outcomes in A and the values are their prior probabilities. P(A)

        prior_of_b: This is a Probability Mass Function (PMF). This variable is a dictionary that holds all the possible
        outcomes in B and the values are their prior probabilities. P(B)

        likelihood: This is the likelihood that the outcome of A and the outcome of B occur. P(D | A, B)
    Returns:
        marginal_of_a: This is a Probability Mass Function (PMF). This variable is a dictionary that holds all the
        possible marginal posterior probabilities for each outcome of A. Marginal means that we want to consider the
        posterior of the outcomes of parameter A separately from the outcomes of parameter B. From this, we must perform
        the summation of P(A, B | D) across all possible outcomes of parameter B to determine the marginal posterior of
        the outcomes of parameter A.

            P(A | D) = sum( P(A, B | D) )dB

        marginal_of_b: This os a Probability Mass Function (PMF). This variable is a dictionary that holds all the
        possible marginal posterior probabilities for each outcome of B. Marginal means that we want to consider the
        posterior of the outcomes of parameter B separately from the outcomes of parameter A. From this, we must perform
        the summation of P(A, B | D) across all possible outcomes of parameter A to determine the marginal posterior of
        the outcomes of parameter B.

            P(B | D) = sum( P(A, B | D) )dA

    """

    # Initialize the dictionary of marginal posterior
    marginal_of_a = dict()
    marginal_of_b = dict()

    # Define the number of outcomes
    num_of_outcomes_a = len(prior_of_a.keys())
    num_of_outcomes_b = len(prior_of_b.keys())

    # Initialize the posterior matrix where the posterior values are stored. These are determined by the Bayesian
    # Inference
    posterior_matrix = np.zeros((num_of_outcomes_a, num_of_outcomes_b))

    # Numerator of the Bayesian Rule
    for i in range(num_of_outcomes_a):
        for j in range(num_of_outcomes_b):
            posterior_matrix[i][j] = likelihood[(list(prior_of_a)[i],
                                                 list(prior_of_b)[j])] \
                                     * prior_of_a[list(prior_of_a)[i]] \
                                     * prior_of_b[list(prior_of_b)[j]]

    # Denomenator of the Bayesian Rule
    sum_of_matrix = np.sum(posterior_matrix)
    posterior_matrix /= sum_of_matrix

    # Determine the marginal posteriors via summation
    # marginal posterior of A is the summation across the column dimension.
    # marginal posterior of B is the summation across the row dimension.
    marginal_posterior_of_a = np.sum(posterior_matrix, axis=1)
    marginal_posterior_of_b = np.sum(posterior_matrix, axis=0)

    # Variable used to specify the index for the posterior values
    iter_a = 0
    iter_b = 0

    # Storing the marginal posterior values in the dictionary format
    for idx_a in prior_of_a.keys():
        marginal_of_a[idx_a] = marginal_posterior_of_a[iter_a]
        iter_a += 1

    for idx_b in prior_of_b.keys():
        marginal_of_b[idx_b] = marginal_posterior_of_b[iter_b]
        iter_b += 1

    return [marginal_of_a, marginal_of_b]


def main():
    example_one_prior_of_a = {'a0': .5, 'a1': .5}
    example_one_prior_of_b = {'b0': .25, 'b1': .75}
    example_one_likelihood = {('a0', 'b0'): 0.42, ('a0', 'b1'): 0.12, ('a1', 'b0'): 0.07, ('a1', 'b1'): 0.02}
    print(get_posterior(example_one_prior_of_a, example_one_prior_of_b, example_one_likelihood))

    example_two_prior_of_a = {'red': 1 / 10, 'blue': 4 / 10, 'green': 2 / 10, 'purple': 3 / 10}
    example_two_prior_of_b = {'x': 1 / 5, 'y': 2 / 5, 'z': 2 / 5}
    example_two_likelihood = {('red', 'x'): 0.2, ('red', 'y'): 0.3, ('red', 'z'): 0.4, ('blue', 'x'): 0.08,
                              ('blue', 'y'): 0.12, ('blue', 'z'): 0.16, ('green', 'x'): 0.24, ('green', 'y'): 0.36,
                              ('green', 'z'): 0.48, ('purple', 'x'): 0.32, ('purple', 'y'): 0.48, ('purple', 'z'): 0.64}
    print(get_posterior(example_two_prior_of_a, example_two_prior_of_b, example_two_likelihood))


if __name__ == '__main__':
    main()
