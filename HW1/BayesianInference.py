#!/usr/bin/env python
""" BayesianInference.py

This program performs Bayesian Inference

Course: STAT 232C: Cognitive Artificial Intelligence
Author: Zachary Lacey
Date: April 2nd, 2021
"""

__author__ = 'Zachary Lacey'

import numpy as np

def getPosterior(priorOfA, priorOfB, likelihood):
    """ Get Posterior Function

    Calculates the Posterior

    Args:
        priorOfA: This is a Probability Mass Function (PMF). This variable is a dictionary that holds all the possible
        outcomes in A and the values are their prior probabilities. P(A)

        priorOfB: This is a Probability Mass Function (PMF). This variable is a dictionary that holds all the possible
        outcomes in B and the values are their prior probabilities. P(B)

        likelihood: This is the likelihood that the outcome of A and the outcome of B occur. P(D | A, B)

    Returns:
        marginalOfA: This is a Probability Mass Function (PMF). This variable is a dictionary that holds all the
        possible marginal posterior probabilities for each outcome of A. Marginal means that we want to consider the
        posterior of the outcomes of parameter A separately from the outcomes of parameter B. From this, we must perform
        the summation of P(A, B | D) across all possible outcomes of parameter B to determine the marginal posterior of
        the outcomes of parameter A.

            P(A | D) = sum( P(A, B | D) )dB

        marginalOfB: This os a Probability Mass Function (PMF). This variable is a dictionary that holds all the
        possible marginal posterior probabilities for each outcome of B. Marginal means that we want to consider the
        posterior of the outcomes of parameter B separately from the outcomes of parameter A. From this, we must perform
        the summation of P(A, B | D) across all possible outcomes of parameter A to determine the marginal posterior of
        the outcomes of parameter B.

            P(B | D) = sum( P(A, B | D) )dA

    """

    # Initialize the dictionary of marginal posterior
    marginalOfA = dict()
    marginalOfB = dict()

    # Define the number of outcomes
    num_of_outcomes_a = len(priorOfA.keys())
    num_of_outcomes_b = len(priorOfB.keys())

    # Initialize the posterior matrix where the posterior values are stored. These are determined by the Bayesian
    # Inference
    posterior_matrix = np.zeros((num_of_outcomes_a, num_of_outcomes_b))

    # Numerator of the Bayesian Rule
    for i in range(num_of_outcomes_a):
        for j in range(num_of_outcomes_b):
            posterior_matrix[i][j] = likelihood[(list(priorOfA)[i], list(priorOfB)[j])] * priorOfA[list(priorOfA)[i]]\
                                     * priorOfB[list(priorOfB)[j]]

    # Denomenator of the Bayesian Rule
    sumOfMatrix = np.sum(posterior_matrix)
    posterior_matrix /= sumOfMatrix

    # Determine the marginal posteriors via summation
    # marginal posterior of A is the summation across the column dimension.
    # marginal posterior of B is the summation across the row dimension.
    marginal_posterior_of_a = np.sum(posterior_matrix, axis=1)
    marginal_posterior_of_b = np.sum(posterior_matrix, axis=0)

    # Variable used to specify the index for the posterior values
    iter_a = 0
    iter_b = 0

    # Storing the marginal posterior values in the dictionary format
    for idx_a in priorOfA.keys():
        marginalOfA[idx_a] = marginal_posterior_of_a[iter_a]
        iter_a += 1

    for idx_b in priorOfB.keys():
        marginalOfB[idx_b] = marginal_posterior_of_b[iter_b]
        iter_b += 1

    return ([marginalOfA, marginalOfB])


def main():
    exampleOnePriorofA = {'a0': .5, 'a1': .5}
    exampleOnePriorofB = {'b0': .25, 'b1': .75}
    exampleOneLikelihood = {('a0', 'b0'): 0.42, ('a0', 'b1'): 0.12, ('a1', 'b0'): 0.07, ('a1', 'b1'): 0.02}
    print(getPosterior(exampleOnePriorofA, exampleOnePriorofB, exampleOneLikelihood))

    exampleTwoPriorofA = {'red': 1 / 10, 'blue': 4 / 10, 'green': 2 / 10, 'purple': 3 / 10}
    exampleTwoPriorofB = {'x': 1 / 5, 'y': 2 / 5, 'z': 2 / 5}
    exampleTwoLikelihood = {('red', 'x'): 0.2, ('red', 'y'): 0.3, ('red', 'z'): 0.4, ('blue', 'x'): 0.08,
                            ('blue', 'y'): 0.12, ('blue', 'z'): 0.16, ('green', 'x'): 0.24, ('green', 'y'): 0.36,
                            ('green', 'z'): 0.48, ('purple', 'x'): 0.32, ('purple', 'y'): 0.48, ('purple', 'z'): 0.64}
    print(getPosterior(exampleTwoPriorofA, exampleTwoPriorofB, exampleTwoLikelihood))


if __name__ == '__main__':
    main()
