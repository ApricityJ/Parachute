"""
Module 4: COMMON PROBABILITY DISTRIBUTIONS
"""
from typing import List, Tuple

"""
Shortfall risk is the probability that a portfolio value or return will fall below a particular (target) value or 
return over a given time period.

Roy’s safety-first criterion states that the optimal portfolio minimizes the probability that the return of the 
portfolio falls below some minimum acceptable level.
minimize P(R_p < R_L) where R_p :=portfolio return, R_L :=threshold level return
"""


def calculate_normally_distributed_SFRatio(portfolio_params: List[list],
                                           threshold_level: float) -> Tuple[List[float], int]:
    """
    maximize the SFRatio, where SFRatio := (E(portfolio_return) - threshold_level_return) / standard_deviation_of_return

    Parameters
    ----------
    portfolio_params: list of [expect, standard deviation] of every portfolio
    threshold_level: threshold level of return

    return: results and the max index(begin with 1)
    """
    results = [((portfolio[0] - threshold_level) / portfolio[1]) for portfolio in portfolio_params]
    return results, results.index(max(results)) + 1


# test
# print(calculate_normally_distributed_SFRatio([[9, 12], [11, 20], [6.6, 8.2]], 3))
