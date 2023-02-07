"""
Module 5: SAMPLING AND ESTIMATION
"""
from typing import Tuple
import math


def calculate_confidence_interval(x_bar: float, reliability_factor: float, sigma: float, n: int) -> Tuple[float, float]:
    """
    calculate the confidence interval for the population mean

    Parameters
    ----------
    x_bar: point estimate of the population mean
    reliability_factor: z(alpha/2) when z-statistic, t(alpha/2) when t-statistic (need to look for a table)
    sigma: population standard deviation when z-statistic, sample standard deviation when t-statistic
    n: sample size

    Returns
    -------
    confidence interval for the population mean

    """

    upper_bound = x_bar + reliability_factor * sigma / math.sqrt(n)
    lower_bound = x_bar - reliability_factor * sigma / math.sqrt(n)
    return lower_bound, upper_bound


# test
# print(calculate_confidence_interval(80, 2.58, 15, 36))
