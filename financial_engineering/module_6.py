"""
Module 6: HYPOTHESIS TESTING
"""
import math


def t_statistic_of_population_mean(x_bar: float, null_mean: float, s: float, n: int) -> float:
    """
    hypothesis test of a population mean, a t-statistic with n-1 degrees of freedom

    Parameters
    ----------
    x_bar: sample mean
    null_mean: hypothesized population mean(the null)
    s: standard deviation of the sample
    n: sample size

    Returns
    -------
    the test statistic

    """

    return (x_bar - null_mean) / (s / math.sqrt(n))


def z_statistic_of_population_mean(x_bar: float, null_mean: float, sigma: float, n: int) -> float:
    """
    hypothesis test of a population mean, a z-statistic

    Parameters
    ----------
    x_bar: sample mean
    null_mean: hypothesized population mean(the null)
    sigma: standard deviation of the population (when the sample size is large and the population variance is unknown,
            sigma can be standard deviation of the sample)
    n: sample size

    Returns
    -------
    the test statistic

    """

    return (x_bar - null_mean) / (sigma / math.sqrt(n))


def t_statistic_of_differences_in_means(x_bar_1: float, x_bar_2: float, s_square_1: float, s_square_2: float,
                                        n1: int, n2: int) -> float:
    """
    hypothesis test of a difference in means, a t-statistic with (n1 + n2 -2) degrees of freedom

    Parameters
    ----------
    x_bar_1, x_bar_2: sample means
    s_square_1, s_square_2: variances of the two samples
    n1, n2: sample sizes

    Returns
    -------
    the test statistic

    """

    s_p_square = ((n1 - 1) * s_square_1 + (n2 - 1) * s_square_2) / (n1 + n2 -2)
    denominator = math.sqrt((s_p_square / n1) + (s_p_square / n2))
    return (x_bar_1 - x_bar_2) / denominator


def t_statistic_of_paired_comparisons_test(differences: list, mean_dz: float, n: int) -> float:
    """
    hypothesis test of paired comparisons test, a t-statistic with (n - 1) degrees of freedom

    Parameters
    ----------
    differences: differences between the ith pair of observations
    mean_dz: hypothesized mean of paired differences, which is commonly zero
    n: the number of paired observations

    Returns
    -------
    the test statistic

    """

    d_bar = sum(differences) / len(differences)
    s_d = 0
    for item in differences:
        s_d += pow((item - d_bar), 2)
    s_d = s_d / (n - 1)
    s_d_bar = s_d / math.sqrt(n)
    return (d_bar - mean_dz) / s_d_bar
