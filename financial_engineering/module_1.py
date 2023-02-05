"""
Module 1: THE TIME VALUE OF MONEY
"""
from typing import List

"""
Module 1.1: EAY AND COMPOUNDING FREQUENCY
    EAY: effective annual rate / effective annual yield
"""


def calculate_EAY(annual_rate: float, periods_per_year: int) -> float:
    """
    calculate EAY:= (1 + periodic_rate)^periods_per_year - 1

    Parameters
    ----------
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    """
    periodic_rate = annual_rate / periods_per_year
    return pow((1 + periodic_rate), periods_per_year) - 1


# test
# print(calculate_EAY(0.12, 4))


"""
Module 1.2: CALCULATING PV AND FV
    PV: present value, FV: future value
    Annuity: An annuity is a stream of equal cash flows that occurs at equal intervals over a given period.
"""


def calculate_FV(present_value: float, annual_rate: float, periods_per_year: int, years: int) -> float:
    """
    calculate FV:= PV * (1 + periodic_rate)^total_num_of_compounding_periods

    Parameters
    ----------
    present_value: PV
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    years: compounding years
    """
    periodic_rate = annual_rate / periods_per_year
    total_num_of_compounding_periods = periods_per_year * years
    return present_value * pow((1 + periodic_rate), total_num_of_compounding_periods)


# test
# print(calculate_FV(200, 0.1, 1, 2))


def calculate_PV(future_value: float, annual_rate: float, periods_per_year: int, years: int) -> float:
    """
    calculate PV:= FV * [1 /(1 + periodic_rate)^total_num_of_compounding_periods]

    Parameters
    ----------
    future_value: FV
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    years: compounding years
    """
    periodic_rate = annual_rate / periods_per_year
    total_num_of_compounding_periods = periods_per_year * years
    return future_value * (1 / pow((1 + periodic_rate), total_num_of_compounding_periods))


# test
# print(calculate_PV(200, 0.1, 1, 2))


def calculate_FV_of_ordinary_annuity(cash_flow_per_period: float, annual_rate: float, periods_per_year: int, years: int) -> float:
    """
    calculate FV_ordinary_annuity:= C * [((1 + periodic_rate)^number_of_payments - 1) / periodic_rate]

    Parameters
    ----------
    cash_flow_per_period: C in the formula
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    years: compounding years
    """
    periodic_rate = annual_rate / periods_per_year
    number_of_payments = periods_per_year * years
    return cash_flow_per_period * ((pow((1 + periodic_rate), number_of_payments) - 1) / periodic_rate)


# test
# print(calculate_FV_of_ordinary_annuity(200, 0.1, 1, 3))


def calculate_PV_of_ordinary_annuity(cash_flow_per_period: float, annual_rate: float, periods_per_year: int, years: int) -> float:
    """
    calculate PV_ordinary_annuity:= C * [(1 - (1 + periodic_rate)^(-number_of_payments)) / periodic_rate]

    Parameters
    ----------
    cash_flow_per_period: C in the formula
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    years: compounding years
    """
    periodic_rate = annual_rate / periods_per_year
    number_of_payments = periods_per_year * years
    return cash_flow_per_period * ((1 - pow((1 + periodic_rate), -number_of_payments)) / periodic_rate)


# test
# print(calculate_PV_of_ordinary_annuity(200, 0.1, 1, 3))


def calculate_FV_of_annuity_due(cash_flow_per_period: float, annual_rate: float, periods_per_year: int, years: int) -> float:
    """
    calculate FV_annuity_due:= C * [((1 + periodic_rate)^number_of_payments - 1) / periodic_rate] * (1 + periodic_rate)

    Parameters
    ----------
    cash_flow_per_period: C in the formula
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    years: compounding years
    """
    periodic_rate = annual_rate / periods_per_year
    number_of_payments = periods_per_year * years
    return cash_flow_per_period * ((pow((1 + periodic_rate), number_of_payments) - 1) / periodic_rate) * (1 + periodic_rate)


# test
# print(calculate_FV_of_annuity_due(200, 0.1, 1, 3))


def calculate_PV_of_annuity_due(cash_flow_per_period: float, annual_rate: float, periods_per_year: int, years: int) -> float:
    """
    calculate PV_annuity_due:= C * [(1 - (1 + periodic_rate)^(-number_of_payments)) / periodic_rate] * (1 + periodic_rate)

    Parameters
    ----------
    cash_flow_per_period: C in the formula
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    years: compounding years
    """
    periodic_rate = annual_rate / periods_per_year
    number_of_payments = periods_per_year * years
    return cash_flow_per_period * ((1 - pow((1 + periodic_rate), -number_of_payments)) / periodic_rate) * (1 + periodic_rate)


# test
# print(calculate_PV_of_annuity_due(200, 0.1, 1, 3))


def calculate_FV_of_uneven_cash_flow(cash_flow_list: List[float], annual_rate: float, periods_per_year: int) -> float:
    """
    calculate FV of uneven cash flow series

    Parameters
    ----------
    cash_flow_list: cash flow series
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    """
    cash_flow_length = len(cash_flow_list)
    total = 0
    for i, item in enumerate(cash_flow_list):
        total += calculate_FV(item, annual_rate, periods_per_year, cash_flow_length - i - 1)
    return total


# test
# print(calculate_FV_of_uneven_cash_flow([300, 600, 200], 0.1, 1))


def calculate_PV_of_uneven_cash_flow(cash_flow_list: List[float], annual_rate: float, periods_per_year: int) -> float:
    """
    calculate PV of uneven cash flow series

    Parameters
    ----------
    cash_flow_list: cash flow series
    annual_rate: stated annual rate
    periods_per_year: the number of compounding periods per year
    """
    total = 0
    for i, item in enumerate(cash_flow_list):
        total += calculate_PV(item, annual_rate, periods_per_year, i + 1)
    return total


# test
print(calculate_PV_of_uneven_cash_flow([300, 600, 200], 0.1, 1))
