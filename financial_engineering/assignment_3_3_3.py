"""
A program that prints out the monthly amortization schedule.
The inputs are the loan, the years to pay and the annual interest rate.

"""


def print_monthly_amortization_schedule(loan: float, years: int, year_interest_rate: float) -> None:
	monthly_interest_rate = year_interest_rate / 12
	# according to eq.3.6 on page 34, calculate C
	monthly_to_pay = loan / ((1 - pow((1 + monthly_interest_rate), -(years * 12))) / monthly_interest_rate)

	print('{:20s}{:20s}{:20s}{:20s}{:20s}'.format('Month', 'Payment', 'Interest', 'Principal', 'Remaining principal'))
	print('{:>90.3f}'.format(loan))

	total_payment = 0
	total_interest = 0
	total_principal = 0
	remaining_principal = loan
	for i in range(1, years*12 + 1):
		interest = remaining_principal * monthly_interest_rate
		principal = monthly_to_pay - interest
		remaining_principal -= principal

		total_interest += interest
		total_principal += principal
		total_payment += monthly_to_pay
		print(f'{i:<20d}{monthly_to_pay:<20,.3f}{interest:<20,.3f}{principal:<20,.3f}{remaining_principal:<20,.3f}')
	print(f'{"Total":20s}{total_payment:<20,.3f}{total_interest:<20,.3f}{total_principal:<20,.3f}')


# test
print_monthly_amortization_schedule(250_000, 15, 0.08)
