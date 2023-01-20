average_daily_salary = 770 # before tax: 1028, after tax: 770
working_environment_coefficient = 1.1
heterosexual_environment_coefficient = 1
colleague_environment_coefficient = 1.05
working_hours = 8
commuting_time = 1
fishing_time = 2.5
education_coefficient = 1.8
start_time = 9

comprehensive_environmental_coefficient = 1 * working_environment_coefficient * heterosexual_environment_coefficient * colleague_environment_coefficient
cost_effective_of_work = average_daily_salary * comprehensive_environmental_coefficient / (
        35 * (working_hours + commuting_time - 0.5 * fishing_time) * education_coefficient)

if start_time <= 8:
    cost_effective_of_work *= 0.95

print('your cost-effective of work is ', cost_effective_of_work)
if cost_effective_of_work < 0.8:
    print('poor guy, you better change your job!')
elif cost_effective_of_work >= 0.8 and cost_effective_of_work <= 1.5:
    print('Not bad, but you can get a better one.')
elif cost_effective_of_work > 1.5 and cost_effective_of_work <= 2.0:
    print('you have a good job, come on!')
else:
    print('your job is soooooooooooooooooooo great!!!')
