import numpy as np


def clean_data(data):
    """Function for cleaning the data set. 
    The sample includes household heads only, which can detected with
    the variable “Relationship to household head” (hhrel2).
    The household head is of working age that we define to be 20–64 years 
    old, and works at least 260 hours annualy.
    The total annual earnings of the head are strictly positive. 
    Define total earnings as the sum of wage income from wage and 
    salary and income from self-employment. The total annual earnings of the head are 
    strictly positive, so we can also drop  households who report a 
    zero number of weeks worked.
    """
    data = data[data['hhrel2']=="Householder"]
    data = data[(data['age'] >= 20) & (data['age'] <= 64)]
    data['hrs'] = data['uhours'] * data['weeks']
    data = data[data['hrs'] >= 260]
    data['rincp_wag'].replace({np.nan: 0}, inplace=True)
    data['rincp_se'].replace({np.nan: 0}, inplace=True)
    data['hh_earnings'] = data['rincp_wag'] + data['rincp_se']
    data = data[data['hh_earnings'] > 0]
    data = data[data['weeks'] != 0]

    """Generate a dummy variable col for the household’s educational 
    level by assuming that a household head is a college graduate if 
    he/she has a completed college degree or higher (col = 1); otherwise,
    a household head is a high school graduate."""

    data['col'] = data['educ'].apply(lambda x: 1 if x in ['Some college','College', 'Advanced'] else 0)

    #  Compute the household head’s hourly wage by dividing total annual earning by theannual hours worked.

    data['hourly_wage'] = data['hh_earnings'] / data['hrs']

    data = data[['hourly_wage', 'hhwgt', 'age', 'col']]

    data['lwage'] = np.log(data['hourly_wage'])
    data['age2'] = data['age'].apply(lambda x: x**2)
    return data
print('Data cleaned')
