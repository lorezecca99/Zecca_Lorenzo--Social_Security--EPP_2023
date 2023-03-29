import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def predict_eff(data):
    """Fit a quadratic polynomial function of age to the household’s log hourly wage, by
    education to predict the age-efficiency profile of US workers between 2014 and 2018."""
    # Generate some sample data
    X = data[['age','age2','col']]
    y = data['lwage']
    weights = data['hhwgt']

    # Create a LinearRegression object and fit the model using the weighted data
    lr = LinearRegression()
    lr.fit(X, y, sample_weight=weights)

    # Predict fitted values and residuals
    data['lwage_fit'] = lr.predict(X)

    data['resid'] = y - data['lwage_fit']

    # Generate new variable
    data['wage_fit'] = np.exp(data['lwage_fit'])

    # Compute average wage in sample
    result = data['wage_fit'].mean()

    # Normalize wages
    data['wage_fit_norm'] = data['wage_fit'] / result

    data = data.sort_values(['age', 'col'])
    data = data.drop_duplicates(subset=['age', 'col'], keep='first')

    data = data[['age', 'wage_fit_norm', 'col']]
    data = data.sort_values(['col', 'age'])
    data = data.pivot(index='age', columns='col', values='wage_fit_norm')

    """For the sake of simplicity, let us take the average between the two types of workers. 
    Later on, we will, distinguish the two types of workers following the approach 
    adopted by Conesa and Krueger (1999)."""
    data['average_eff'] = (data[0] + data[1]) / 2

    return data
print('Age-efficiency estimated')

def predict_eff_age(data):
    """Function to plot the age-efficiency profile."""
    #data = data.pivot(index=data[]'age', columns='col', values='wage_fit_norm')
    plt.figure(1)
    plt.plot(data.index, data['average_eff'])
    plt.ylabel('Average efficiency')
    plt.xlabel('Age')
    return plt.figure(1)