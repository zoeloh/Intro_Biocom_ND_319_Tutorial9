import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from plotnine import *

def linear(p, obs):
    B0 = p[0]
    B1 = p[1]
    sigma = p[2]
    
    expected = B0 + (B1 * obs.mutation)
    nll = -1 * norm(expected, sigma).logpdf(obs.ponzr1Counts).sum()
    return nll
    
def nullH(p, obs):
    B0 = p[0]
    sigma = p[1]
    
    
    expected = B0
    nll = -1 * norm(expected, sigma).logpdf(obs.ponzr1Counts).sum()
    return nll
    

data = pandas.read_csv('ponzr1.csv', header = 0, sep = ',')
data['mutation'] = data["mutation"].map({'WT' : 0, 'M124K' : 1, 'V456D' : 2, 'I213N' : 3})

subset=data.loc[data.mutation.isin(['0','1']),:]

initialLinGuess = numpy.array([1,1,1])
linfit = minimize(linear, initialLinGuess, method="Nelder-Mead", options={'disp': True}, args=subset)

initialNullGuess = numpy.array([1,1])
nullfit = minimize(nullH, initialNullGuess, method="Nelder-Mead", options={'disp': True}, args=subset)

linfit.fun
nullfit.fun

D=2*(nullfit.fun - linfit.fun)
1-chi2.cdf(x=D, df=1)
