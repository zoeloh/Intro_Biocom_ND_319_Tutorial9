import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from plotnine import *

def monod(p, obs):
    UMAX = p[0]
    KS = p[1]
    sigma = p[2]
    
    expected = UMAX*(obs.S / (obs.S + KS))
    nll = -1 * norm(expected, sigma).logpdf(obs.u).sum()
    return nll
    
data = pandas.read_csv('MmarinumGrowth.csv', header = 0, sep = ',')

initialGuess = numpy.array([1,1,1])
fit = minimize(monod, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data)

print(fit.x)