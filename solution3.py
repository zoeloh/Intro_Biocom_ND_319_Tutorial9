import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from plotnine import *

def const(p, obs):
    a = p[0]
    sigma = p[1]
    
    expected = a
    nll = -1 * norm(expected, sigma).logpdf(obs.decomp).sum()
    return nll
    
def linear(p, obs):
    a = p[0]
    b = p[1]
    sigma = p[2]
    
    expected = a + b * obs.Ms
    nll = -1 * norm(expected, sigma).logpdf(obs.decomp).sum()
    return nll
    
def hump(p, obs):
    a = p[0]
    b = p[1]
    c = p[2]
    sigma = p[3]
    
    expected = a + b*obs.Ms + c*(obs.Ms) * (obs.Ms)
    nll = -1 * norm(expected, sigma).logpdf(obs.decomp).sum()
    return nll
    
data = pandas.read_csv('leafDecomp.csv', header = 0, sep = ',')

initialGuess = numpy.array([1,1,1,1])
constfit = minimize(const, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data)
linfit = minimize(linear, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data)
humpfit = minimize(hump, initialGuess, method="Nelder-Mead", options={'disp': True}, args=data)


print(humpfit.x)

D=2*(const.fun - linfit.fun)
1-chi2.cdf(x=D, df=1)
