import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

def nllike(p, obs):
    B0 = p[0]
    B1 = p[1]
    sigma = p[2]
    
    expected = B0 + (B1 * obs.x)
    nll = -1 * norm(expected, sigma).logpdf(obs.y).sum()
    return nll

data = pandas.read_csv('ponzr1.csv', header = 0, sep = ',')
