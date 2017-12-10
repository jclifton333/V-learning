# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:18:39 2017

@author: Jesse

Implementing optimization similar to that described in https://arxiv.org/pdf/1611.03531.pdf
  "In order to avoid local maxima,
  simulated annealing with 1000 function evaluations is used to find a neighborhood of
  the maximum; this solution is then used as the starting value for the BFGS algorithm."
"""

import scipy.optimize as optim

def VLopt(objective, x0, basinhop=True):
  if basinhop: 
    x0 = optim.basinhopping(objective, x0=x0, niter=1000).x
  xOpt = optim.minimize(objective, x0=x0, method='L-BFGS-B').x
  return xOpt
    