# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:18:39 2017

@author: Jesse

Implementing optimization similar to that described in https://arxiv.org/pdf/1611.03531.pdf
  "In order to avoid local maxima,
  simulated annealing with 1000 function evaluations is used to find a neighborhood of
  the maximum; this solution is then used as the starting value for the BFGS algorithm."
"""

NUM_MULTISTART = 5 #Number of random starts for mulitstart option 

import scipy.optimize as optim
import numpy as np

def VLopt(objective, x0, initializer=None):
  '''
  Used to optimize V-learning objective.  
  
  :param objective: V-learning objective function 
  :param x0: initial value 
  :param initializer: value in [None, 'basinhop', 'multistart'] for method to initialize L-BFGS-B optimizer.
  :return res.x: optimal parameter array returned by initializer + L-BFGS-B
  '''    
  if initializer == 'basinhop': 
    x0 = optim.basinhopping(objective, x0=x0, niter=100).x
    res = optim.minimize(objective, x0=x0, method='L-BFGS-B')
  elif initializer == 'multistart': 
    random_starts = [np.random.normal(size=len(x0)) for start in range(NUM_MULTISTART)]
    results = [optim.minimize(objective, random_start) for random_start in random_starts]
    #for r in results: 
    #  print('func value: {} beta hat: {}'.format(r.fun, r.x))
    func_vals = [r.fun for r in results]
    #print('Multistart function values: {}'.format(func_vals))
    res = results[np.argmin(func_vals)]
  elif initializer is None: 
    res = optim.minimize(objective, x0=x0, method='L-BFGS-B')  
  else: 
    raise ValueError('Invalid initializer value.')  
  return res.x
    