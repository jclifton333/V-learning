# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:18:39 2017

@author: Jesse

Implementing optimization similar to that described in https://arxiv.org/pdf/1611.03531.pdf
  "In order to avoid local maxima,
  simulated annealing with 1000 function evaluations is used to find a neighborhood of
  the maximum; this solution is then used as the starting value for the BFGS algorithm."
"""

NUM_MULTISTART = 5 #Number of random starts for multistart option 

import scipy.optimize as optim
import numpy as np
import multiprocessing as mp
from functools import partial


def minimize_worker(x0, objective, out_q): 
  '''
  Worker for multistart multiprocessing; pushes result of 
  minimization with initial value x0 to queue. 
  
  :param x0: Initial iterate 
  :param objective: Objective to be minimize (V-learning objective) 
  :param out_q: mp.Queue() 
  '''
  res = optim.minimize(objective, x0, method='L-BFGS-B')
  out_q.put(res)   

def mp_multistart(random_starts, objective): 
  '''
  Multiprocessing for the multistart initialization option. 
  
  :param random_starts: list of initial values for minimization routine 
  :param objective: function to be minimized (V-learning objective) 
  :return results_list: list of optim.minimize results corresponding to random_starts 
  '''
  out_q = mp.Queue() 
  procs = [] 
  for i in range(len(random_starts)): 
    p = mp.Process(target=minimize_worker, args=(random_starts[i], objective, out_q))
    p.daemon = False
    procs.append(p)
    p.start() 
  results_lst = [] 
  
  for i in range(len(procs)): 
    results_lst.append(out_q.get())
  
  for p in procs:
    p.join() 
  return results_lst 

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
    random_starts = [x0] + [np.random.multivariate_normal(mean=np.zeros(len(x0)),cov=100*np.eye(len(x0))) for start in range(NUM_MULTISTART-1)]
    #results = mp_multistart(random_starts, objective)
    results = [optim.minimize(objective, x0=random_start, method='L-BFGS-B') for random_start in random_starts]
    func_vals = [r.fun for r in results]
    res = results[np.argmin(func_vals)]
  elif initializer is None: 
    res = optim.minimize(objective, x0=x0, method='L-BFGS-B')  
  else: 
    raise ValueError('Invalid initializer value.')  
  return res.x
    