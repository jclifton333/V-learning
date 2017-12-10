# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:39:38 2017

@author: Jesse

This file contains functions that return actions and probabilities associated with policies in V-learning.
"""

import numpy as np

def piBin(s, beta):
  '''
  For binary action space ; returns the probability of taking action 1 at state s given policy
  parameters beta. 
  
  Parameters
  ----------
  s: state (1d array)
  beta: policy parameters (1d array)
  
  Returns
  -------
  Value of softmax policy with parameters beta at state s.  
  '''
  
  dot = np.dot(s, beta)
  max_ = np.max((dot,0)) #Subtract this in exponent for numerical stability
  return np.exp(dot - max_) / (np.exp(-max_) + np.exp(dot - max_))
 
  
def policyProbsBin(a, s, beta, eps = 0.0):
  '''
  For binary action space ; returns probabiltiy of taking action a at state s given policy parameters
  beta and epsilon = eps. 
  
  Parameters
  ----------
  a: binary action (0 or 1)
  s: state (1d array)
  beta: policy parameters (1d array)
  eps: epsilon used for epsilon-greedy
  
  Returns
  -------
  Probability of taking action a at state s under policy with parameters beta and eps-greedy  
  '''    
  
  p = piBin(s, beta)
  return a*p*(1-eps) + (1-a)*(1-p*(1-eps))

