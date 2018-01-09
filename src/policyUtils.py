# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:39:38 2017

@author: Jesse

This file contains functions that return actions and probabilities associated with policies in V-learning.
"""

import numpy as np

def pi(s, beta): 
  '''
  Return array of action probabilities corresponding to rows of beta. 
  :param s: state array at which to compute policy 
  :param beta: nA x nS array of policy parameters, rows corresponding to each action 
  :return: array of probabilities of each action 
  '''
  dots = np.array([np.dot(s, b) for b in beta])
  max_ = np.max(dots) 
  exp_ = np.exp(dots - max_)  #Subtract this in exponent to avoid overflow
  return exp_ / np.sum(exp_) 
  
def policyProbs(a, s, beta, eps = 0.0): 
  '''
  :param a: integer or onehot encoding of action 
  :param s: state array 
  :param beta: nA x nS array of policy parameters 
  :param eps: epsilon used in epsilon-greedy 
  :return: probability of action a in state s under policy with parameters beta. 
  '''
  p_list = pi(s, beta) 
  nA = beta.shape[0]
  if isinstance(a, int):
    a_vec = np.zeros(nA)
    a_vec[a] = 1
  else:
    a_vec = a 
  p_a = np.dot(a_vec, p_list) 
  return p_a * (1 - eps) + (eps / nA) 
 
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
  max_ = np.max((dot,0)) #Subtract this in exponent to avoid overflow
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

