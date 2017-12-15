#!/usr/bin/python3
'''
Created on Thu Oct 12 00:32:24 2017

@author: Jesse

This file contains functions that implement V-learning.

theta refers to v-function parameters, 
beta refers to policy parameters. 

\hat{theta}_T = (M_T)^-1 . v_T, where
  M_T = sum_t w_t * <psi(s_t), gamma * psi(s_tp1) - psi(s_t)>
  v_T = -sum_t w_t * r_t psi(s_t)
  w_t is the importance sampling ratio (depends on policy)
'''

import numpy as np
from VLopt import VLopt
import pdb

def thetaPi(beta, policyProbs, eps, M, A, R, Xbeta, Xtheta, Mu, btsWts):  
  '''
  Estimates theta associated with policy indexed by beta.    
  
  Parameters
  ----------
  beta : policy parameters (1d array)
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M : array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (3d array of size T x nV x nV)
  A : array of actions (1d or 2d array)
  R : array of rewards (1d array)
  Xbeta : Policy features at each timestep (2d array of size T x nPi)
  Xtheta : V-function features at each timestep (2d array of size T x nV)
  Mu : Probabilities of observed actions under policies mu_t (1d array of size T)
  
  Returns
  -------
  Estimate of theta 
  '''
  T, p = Xtheta.shape[0] - 1, Xtheta.shape[1]
  if len(A.shape) == 1:
    w = np.array([btsWts[i] * float(policyProbs(A[i], Xbeta[i,:], beta, eps=eps)) / Mu[i] for i in range(T)])
  else:
    w = np.array([btsWts[i] * float(policyProbs(A[i,:], Xbeta[i,:], beta, eps=eps)) / Mu[i] for i in range(T)])
  sumRS = np.sum(np.multiply(Xtheta[:-1,:], np.multiply(w, R).reshape(T,1)), axis=0)
  sumM  = np.sum(np.multiply(M, w.reshape(T, 1, 1)), axis=0)
  sumMInv = np.linalg.inv(sumM + 0.01*np.eye(p))
  return np.dot(sumMInv, sumRS)

def vPi(beta, policyProbs, eps, M, A, R, Xbeta, Xtheta, Mu, btsWts, refDist=None):
  '''
  Returns estimated value of policy indexed by beta.  
  
  Parameters
  ----------
  beta : policy parameters (1d array)
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M : array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (3d array of size T x nV x nV)
  A : array of actions (1d or 2d array)
  R : array of rewards (1d array)
  Xbeta : Policy features at each timestep (2d array of size T x nPi)
  Xtheta : V-function features at each timestep (2d array of size T x nV)
  Mu : Probabilities of observed actions under policies mu_t (1d array of size T)
  refDist : Reference distribution for estimating value (2d array with v-function features as rows).
           If None, uses Xtheta as reference distribution.  
  
  Returns
  -------
  (Negative) estimated value of policy indexed by beta wrt refDist.  
  '''
  
  theta = thetaPi(beta, policyProbs, eps, M, A, R, Xbeta, Xtheta, Mu, btsWts)
  if refDist is None:
    return -np.mean(np.dot(Xtheta, theta))
  else: 
    return -np.mean(np.dot(refDist, theta))

def betaOpt(policyProbs, eps, M, A, R, Xbeta, Xtheta, Mu, wStart=None, refDist=None, bts=True):
  '''
  Optimizes policy value over class of softmax policies indexed by beta. 
  Currently only working for binary action spaces! 
  
  Parameters
  ----------
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M: array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (3d array of size T x nV x nV)
  A: array of actions (1d or 2d array)
  R: array of rewards (1d array)
  Xbeta: Policy features at each timestep (2d array of size T x nPi)
  Xtheta: V-function features at each timestep (2d array of size T x nV)
  Mu: Probabilities of observed actions under policies mu_t (1d array of size T)
  wStart : Warm start for beta optimization (1d array). 
           If None, initializes to all 0s.  
  refDist: Reference distribution for estimating value (2d array with v-function features as rows).
           If None, use Xtheta as reference distribution.  

  
  Returns
  -------
  Estimate of beta
  '''
  nPi = Xbeta.shape[1]
  if Xtheta.shape[0] < Xtheta.shape[1]: 
    return np.zeros(nPi) 
  else:    
    if bts: 
      btsWts = np.random.exponential(size = Xtheta.shape[0] - 1) 
    else: 
      btsWts = np.ones(Xtheta.shape[0] - 1)
    objective = lambda beta: vPi(beta, policyProbs, eps, M, A, R, Xbeta, Xtheta, Mu, btsWts, refDist=refDist)
    if wStart is None:       
      wStart = np.zeros(nPi)
    betaOpt = VLopt(objective, x0=wStart)
    #print('Optimal theta: {}'.format(thetaPiMulti(betaOpt, policyProbs, A, R, Xtheta, Xbeta, Mu, gamma, eps)))
    return betaOpt
      
      
  
  
  

  
  
  