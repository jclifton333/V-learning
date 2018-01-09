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
import scipy.linalg as la 

def thetaPi(beta, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts):  
  '''
  Estimates theta associated with policy indexed by beta.    
  
  Parameters
  ----------
  beta : policy parameters (1d array)
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M : array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (3d array of size T x nV x nV)
  A : array of actions (1d or 2d array; if 2d actions must be rows)
  R : array of rewards (1d array)
  F_Pi : Policy features at each timestep (2d array of size T x nPi)
  F_V : V-function features at each timestep (2d array of size T x nV)
  Mu : Probabilities of observed actions under policies mu_t (1d array of size T)
  
  Returns
  -------
  Estimate of theta 
  '''
  binary = (len(A.shape) == 1) 
  nA, nPi = A.shape[1], F_Pi.shape[1]
  if not binary: 
    beta = beta.reshape(nA, nPi)
  T, p = F_V.shape[0] - 1, F_V.shape[1]
  if binary == 1:
    w = np.array([btsWts[i] * float(policyProbs(A[i], F_Pi[i,:], beta, eps=eps)) / Mu[i] for i in range(T)])
  else:
    w = np.array([btsWts[i] * float(policyProbs(A[i,:], F_Pi[i,:], beta, eps=eps)) / Mu[i] for i in range(T)])
  sumRS = np.sum(np.multiply(F_V[:-1,:], np.multiply(w, R).reshape(T,1)), axis=0)
  sumM  = np.sum(np.multiply(M, w.reshape(T, 1, 1)), axis=0)
  LU = la.lu_factor(sumM + 0.01*np.eye(p)) 
  return la.lu_solve(LU, sumRS)
  
def vPi(beta, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts, refDist=None):
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
  F_Pi : Policy features at each timestep (2d array of size T x nPi)
  F_V : V-function features at each timestep (2d array of size T x nV)
  Mu : Probabilities of observed actions under policies mu_t (1d array of size T)
  refDist : Reference distribution for estimating value (2d array with v-function features as rows).
           If None, uses F_V as reference distribution.  
  
  Returns
  -------
  (Negative) estimated value of policy indexed by beta wrt refDist.  
  '''
  
  theta = thetaPi(beta, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts)
  if refDist is None:
    return -np.mean(np.dot(F_V, theta))
  else: 
    return -np.mean(np.dot(refDist, theta))

      
def betaOpt(policyProbs, eps, M, A, R, F_Pi, F_V, Mu, wStart=None, refDist=None, bts=True, initializer=None):
  '''
  Optimizes policy value over class of softmax policies indexed by beta. 
  
  Parameters
  ----------
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M: array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (3d array of size T x nV x nV)
  A: array of actions (1d or 2d array)
  R: array of rewards (1d array)
  F_Pi: Policy features at each timestep (2d array of size T x nPi)
  F_V: V-function features at each timestep (2d array of size T x nV)
  Mu: Probabilities of observed actions under policies mu_t (1d array of size T)
  wStart : Warm start for beta optimization (1d array). 
           If None, initializes to all 0s.  
  refDist: Reference distribution for estimating value (2d array with v-function features as rows).
           If None, use F_V as reference distribution.  
  bts: boolean for using (exponential) bootstrap Thompson sampling
  initializer: value in [None, 'basinhop', 'multistart'] for optimizer initialization method
  
  Returns
  -------
  Dictionary {'betaHat':estimate of beta, 'thetaHat':estimate of theta, 'objective':objective function (of policy parameters)}
  '''
  nPi = F_Pi.shape[1]
  nV = F_V.shape[1]
  nA = A.shape[1]
  if F_V.shape[0] < nV: 
    objective = lambda x: None 
    return {'betaHat':np.zeros((nA, nPi)), 'thetaHat':np.zeros(nV), 'objective':objective}
  else:    
    if bts: 
      btsWts = np.random.exponential(size = F_V.shape[0] - 1) 
    else: 
      btsWts = np.ones(F_V.shape[0] - 1)
    objective = lambda beta: vPi(beta, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts, refDist=refDist)
    if wStart is None:       
      wStart = np.random.normal(scale=1000, size=(nA, nPi))
    betaOpt = VLopt(objective, x0=wStart, initializer=initializer)
    thetaOpt = thetaPi(betaOpt, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts)
    return {'betaHat':betaOpt, 'thetaHat':thetaOpt, 'objective':objective}
  
  

  
  
  