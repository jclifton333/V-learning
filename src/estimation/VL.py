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
import pdb

def compute_EE_sums_VL(beta, policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, bts):
  '''
  Computes sums used in solving the V-function estimating equation.  
  
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
  btsWts : 1d array of weights for terms in V-function estimating equation 

  Returns
  -------
  sumM : sum_t (importance weight)_t * outer(fV_t, gamma * fV_tp1 - fV_t) (nV x nV-size array) 
  sumRS : sum_t (importance weight)_t * r_t * fV_t ) (nV-size array)
  '''
  nA, nPi = A_list[0].shape[1], F_Pi_list[0].shape[1]
  beta = beta.reshape(nA, nPi)
  nV = F_V_list[0].shape[1]
  sumRS = np.zeros(nV) 
  sumM = np.zeros((nV,nV))
  for t in range(len(A_list)):
    episode_length = A_list[t].shape[0]
    if bts: 
      btsWts = np.random.exponential(size=episode_length) 
    else:
      btsWts = np.ones(episode_length)
    w = np.array([btsWts[i] * float(policyProbs(A_list[t][i,:], F_Pi_list[t][i,:], beta, eps)) / Mu_list[t][i] for i in range(episode_length)])
    sumRS += np.sum(np.multiply(F_V_list[t][:-1,:], np.multiply(w, R_list[t]).reshape(episode_length,1)), axis=0)
    sumM += np.sum(np.multiply(M_list[t], w.reshape(episode_length, 1, 1)), axis=0)
  return sumM, sumRS

def thetaPiVL(beta, policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, bts, thetaTilde):  
  '''
  Estimates theta associated with policy indexed by beta.    
  
  Parameters
  ----------
  beta : policy parameters (1d array)
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M : array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (3d array of size T x nV x nV)
  A : (T x nA)-size array of onehot action encodings
  R : array of rewards (1d array)
  F_Pi : Policy features at each timestep (2d array of size T x nPi)
  F_V : V-function features at each timestep (2d array of size T x nV)
  Mu : Probabilities of observed actions under policies mu_t (1d array of size T)
  btsWts : 1d array of weights for terms in V-function estimating equation 
  thetaTilde : 1d vector to shrink towards in V-function estimation 

  Returns
  -------
  Estimate of theta 
  '''
  sumM, sumRS = compute_EE_sums_VL(beta, policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, bts)
  nV = len(sumRS)
  LU = la.lu_factor(sumM + 0.01*np.eye(nV)) 
  return la.lu_solve(LU, thetaTilde + sumRS)
  
def thetaPiMulti(beta, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts, thetaTilde):
  '''
  Estimates theta associated with policy indexed by beta, where there are multiple
  (nRep) replicates of a trajectory. 
  
  beta : policy parameters (1d array)
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M : array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (4d array of size nRep x T x nV x nV)
  A : (nRep x T x nA)-size array of onehot action encodings
  R : array of rewards (nRep x T-size array)
  F_Pi : Policy features at each timestep (3d array of size nRep x T x nPi)
  F_V : V-function features at each timestep (3d array of size nRep x T x nV)
  Mu : Probabilities of observed actions under policies mu_t (2d array of size nRep x T)
  btsWts : (nRep x T)-size array of weights for terms in V-function estimating equation 
  thetaTilde : 1d vector to shrink towards in V-function estimation 

  Returns
  -------
  Estimate of theta 
  '''
  nRep = A.shape[0]
  nV = F_V.shape[2]
  sumRS = np.zeros(nV) 
  sumM = np.zeros((nV, nV))
  for rep in nRep: 
    sumM_rep, sumRS_rep = compute_EE_sums_VL(beta, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts[rep,:])
    sumM += sumM_rep 
    sumRS += sumRS_rep 
  LU = la.lu_factor(sumM + 0.1*np.eye(nV)) 
  return la.lu_solve(LU, thetaTilde + sumRS)
  
def vPi(beta, policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, bts, thetaTilde, refDist=None):
  '''
  Returns estimated value of policy indexed by beta.  
  
  Parameters
  ----------
  beta : policy parameters, missing first action; (nA-1 x nS)-array
  policyProbs : function returning probability of observed action at given state,
                under policy with parameter beta
  eps : epsilon used in epsilon-greedy
  M : array of matrices outer(psi_t, psi_t) - gamma*outer(psi_t, psi_tp1) (3d array of size T x nV x nV)
  A : array of actions (1d or 2d array)
  R : array of rewards (1d array)
  F_Pi : Policy features at each timestep (2d array of size T x nPi)
  F_V : V-function features at each timestep (2d array of size T x nV)
  Mu : Probabilities of observed actions under policies mu_t (1d array of size T)
  btsWts : 1d array of weights for terms in V-function estimating equation 
  thetaTilde : 1d vector to shrink towards in V-function estimation 
  refDist : Reference distribution for estimating value (2d array with v-function features as rows).
           If None, uses F_V as reference distribution.  
  
  Returns
  -------
  (Negative) estimated value of policy indexed by beta wrt refDist.  
  '''
  beta = np.append(np.zeros(F_Pi_list[0].shape[1]), beta)  #add row of zeros corresponding to first action 
  theta = thetaPiVL(beta, policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, bts, thetaTilde)
  if refDist is None:
    return -np.mean(np.dot(np.vstack(F_V_list), theta))
  else: 
    return -np.mean(np.dot(refDist, theta))

      
def betaOptVL(policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, wStart=None, refDist=None, bts=True, randomShrink=True, initializer=None):
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
  randomShrink: boolean for shrinking towards randomly generated vector in v-function estimation
  bts: boolean for using (exponential) bootstrap Thompson sampling
  initializer: value in [None, 'basinhop', 'multistart'] for optimizer initialization method
  
  Returns
  -------
  Dictionary {'betaHat':estimate of beta, 'thetaHat':estimate of theta, 'objective':objective function (of policy parameters)}
  '''
  nPi = F_Pi_list[0].shape[1]
  nV = F_V_list[0].shape[1]
  nA = A_list[0].shape[1]
  if np.sum([F_Pi.shape[0] for F_Pi in F_Pi_list]) < nV:     
    objective = lambda x: None 
    return {'betaHat':np.random.normal(size=(nA, nPi)), 'thetaHat':np.zeros(nV), 'objective':objective}
  else:    
    if randomShrink: 
      thetaTilde = np.random.normal(size=nV)
    else:
      thetaTilde = np.zeros(nV)
    objective = lambda beta: vPi(beta, policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, bts, thetaTilde, refDist=refDist)
    if wStart is None:       
      wStart = np.random.normal(scale=1000, size=(nA-1, nPi)) #Leave out first action, since this is all zeros 
    betaOpt = VLopt(objective, x0=wStart, initializer=initializer)
    betaOpt = np.vstack((np.zeros(betaOpt.shape[1]), betaOpt))
    thetaOpt = thetaPiVL(betaOpt.ravel(), policyProbs, eps, M_list, A_list, R_list, F_Pi_list, F_V_list, Mu_list, bts, thetaTilde)
    return {'betaHat':betaOpt, 'thetaHat':thetaOpt, 'objective':objective}
  
  

  
  
  