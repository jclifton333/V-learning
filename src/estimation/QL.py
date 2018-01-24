import numpy as np
import scipy.linalg as la 

def compute_EE_QL(beta, eps, M, A, R, F_Pi, F_V, btsWts):
  '''
  Computes the sum fo rthe estimating equation in QL.  
  '''
   

def compute_EE_sums_QL(beta, policyProbs, eps, M, A, R, F_Pi, F_V, Mu, btsWts):
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
  nA, nPi = A.shape[1], F_Pi.shape[1]
  beta = beta.reshape(nA, nPi)
  T, p = F_V.shape[0] - 1, F_V.shape[1]
  w = np.array([btsWts[i] * float(policyProbs(A[i,:], F_Pi[i,:], beta, eps=eps)) / Mu[i] for i in range(T)])
  sumRS = np.sum(np.multiply(F_V[:-1,:], np.multiply(w, R).reshape(T,1)), axis=0)
  sumM  = np.sum(np.multiply(M, w.reshape(T, 1, 1)), axis=0)
  return sumM, sumRS
