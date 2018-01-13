import sys 
sys.path.append('../utils')
from policy_utils import policyProbs 
import pdb

import numpy as np

def log_policy_gradient(a, s_vec, beta):
  '''
  Compute the gradient with respect to beta of
  log pi_beta(a | s), where pi_beta is the softmax policy 
  with parameters beta.
  
  :param a: onehot encoding for action 
  :param s_vec: state (policy feature) vector 
  :param beta: NUM_ACTION x NUM_POLICY_FEATURE - size array of policy parameters 
  :return gradient: NUM_ACTION x NUM_POLICY_FEATURE - size array for log policy gradient 
  '''
  assert(len(s_vec) == beta.shape[1], "State vector and policy parameter rows not of equal length.")
  gradient = np.zeros(beta.shape)
  action_prob = policyProbs(a, s_vec, beta, eps = 0.0) 
  action_index = np.flatnonzero(a)[0]
  gradient[action_index, :] = s_vec * (1 - action_prob)
  return gradient
  
def advantage_estimate(r, f_V, f_Vp1, thetaHat): 
  return r + np.dot(thetaHat, f_Vp1) - np.dot(thetaHat, f_V)
  
def total_policy_gradient(beta, A, R, F_Pi, F_V, thetaHat): 
  T, p = F_V.shape[0] - 1, F_Pi.shape[1]
  gradient = np.zeros(beta.shape) 
  for i in range(T): 
    grad = log_policy_gradient(A[i, :], F_Pi[i,:], beta) 
    advantage = advantage_estimate(R[i], F_V[i,:], F_V[i+1,:], thetaHat)
    gradient += grad*advantage
  return gradient    

  
  