import sys 
sys.path.append('../utils')
from policy_utils import policyProbs 
import pdb

import numpy as np

def log_policy_gradient(a, s_vec, beta, epsilon):
  '''
  Compute the gradient with respect to beta of
  log pi_beta(a | s), where pi_beta is the softmax policy 
  with parameters beta.
  
  :param a: onehot encoding for action 
  :param s_vec: state (policy feature) vector 
  :param beta: NUM_ACTION x NUM_POLICY_FEATURE - size array of policy parameters 
  :return gradient: NUM_ACTION x NUM_POLICY_FEATURE - size array for log policy gradient 
  '''
  gradient = np.zeros(beta.shape)
  action_prob = policyProbs(a, s_vec, beta, eps = epsilon) 
  action_index = np.flatnonzero(a)[0]
  gradient[action_index, :] = s_vec * (1 - action_prob)
  return gradient
  
def advantage_estimate(gamma, r, f_V, f_Vp1, thetaHat): 
  '''
  Computes the advantage estimate. 
  
  :param gamma: discount 
  :param r: reward
  :param f_V: v-function state feature 
  :param f_Vp1: v-function state feature at next timestep 
  :param thetaHat: v-function parameters
  :return: advantage estimate (scalar) 
  '''
  return r + gamma * np.dot(thetaHat, f_Vp1) - np.dot(thetaHat, f_V)
  
def total_policy_gradient(beta, A, R, F_Pi, F_V, thetaHat, gamma, epsilon): 
  '''
  Compute the sum of log policy gradients for a given dataset. 
  
  :param beta: NUM_ACTION x nPi - size softmax policy parameter value 
  :param A: NUM_TIMESTEPS x NUM_ACTION - size array of actions 
  :param R: NUM_TIMESTEPS - size array of rewards
  :param F_Pi: NUM_TIMESTEPS + 1 x nPi - size array of state policy function features 
  :param F_V: NUM_TIMESTEPS + 1 x nV - size array of state v-function features 
  :param thetaHat: nV - size array of estimate v-function parameters  
  :param gamma: discount 
  :return gradient: NUM_ACTION x nPi - size policy gradient
  '''

  T, p = F_V.shape[0] - 1, F_Pi.shape[1]
  gradient = np.zeros(beta.shape) 
  for i in range(T): 
    grad = log_policy_gradient(A[i, :], F_Pi[i,:], beta, epsilon) 
    advantage = advantage_estimate(gamma, R[i], F_V[i,:], F_V[i+1,:], thetaHat)
    gradient += grad*advantage
  return gradient    

def total_policy_gradient_multi(beta, A, R, F_Pi, F_V, thetaHat, gamma, epsilon):
  '''
  Computes policy gradient for nRep replicated trajectories. 
  
  :param beta: NUM_ACTION x nPi - size softmax policy parameter value 
  :param A: nRep x NUM_TIMESTEPS x NUM_ACTION - size array of actions 
  :param R: nRep x NUM_TIMESTEPS - size array of rewards
  :param F_Pi: nRep x NUM_TIMESTEPS + 1 x nPi - size array of state policy function features 
  :param F_V: nRep x NUM_TIMESTEPS + 1 x nV - size array of state v-function features 
  :param thetaHat: nV - size array of estimate v-function parameters  
  :param gamma: discount 
  :return gradient: NUM_ACTION x nPi - size policy gradient
  '''
  gradient = np.zeros(beta.shape)
  nRep = A.shape[0]
  for rep in range(nRep):
    gradient += total_policy_gradient(beta, A[rep,:], R[rep,:], F_Pi[rep,:], F_V[rep,:], thetaHat, gamma, epsilon)
  return gradient / nRep
  
  
  