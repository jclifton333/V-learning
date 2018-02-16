import numpy as np
from scipy.optimize import minimize 
import pdb
from VLopt import VLopt

def QL_TD_error(beta, gamma, A_list, R_list, F_Pi_list, btsWts, state_action_features_list):
  beta = beta.reshape(A_list[0].shape[1], F_Pi_list[0].shape[1])
  TD = np.zeros(0) 
  for t in range(len(A_list)):
    if A_list[t].shape[0] > 1:
      F_Pi = F_Pi_list[t] 
      R = R_list[t] 
      state_action_features = state_action_features_list[t]    
      Qmax = np.max(np.dot(F_Pi[1:,:], beta.T), axis=1)
      Q = np.dot(state_action_features, beta.ravel())
      TD_t = R + gamma*Qmax - Q 
      if btsWts is not None:
        TD_t = np.multiply(TD_t, btsWts[t])
      TD = np.append(TD, TD_t)
  return TD

def QL_objective(beta, gamma, A_list, R_list, F_Pi_list, btsWts, state_action_features_list):
  TD = QL_TD_error(beta, gamma, A_list, R_list, F_Pi_list, btsWts, state_action_features_list)
  return np.dot(TD, TD)

def estimating_equation_QL(beta, gamma, A, R, F_Pi, btsWts, state_action_features):
  '''
  Computes the sum for the estimating equation in QL.  
  '''
  TD = QL_TD_error(beta, gamma, A, R, F_Pi, btsWts, state_action_features)
  try:
    EEarr = np.multiply(np.vstack(state_action_features), TD.reshape(len(TD), 1))  
  except:
    pdb.set_trace()
  EEsum = np.sum(EEarr, axis=0) + 0.1*beta.ravel()
  return EEsum
  
def squared_EE_norm_QL(beta, gamma, A, R, F_Pi, btsWts, state_action_features):
  EEsum = estimating_equation_QL(beta, gamma, A, R, F_Pi, btsWts, state_action_features)
  return np.dot(EEsum, EEsum)
  
def betaOptQL(gamma, A_list, R_list, F_Pi_list, wStart=None, bts=True):
  '''
  Estimates optimal Q-function parameters beta.
  '''
  nPi = F_Pi_list[0].shape[1]
  nA = A_list[0].shape[1] 
  nV = nA * nPi 
  totalSteps = np.sum([F_Pi.shape[0] for F_Pi in F_Pi_list])
  if totalSteps < nV: 
    objective = lambda x: None 
    return {'betaHat':np.random.normal(size=(nA, nPi)), 'thetaHat':None, 'objective':objective}
  else:    
    if bts: 
      btsWts = [np.random.exponential(size=A.shape[0]) for A in A_list]
    else:
      btsWts = None
    state_action_features_list = [np.array([np.kron(A_list[t][i,:], F_Pi_list[t][i,:]) for i in range(A_list[t].shape[0])]) for t in range(len(A_list))]
    objective = lambda beta: QL_objective(beta, gamma, A_list, R_list, F_Pi_list, btsWts, state_action_features_list)
    if wStart is None:       
      wStart = np.random.normal(size=(nA, nPi)) 
    betaOpt = VLopt(objective, wStart, initializer='multistart')
    return {'betaHat':betaOpt, 'thetaHat':None, 'objective':objective}
  