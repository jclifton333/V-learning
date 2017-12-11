import numpy as np
from VLenvironment import Cartpole 
from VL import betaOpt
from policyUtils import piBin, policyProbsBin
import pdb

#Simulation settings 
GAMMA = 0.9 
EPSILON = 0.05 
SIGMA_SQ = 1 
GRIDPOINTS = 5 
FEATURE_CHOICE = 'gRBF' 
NUM_EP = 200 
DEFAULT_REWARD = False
V_ARGS = {'featureChoice':FEATURE_CHOICE, 'sigmaSq':SIGMA_SQ, 'gridpoints':GRIDPOINTS}

#Initialize environment 
env = Cartpole(gamma = GAMMA, epsilon = EPSILON, defaultReward = DEFAULT_REWARD, vFeatureArgs = V_ARGS)
bHat = np.zeros(2) 
for ep in range(NUM_EP): 
  fPi = env.reset() 
  done = False 
  score = 0 
  while not done: 
    score += 1
    aProb = piBin(fPi, bHat) 
    a = np.random.random() < aProb * (1 - EPSILON) 
    fV, F_V, F_Pi, A, R, Mu, M, done = env.step(a, bHat)
    if not done: 
      bHat = betaOpt(policyProbsBin, EPSILON, M, A, R, F_Pi, F_V, Mu)
  print('Episode {} Score: {}'.format(ep, score))
    
    
  