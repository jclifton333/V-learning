import numpy as np
from VLenvironment import Cartpole 
from VL import betaOpt

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
  env.reset() 
  done = False 
  
    
  