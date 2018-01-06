import sys 
sys.path.append('src')

import numpy as np
import pdb
import argparse 
import multiprocessing as mp
from VLenvironment import Cartpole, FlappyBirdEnv, randomFiniteMDP
from VL import betaOpt
from policyUtils import piBin, policyProbsBin
from functools import partial 
from utils import str2bool, intOrNone, strOrNone
import time
import pickle as pkl

'''
Global simulation variables. 
'''
VALID_ENVIRONMENT_NAMES = ['FlappyBird', 'randomFiniteMDP', 'Cartpole'] 

#Cartpole
DEFAULT_REWARD = False

#randomFiniteMDP
NUM_FINITE_STATE = 3 #Number of states if randomFiniteMDP is chosen
MAX_T_FINITE = 50 #Max number of timesteps per episode if randomFiniteMDP is chosen

#Flappy Bird
DISPLAY_SCREEN = False

class data(object):
  '''
  For storing and saving data from simulations.  
  '''

  def __init__(self, environment, method, label, optimalPolicy='Not specified'):
    '''
    :param environment: a VLenv object (Cartpole, randomFiniteMDP, or FlappyBirdEnv)
    :param method: string describing hyperparameters, e.g. 'eps-0.05-gamma-0.9'
    :param label: arbitrary label for writing to file 
    :param optimalPolicy: array for optimal policy parameters, or 'Not specified'
    '''
    self.environment = environment
    self.method = method
    self.label = label
    self.episode = []
    self.score = []
    self.beta_hat = []
    self.theta_hat = []
    self.optimalPolicy = optimalPolicy
    self.name = '{}-{}-{}.p'.format(self.environment, self.method, self.label)
    
  def update(self, episode, score, beta_hat, theta_hat):
    '''
    Update data.
    
    :param episode: integer for episode
    :param score: integer for score
    :param beta_hat: array for policy parameter estimate
    :param theta_hat: array for v-function parameter estimate
    '''
    self.episode.append(episode)
    self.score.append(score)
    self.beta_hat.append(beta_hat)
    self.theta_hat.append(theta_hat)
    
  def write(self):
    '''
    Dump current data to pickle. 
    '''
    data_dict = {'label':self.label, 'method':self.method, 'episode':self.episode, 'score':self.score,
                 'beta_hat':self.beta_hat, 'theta_hat':self.theta_hat, 'optimalPolicy':self.optimalPolicy}
    filename = 'results/{}/'.format(self.environment) + self.name 
    pkl.dump(data_dict, open(filename, 'wb')) 
    
def getEnvironment(envName, gamma, epsilon, vFeatureArgs, piFeatureArgs):
  '''
  Returns a VLenv object corresponding to the given parameters and globals.
  '''
  if envName == 'Cartpole':
    return Cartpole(gamma, epsilon, DEFAULT_REWARD, vFeatureArgs, piFeatureArgs)
  elif envName == 'randomFiniteMDP':
    return randomFiniteMDP(NUM_FINITE_STATE, MAX_T_FINITE, gamma, epsilon, vFeatureArgs, piFeatureArgs)
  elif envName == 'FlappyBird':
    return FlappyBirdEnv(gamma, epsilon, vFeatureArgs, piFeatureArgs, displayScreen = DISPLAY_SCREEN)
  else: 
    raise ValueError('Incorrect environment name.  Choose name in {}.'.format(VALID_ENVIRONMENT_NAMES))
    
def simulate(bts, epsilon, initializer, label, envName, gamma, vArgs, piArgs, nEp, fixUpTo, write = False):
  '''
  Runs simulation with V-learning for an environment with a binary action space.
  
  Parameters 
  ----------
  bts : boolean for exponential Thompson sampling
  epsilon : exploration rate 
  initializer: value in [None, 'basinhop', 'multistart'] for optimizer initialization method
  label : label for filename, if write is True
  envName : string in VALID_ENVIRONMENT_NAMES specifying environment to simulate
  gamma : discount factor
  vArgs : dictionary {'featureChoice', 'sigmaSq', 'gridpoints'} for v-function arguments 
  piArgs : '' for policy arguments   
  nEp: number of episodes 
  fixUpTo : if integer is given use first _fixUpTo_ observations as reference distribution; 
            otherwise, always use entire observation history   
  write : boolean for writing results to file 
  '''

  #Initialize  
  env = getEnvironment(envName, gamma, epsilon, vArgs, piArgs)
  betaHat = np.zeros(env.nPi)
  totalStepsCounter = 0

  #Data collection and writing settings 
  if write: 
    method = 'eps-{}-bts-{}-fix-{}-initializer-{}'.format(epsilon, bts, fixUpTo, initializer)
    if envName == 'randomFiniteMDP':
      optimalPolicy = env.optimalPolicy
    else:
      optimalPolicy = 'Not specified'
    save_data = data(envName, method, label, optimalPolicy)
    
  #Run sim
  betaHat = np.array([991.32845, 40.486286])  #This is the optimal beta for cartpole 
  for ep in range(nEp): 
    fPi = env.reset() 
    done = False 
    score = 0 
    while not done: 
      totalStepsCounter += 1
      aProb = piBin(fPi, betaHat) 
      a = np.random.random() < aProb * (1 - epsilon) 
      fPi, F_V, F_Pi, A, R, Mu, M, done, reward = env.step(a, betaHat)
      if not done:
        score += 1 
        if fixUpTo is not None: 
          refDist = F_V[:fixUpTo,:]
        else:
          refDist = F_V 
        res = betaOpt(policyProbsBin, epsilon, M, A, R, F_Pi, F_V, Mu, bts = bts, refDist = refDist, initializer = initializer)
        betaHat = res['betaHat']
    print('Episode {} Score: {}'.format(ep, score))
    if envName == 'randomFiniteMDP': #Display policy and value information for finite MDP
      env.evaluatePolicies(betaHat)
    if write: 
      save_data.update(ep, score, betaHat, tHat)
      save_data.write()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--envName', type=str, help="String for environment name.", choices=VALID_ENVIRONMENT_NAMES)
  parser.add_argument('--gamma', type=float, help="Discount factor.")
  parser.add_argument('--epsilon', type=float, help="Exploration rate.")
  parser.add_argument('--bts', type=str2bool, help="Boolean for using (exponential) BTS.")
  parser.add_argument('--initializer', type=strOrNone, choices=['None', 'basinhop', 'multistart'], help="String or None for optimization initialization method.")
  parser.add_argument('--sigmaSqV', type=float, help="Gaussian kernel variance for v-function features.")
  parser.add_argument('--gridpointsV', type=int, help="Number of basis function points per dimension for v-function features.")
  parser.add_argument('--featureChoiceV', choices=['gRBF', 'identity', 'intercept'], help="Choice of v-function features.")
  parser.add_argument('--sigmaSqPi', type=float, help="Gaussian kernel variance for policy features.")
  parser.add_argument('--gridpointsPi', type=int, help="Number of basis function points per dimension for policy features.")
  parser.add_argument('--featureChoicePi', choices=['gRBF', 'identity', 'intercept'], help="Choice of policy features.")
  parser.add_argument('--randomSeed', type=int) 
  parser.add_argument('--nEp', type=int, help="Number of episodes per replicate.")
  parser.add_argument('--nRep', type=int, help="Number of replicates.")
  parser.add_argument('--write', type=str2bool, help="Boolean for writing results to file.")
  parser.add_argument('--fixUpTo', type=intOrNone, help="Integer for number of observations to include in reference distribution, or None")
  args = parser.parse_args()
    
  #np.random.seed(args.randomSeed)
  
  vArgs  = {'featureChoice':args.featureChoiceV, 'sigmaSq':args.sigmaSqV, 'gridpoints':args.gridpointsV}
  piArgs = {'featureChoice':args.featureChoicePi, 'sigmaSq':args.sigmaSqPi, 'gridpoints':args.gridpointsPi} 
  
  pool = mp.Pool(args.nRep)

  simulate_partial = partial(simulate, envName = args.envName, gamma = args.gamma, vArgs = vArgs, piArgs = piArgs, 
                             nEp = args.nEp, fixUpTo = args.fixUpTo, write = args.write) 
  argList = [(args.bts, args.epsilon, args.initializer, label) for label in range(args.nRep)]
  pool.starmap(simulate_partial, argList) 
