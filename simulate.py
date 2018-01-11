import sys 
sys.path.append('src')
sys.path.append('src/environments')
sys.path.append('src/utils')
sys.path.append('src/estimation')

from Cartpole import Cartpole 
from Flappy import Flappy 
from FiniteMDP import RandomFiniteMDP, SimpleMDP, Gridworld
from VL import betaOpt
from policy_utils import pi, policyProbs
from functools import partial 
from utils import str2bool, intOrNone, strOrNone
import numpy as np
import argparse 
import multiprocessing as mp
import multiprocessing.pool as pl
import FiniteMDP
import time
import pickle as pkl

'''
Global simulation variables. 
'''
VALID_ENVIRONMENT_NAMES = ['Flappy', 'SimpleMDP', 'Gridworld', 'RandomFiniteMDP', 'Cartpole'] 

#Cartpole
DEFAULT_REWARD = False

#FiniteMDPs 
NUM_RANDOM_FINITE_STATE = 3 #Number of states if randomFiniteMDP is chosen
NUM_RANDOM_FINITE_ACTION = 3
MAX_T_FINITE = 50 #Max number of timesteps per episode if randomFiniteMDP is chosen

#Flappy Bird
DISPLAY_SCREEN = False

class data(object):
  '''
  For storing and saving data from simulations.  
  '''
  def __init__(self, environment, fixUpTo, initializer, label, epsilon, bts, write = False):
    '''
    :param environment: a string naming a VLenv object 
    :param fixUpTo: integer or None for number of obs to include in reference distribution 
    :param initializer: string in ['multistart', 'basinhop'], or None 
    :param label: arbitrary label for writing to file 
    :param write: boolean for writing data to pickle at each update 
    '''
    self.environment = environment
    self.epsilon = epsilon 
    self.bts = bts 
    self.initializer = initializer 
    self.description = 'eps-{}-bts-{}-fix-{}-initializer-{}'.format(epsilon, bts, fixUpTo, initializer)
    self.label = label
    self.episode = []
    self.score = []
    self.beta_hat = []
    self.theta_hat = []
    self.name = '{}-{}-{}.p'.format(self.environment, self.description, self.label)      
    self.write = write
    
  def update(self, episode, score, beta_hat, theta_hat):
    '''
    Update data, and write to pickle if self.write == True. 
    
    :param episode: integer for episode
    :param score: integer for score
    :param beta_hat: array for policy parameter estimate
    :param theta_hat: array for v-function parameter estimate
    '''
    self.episode.append(episode)
    self.score.append(score)
    self.beta_hat.append(beta_hat)
    self.theta_hat.append(theta_hat)
    
    if self.write: 
      self.write() 
    
  def write(self):
    '''
    Dump current data to pickle. 
    '''
    data_dict = {'label':self.label, 'description':self.description, 'episode':self.episode, 'score':self.score,
                 'beta_hat':self.beta_hat, 'theta_hat':self.theta_hat,
                 'epsilon':self.epsilon, 'bts':self.bts, 'initializer':self.initializer}
    filename = 'results/{}/'.format(self.environment) + self.name 
    pkl.dump(data_dict, open(filename, 'wb')) 
    
def getEnvironment(envName, gamma, epsilon, fixUpTo):
  '''
  Returns a VLenv object corresponding to the given parameters and globals.
  '''
  if envName == 'Cartpole':
    return Cartpole(gamma, epsilon, DEFAULT_REWARD, fixUpTo = fixUpTo)
  elif envName == 'Flappy':
    return Flappy(gamma, epsilon, displayScreen = DISPLAY_SCREEN, fixUpTo = fixUpTo)
  elif envName == 'RandomFiniteMDP':
    return RandomFiniteMDP(MAX_T_FINITE,  nA = NUM_RANDOM_FINITE_ACTION, 
                                     nS = NUM_RANDOM_FINITE_STATE, gamma = gamma, epsilon = epsilon, fixUpTo = fixUpTo)
  elif envName == 'SimpleMDP':
    return SimpleMDP(MAX_T_FINITE, gamma, epsilon, fixUpTo = fixUpTo) 
  elif envName == 'Gridworld':
    return Gridworld(MAX_T_FINITE, gamma, epsilon, fixUpTo = fixUpTo) 
  else: 
    raise ValueError('Incorrect environment name.  Choose name in {}.'.format(VALID_ENVIRONMENT_NAMES))
    
def simulate(bts, epsilon, initializer, label, envName, gamma, nEp, fixUpTo, write = False):
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
  nEp: number of episodes 
  fixUpTo : if integer is given use first _fixUpTo_ observations as reference distribution; 
            otherwise, always use entire observation history   
  write : boolean for writing results to file 
  '''

  #Initialize  
  #TODO: return betaHat in env.reset; make totalStepsCounter an env attribute 
  env = getEnvironment(envName, gamma, epsilon, fixUpTo)
  betaHat = np.zeros((env.NUM_ACTION, env.nPi))
  save_data = data(envName, fixUpTo, initializer, label, epsilon, bts, write = write)
      
  #Run sim
  for ep in range(nEp): 
    fPi = env.reset() 
    done = False 
    score = 0 
    t0 = time.time()
    while not done: 
      a = env._get_action(fPi, betaHat)
      fPi, F_V, F_Pi, A, R, Mu, M, refDist, done, reward = env.step(a, betaHat)
      if not done:
        if env.s in env.PATH: 
          res = env.betaOpt(policyProbs, epsilon, M, A, R, F_Pi, F_V, Mu, bts = bts, wStart = betaHat[1:,:], refDist = refDist, initializer = initializer)
          betaHat, tHat = res['betaHat'], res['thetaHat']
    t1 = time.time()
    print('Episode {} Score: {} BTS: {}'.format(ep, env.episodeSteps, bts))
    #if envName in ['RandomFiniteMDP', 'Gridworld', 'SimpleMDP']: #Display policy and value information for finite MDP
    #  env.evaluatePolicies(betaHat)
      
    save_data.update(ep, score, betaHat, tHat)      
  return 
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--envName', type=str, help="String for environment name.", choices=VALID_ENVIRONMENT_NAMES)
  parser.add_argument('--gamma', type=float, help="Discount factor.")
  parser.add_argument('--epsilon', type=float, help="Exploration rate.")
  parser.add_argument('--bts', type=str2bool, help="Boolean for using (exponential) BTS.")
  parser.add_argument('--initializer', type=strOrNone, choices=[None, 'basinhop', 'multistart'], help="String or None for optimization initialization method.")
  parser.add_argument('--randomSeed', type=int) 
  parser.add_argument('--nEp', type=int, help="Number of episodes per replicate.")
  parser.add_argument('--nRep', type=int, help="Number of replicates.")
  parser.add_argument('--write', type=str2bool, help="Boolean for writing results to file.")
  parser.add_argument('--fixUpTo', type=intOrNone, help="Integer for number of observations to include in reference distribution, or None")
  args = parser.parse_args()
    
  np.random.seed(args.randomSeed)
  
  pool = pl.ThreadPool(args.nRep)
  simulate_partial = partial(simulate, envName = args.envName, gamma = args.gamma, nEp = args.nEp, fixUpTo = args.fixUpTo, write = args.write) 
  argList = [(args.bts, args.epsilon, args.initializer, label) for label in range(args.nRep)]
  pool.starmap(simulate_partial, argList) 
  pool.close()
  pool.join()
