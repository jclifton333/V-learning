import numpy as np
import pdb
import argparse 
import multiprocessing as mp
from VLenvironment import Cartpole, FlappyBirdEnv
from VL import betaOpt
from policyUtils import piBin, policyProbsBin
from functools import partial 
from utils import str2bool, intOrNone
import time
import pickle as pkl

class data(object):
  def __init__(self, method, label):
    '''
    :param method: string describing hyperparameters, e.g. 'eps-0.05-gamma-0.9'
    :param label: arbitrary label for writing to file 
    '''
    self.method = method
    self.label = label
    self.episode = []
    self.score = []
    self.beta_hat = []
    self.theta_hat = []
    self.name = '{}-{}.p'.format(self.method, self.label)
    
  def update(self, episode, score, beta_hat, theta_hat):
    '''
    Update data.
    
    :param episode: integer for episode
    :param score: integer for score
    :param beta_hat: array for policy parameter estimate
    :param theta_hat: array for v-function parameter estimate
    '''
    self.episode.append(episode)
    self.score.append(beta_hat)
    self.beta_hat.append(beta_hat)
    self.theta_hat.append(theta_hat)
    
  def write(self):
    '''
    Dump current data to pickle. 
    '''
    data_dict = {'label':self.label, 'method':self.method, 'episode':self.episode, 'score':self.score,
                 'beta_hat':self.beta_hat, 'theta_hat':self.theta_hat}
    pkl.dump(data_dict, open(self.name, 'wb')) 

def cartpoleVL(bts, epsilon, label, gamma, defaultReward, vArgs, piArgs, nEp, fixUpTo, LU, write = False):
  '''
  Runs cartpole simulation with V-learning.
  
  Parameters 
  ----------
  gamma : discount factor 
  epsilon : exploration rate 
  defaultReward : use default reward (use shaped reward if False)
  vArgs : dictionary {'featureChoice', 'sigmaSq', 'gridpoints'} for v-function arguments 
  piArgs : '' for policy arguments 
  bts : boolean for exponential Thompson sampling 
  write : boolean for writing results to file 
  label : label for filename, if write is True
  fixUpTo : if integer is given use first _fixUpTo_ observations as reference distribution; 
            otherwise, always use entire observation history 
  ''' 
 
  if write: 
    method = 'eps-{}-bts-{}-fix-{}'.format(epsilon, bts, fixUpTo)
    save_data = data(method, label)
    
  #Initialize  
  env = Cartpole(gamma = gamma, epsilon = epsilon, defaultReward = defaultReward, vFeatureArgs = vArgs, piFeatureArgs = piArgs)
  bHat = np.zeros(env.nPi)

  #Run sim
  totalStepsCounter = 0 
  for ep in range(nEp): 
    t0 = time.time() 
    fPi = env.reset() 
    done = False 
    score = 0 
    while not done: 
      totalStepsCounter += 1
      score += 1
      aProb = piBin(fPi, bHat) 
      a = np.random.random() < aProb * (1 - epsilon) 
      fPi, F_V, F_Pi, A, R, Mu, M, done = env.step(a, bHat)
      if not done:
        if fixUpTo is not None: 
          refDist = F_V[:fixUpTo,:]
        else:
          refDist = F_V 
        res = betaOpt(policyProbsBin, epsilon, M, A, R, F_Pi, F_V, Mu, LU, bts = bts, refDist = refDist)
        bHat, tHat = res['bHat'], res['tHat']
    t1 = time.time() 
    print('Episode {} LU: {} Time per step: {} Score: {}'.format(ep, LU, (t1-t0)/score, score))
    if write: 
      save_data.update(ep, score, bHat, tHat)
      save_data.write()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--gamma', type=float, help="Discount factor.")
  parser.add_argument('--epsilon', type=float, help="Exploration rate.")
  parser.add_argument('--bts', type=str2bool, help="Boolean for using (exponential) BTS.")
  parser.add_argument('--sigmaSqV', type=float, help="Gaussian kernel variance for v-function features.")
  parser.add_argument('--gridpointsV', type=int, help="Number of basis function points per dimension for v-function features.")
  parser.add_argument('--featureChoiceV', choices=['gRBF', 'identity', 'intercept'], help="Choice of v-function features.")
  parser.add_argument('--sigmaSqPi', type=float, help="Gaussian kernel variance for policy features.")
  parser.add_argument('--gridpointsPi', type=int, help="Number of basis function points per dimension for policy features.")
  parser.add_argument('--featureChoicePi', choices=['gRBF', 'identity', 'intercept'], help="Choice of policy features.")
  parser.add_argument('--randomSeed', type=int) 
  parser.add_argument('--defaultReward', type=str2bool, help="Use default reward (shaped reward if False).")
  parser.add_argument('--nEp', type=int, help="Number of episodes per replicate.")
  parser.add_argument('--nRep', type=int, help="Number of replicates.")
  parser.add_argument('--write', type=str2bool, help="Boolean for writing results to file.")
  parser.add_argument('--fixUpTo', type=intOrNone, help="Integer for number of observations to include in reference distribution, or None")
  parser.add_argument('--LU', type=str2bool, help="Boolean for using LU decomp")
  args = parser.parse_args()
    
  np.random.seed(args.randomSeed)
  
  vArgs  = {'featureChoice':args.featureChoiceV, 'sigmaSq':args.sigmaSqV, 'gridpoints':args.gridpointsV}
  piArgs = {'featureChoice':'identity', 'sigmaSq':None, 'gridpoints':None} #Currently IGNORING Pi arguments, using identity 
  
  pool = mp.Pool(args.nRep) 
  cartpoleVL_partial = partial(cartpoleVL, gamma = args.gamma, defaultReward = args.defaultReward, 
                               vArgs = vArgs, piArgs = piArgs, nEp = args.nEp, fixUpTo = args.fixUpTo, LU = args.LU, write = args.write) 
  argList = [(args.bts, args.epsilon, label) for label in range(args.nRep)]
  pool.starmap(cartpoleVL_partial, argList) 