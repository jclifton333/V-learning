import numpy as np
import pdb
import argparse 
import multiprocessing as mp
from VLenvironment import Cartpole, FlappyBirdEnv
from VL import betaOpt
from policyUtils import piBin, policyProbsBin
from functools import partial 
from utils import str2bool

def cartpoleVL(bts, epsilon, label, gamma, defaultReward, vArgs, piArgs, nEp, write = False):
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
  '''
  print(bts)
  if write: 
    fname = 'cartpole-results/cartpoleVL-eps-{}-bts-{}-{}.csv'.format(epsilon, bts, label)
    results = ''
  
  #Initialize  
  env = Cartpole(gamma = gamma, epsilon = epsilon, defaultReward = defaultReward, vFeatureArgs = vArgs, piFeatureArgs = piArgs)
  bHat = np.zeros(env.nPi)

  #Run sim
  for ep in range(nEp): 
    fPi = env.reset() 
    done = False 
    score = 0 
    while not done: 
      score += 1
      aProb = piBin(fPi, bHat) 
      a = np.random.random() < aProb * (1 - epsilon) 
      fPi, F_V, F_Pi, A, R, Mu, M, done = env.step(a, bHat)
      if not done: 
        bHat = betaOpt(policyProbsBin, epsilon, M, A, R, F_Pi, F_V, Mu, bts = bts)
    print('Episode {} Score: {}'.format(ep, score))
    if write: 
      results += '{},{},{},{}\n'.format(label, 'eps-{}-bts-{}'.format(epsilon, bts), ep, score) 
      fhandle = open(fname, 'w')
      fhandle.write(results)
      fhandle.close()
      
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
  args = parser.parse_args()
  
  print(args.bts)
  
  np.random.seed(args.randomSeed)
  
  vArgs  = {'featureChoice':args.featureChoiceV, 'sigmaSq':args.sigmaSqV, 'gridpoints':args.gridpointsV}
  piArgs = {'featureChoice':'identity', 'sigmaSq':None, 'gridpoints':None} #Currently IGNORING Pi arguments, using identity 
  
  pool = mp.Pool(args.nRep) 
  cartpoleVL_partial = partial(cartpoleVL, gamma = args.gamma, defaultReward = args.defaultReward, 
                               vArgs = vArgs, piArgs = piArgs, nEp = args.nEp, write = args.write) 
  argList = [(args.bts, args.epsilon, label) for label in range(args.nRep)]
  pool.starmap(cartpoleVL_partial, argList) 