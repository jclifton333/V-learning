#!/bin/bash

GAMMA=0.9                            #Discount factor
SIGMA_SQ_V=1.0                       #Variance for v-function features, if gRBF is used 
GRIDPOINTS_V=5                       #Number of points per dimension to use for v-function feature basis, if gRBF is used 
FEATURE_CHOICE_V='gRBF'              #Choice of features for v-function ('identity', 'intercept', or 'gRBF')
SIGMA_SQ_PI=1.0                      #Variance for policy features, if gRBF is used
GRIDPOINTS_PI=3                      #Number of points per dimension for policy feature basis, if gRBF is used 
FEATURE_CHOICE_PI='identity'         #Choice of features for policy ('identity', 'intercept', or 'gRBF')
NUM_EP=1000                          #Number of episodes 
NUM_REP=1                            #Number of replicates; simulate.py uses one core per rep! 
SEED=3                               #Random seed 
FIX_UP_TO=400                        #Number of observations to use in reference distribution
WRITE='False'                        #Boolean for writing results to file 
ENV_NAME='Cartpole'                  #Simulation environment name ('Cartpole', 'FlappyBirdEnv', or 'randomFiniteMDP'

mkdir -p results

#python3 simulate.py --envName="$ENV_NAME" --gamma="$GAMMA" --epsilon=0.05 --bts='False' --sigmaSqV="$SIGMA_SQ_V" \
#  --initializer='None' --gridpointsV="$GRIDPOINTS_V" --featureChoiceV="$FEATURE_CHOICE_V" --sigmaSqPi="$SIGMA_SQ_PI" \
#  --gridpointsPi="$GRIDPOINTS_PI" --featureChoicePi="$FEATURE_CHOICE_PI" --randomSeed="$SEED" \
#   --nEp="$NUM_EP" --nRep="$NUM_REP" --write="$WRITE" --fixUpTo="$FIX_UP_TO" & 

python3 simulate.py --envName="$ENV_NAME" --gamma="$GAMMA" --epsilon=0.05 --bts='False' --sigmaSqV="$SIGMA_SQ_V" \
  --initializer='multistart' --gridpointsV="$GRIDPOINTS_V" --featureChoiceV="$FEATURE_CHOICE_V" --sigmaSqPi="$SIGMA_SQ_PI" \
  --gridpointsPi="$GRIDPOINTS_PI" --featureChoicePi="$FEATURE_CHOICE_PI" --randomSeed="$SEED" \
  --nEp="$NUM_EP" --nRep="$NUM_REP" --write="$WRITE" --fixUpTo="$FIX_UP_TO" &  
  
wait
echo all processes complete