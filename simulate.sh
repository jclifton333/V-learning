#!/bin/bash

GAMMA=0.5                            #Discount factor
NUM_EP=100                            #Number of episodes 
NUM_REP=1                            #Number of replicates; simulate.py uses one core per rep! 
SEED=3                               #Random seed 
FIX_UP_TO=400                        #Number of observations to use in reference distribution
WRITE='False'                        #Boolean for writing results to file 
ENV_NAME='Glucose'                  #Simulation environment name ('Cartpole', 'FlappyBirdEnv', 'RandomFiniteMDP', 'SimpleMDP', 'Glucose', or 'Gridworld')
RANDOM_SHRINK='False'                 #Boolean for random shrinkage (for added exploration)
ACTOR_CRITIC='False'                  #Boolean for using actor-critic rather than regular v-learning

mkdir -p results
mkdir -p results/$ENV_NAME

python3 simulate.py --envName="$ENV_NAME" --gamma="$GAMMA" --epsilon=0.2 --bts='True' --initializer='multistart' \
   --randomShrink="$RANDOM_SHRINK" --randomSeed="$SEED" --nEp="$NUM_EP" --nRep="$NUM_REP" --write="$WRITE" --fixUpTo="$FIX_UP_TO" \
   --actorCritic="$ACTOR_CRITIC"   
  
wait
echo all processes complete