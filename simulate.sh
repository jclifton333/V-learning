#!/bin/bash
GAMMA=0.9
SIGMA_SQ_V=1.0 
GRIDPOINTS_V=5 
FEATURE_CHOICE_V='gRBF' 
SIGMA_SQ_PI=1.0
GRIDPOINTS_PI=3 
FEATURE_CHOICE_PI='identity'
NUM_EP=1000
NUM_REP=1
SEED=3
FIX_UP_TO=400
WRITE='False'
ENV_NAME='Cartpole'

mkdir results

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