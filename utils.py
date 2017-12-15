'''
Miscellaneous utilities.
'''
import argparse 

def str2bool(v):
  '''
  For parsing boolean arguments with argparse. 
  
  Usage: parser.add_argument("--nice", type=str2bool, help="Activate nice mode.")
  See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse. 
  '''
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
  else:
      raise argparse.ArgumentTypeError('Boolean value expected.')
