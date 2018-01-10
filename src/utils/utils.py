'''
Miscellaneous utilities.
'''
import argparse 
import numpy as np

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

def intOrNone(v):
  '''
  For parsing integer or None arguments with argparse. 
  
  Usage: parser.add_argument("--number", type=intOrNone, help="Integer or None.")
  '''  
  if v == 'None':
    return None 
  elif v.isdigit(): 
    return int(v)
  else: 
    return arparse.ArgumentTypeError('Integer or None expected.') 
    
def strOrNone(v):
  '''
  For parsing string or None arguments with argparse. 
  
  Usage: parser.add_argument("--number", type=intOrNone, help="String or None.")
  '''  
  if v == 'None':
    return None 
  elif isinstance(v, str): 
    return v
  else: 
    return arparse.ArgumentTypeError('String or None expected.') 
    
def onehot(integer, basis_size): 
  '''
  Onehot encoding of integer. 
  '''
  vec = np.zeros(basis_size)
  vec[integer] = 1 
  return vec 
  