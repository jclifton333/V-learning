# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:58:24 2017

@author: Jesse

This file contains functions for computing state features. 
"""

import numpy as np

def intercept(s):
  '''
  Prepends 1 to state s to include intercept. 
  '''
  return np.concatenate(([1], s))
  
def identity(s):
  return s
  
def gKernel(x, sigmaSq):
  '''
  Gaussian kernel.
  '''
  norm = np.linalg.norm(x, axis=1)
  return np.exp(-norm**2/(2 * sigmaSq))

def gRBF(s, basis, sigmaSq = 1):
  '''
  Gaussian radial basis function features.
  
  Parameters
  ----------
  s : state (1d array)
  basis : basis vectors (2d array of size nBasis x nS)
  sigmaSq : variance for gaussian rbf
  
  Returns
  -------
  features : Gaussian RBF features of state s (1d array of length nBasis)  
  '''
  nBasis = basis.shape[0]   
  features = np.zeros(nBasis)
  for i in range(nBasis):
    features[i] = gKernel(s - basis[i, :], sigmaSq)
  return features
  
