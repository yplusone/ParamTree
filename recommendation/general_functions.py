"""

Copyright (C) 2014 Cornell University
See Interpolation.py

Some helpful functions for interpolation

"""

import numpy as np
import itertools
from warnings import warn

def make2d(data):
    # returns a VIEW
    if len(np.shape(data))==1:
        return np.reshape(data,(1,-1))
    return data

def make_poly(samples, deg):
    """
    Given a sample of m points in d dimensions, create a deg degree matrix of
    their polynomial coefficients
    samples is an N*M array

    Example:
    for 2 dimensions and n pointss with second degree polynomial,
    samples = [ x_11 x_12 ]
                  |   |
              [ x_n1 x_n2 ]
    then we have
    X = [ 1 x_11 x_12 x_11*x_11 x_11*x_12 x_12*x_12 ]
        [ 1 x_21 x_22 x_21*x_21 x_21*x_22 x_22*x_22 ]
          |   |    |      |         |         |     ]
        [ 1 x_n1 x_n2 x_n1*x_n1 x_n1*x_n2 x_n2*x_n2 ]
     
    """ 
    ### first create the set of all columns of samples that we multiply together
    # e.g. for the X above, indices = [ [0], [1], [0,0], [0,1], [1,1] ]
    # (this doesn't include the column of one's at the beginning)
    M = np.size(samples,1)
    indices = []
    for ii in range(1,deg+1):
        for inds in itertools.combinations_with_replacement(range(M),ii):
            indices.append(inds)
    ### now construct X
    N = np.size(samples,0)
    X = np.ones((N,len(indices)+1))
    for col, inds in enumerate(indices):
        for i in inds:
            X[:,col+1] *= samples[:,i]
    return X

def remove_point_dims(samples, *args):
    """
    INPUTS:
    - samples - mXn np.array
    - *args - all additional arguments must also be np.arrays with n columns
    RETURNS:
    - if args None, returns a single mXn2 array, with n2<=n
      otherwise, returns a list of arrays, each with n2 columns
    Remove any dimension for which the samples are all the same value
    Remove these same dimensions from all arrays in *args
    e.g.
    samples, args[0] = np.array([[0,0,0],     ,  np.array([[1,2,3],
                                 [0,1,2],                 [4,5,6]])
                                 [0,0,4]])                
    returns          [ np.array([[0,0],       ,  np.array([[2,3],   ]
                                 [1,2],                   [5,6]])
                                 [3,4]])                  
    """
    samples = samples.copy()
    val_min = np.min(samples,0)
    val_max = np.max(samples,0)
    elim = [ ii for ii, (vup, vlo) in enumerate(zip(val_max,val_min)) \
             if vup==vlo ]
    if not args:
        return np.delete(samples,elim,1)
    return [np.delete(samples,elim,1)] + [np.delete(a,elim,1) for a in args]