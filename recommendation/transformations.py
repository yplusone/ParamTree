"""

Copyright (C) 2014 Cornell University
See Interpolation.py

transformation functions for Interpolations

This file also explains how to construct a new transformation method - see T_ID
(the identity transformation) for an example:

   Create the transformation function T(D,R) -> (D',R')
   This function is overloaded, so the type of returns depends on the inputs
   
       out = myT(inverse, canUpdate, tops, *args)

   * inverse is either True or False
     
   * canUpdate is either True or False
     Some transformtions may have parameters that depend on the input samples,
     values.  For example, a map from D to to [0,1]^d could require the minimum
     and maximum samples, elementwise.
     canUpdate indicates whether any model parameters can be updated or not.

    * tops is a dictionary with transformation options


   There are 4 different input/output sets.  In the following, D represents an
   np.array in the domain and D' is an np.array in the transformed domain.
   Similar for R and R'.  Rprime is the derivative of the function; note that
   Rprime may be None.

   # These are for transforming (samples,values,derivatives) tuples
   D', R', Rprime' = myT(False, True,  tops, D, R, Rprime)
   D', R', Rprime' = myT(False, False, tops, D, R, Rprime)

   # This is for transforming the locations that we wish to interpolate at
   D'              = myT(False, False, tops, D)

   # This is for taking the inverse transform of either
   # (samples',values',derivatives') tuples or of (locs',ests',derivs') tuples
   D,  R, Rprime  = myT(True,  False, tops, D', R', Rprime')


"""

import numpy as np
from .general_functions import make2d

"""
This is the identity function, and we describe here the output conditions given
different inputs.
"""
def T_ID(inverse, canUpdate, tops, *args):
    """
    Identity transform, i.e. do nothing
    transformation_options:
    * 'method':'ID'
    """
    # first make copies of *args so that nothing changes in place
    myargs = [ a.copy() if a is not None else None for a in args ]

    if inverse:
        # then we're taking the inverse, so args = [D', R', Rprime'],
        # and return D, R, Rprime
        return myargs[0], myargs[1], myargs[2]

    if canUpdate:
        # If there are any parameters in the transformation model that depend
        # on (samples,values,derivatives), we are free to change them here.
        # Example:
        # tops['some_param_here'] = some_func(args[0],args[1])
        # NOTE: you must change tops IN PLACE
        pass
    
    if len(args)==1:
        # then we're transforming locs into locs', so args = [D] and return D'
        return myargs[0]
    else:
        # we're transforming (samples,values,derivatives) into
        # (samples',values',derivatives'),
        # so that args = [D, R, Rprime], and we return D', R', Rprime'
        return myargs[0], myargs[1], myargs[2]

def T_scale01(inverse, canUpdate, tops, *args):
    """
    This shifts and scales the domain to the [0,1]^d hypercube;
    the range is not transformed.
    
    transformation_options:
    * 'method':'scale01'
    * 'box': a 2Xd np.array, call it T
        Linear shift and scale each dimension di so that all samples lie in the
        hypercube with lower bounds T[0,:] and upper bounds T[1,:]
        Also, T can be supplied as a tuple (L,U) for arbitrary dimension, so
        that the box is [L,U]^n.
        
        Given the set of all samples in D, let vmax be the element-wise maximum
        of all samples and vmin be the element-wise minimum
        Then
            T[0,:] = vmin          # the shift
            T[1,:] = 1/(vmax-vmin) # the scaling
        If vmax=vmin for some elements, then the shift is 0 and the scale is 1
        
        IMPLEMENTATION NOTE: box is treated as a transform (next bullet) by
            calculating the appropriate transform when canUpdate is True
    * 'box01': boolean
        Specific instance of 'box', if True then shifts/scales domain to [0,1]^d
    * 'transform': a 2Xd np.array, call it T
        The first row is the shift, the second is the scale ---
        given a point p in the domain, compute p' as
            p' = (p-T[0,:])*T[1,:]
        T can be supplied by the user        
    """
    # first make copies of *args so that nothing changes in place
    myargs = [ a.copy() if a is not None else None for a in args ]

    if inverse:
        # return the inverse of args = [D', R',Rprime']
        samples = myargs[0]
        values  = myargs[1]
        derivs  = myargs[2]
        ss = tops['transform']
        shift = ss[0,:]
        scale = ss[1,:]
        samples /= scale
        samples += shift
        if derivs is not None:
            derivs *= scale
        return samples, values, derivs

    samples = myargs[0]
    if len(args)==1:
        # then we're transforming the locations at which we interpolate
        derivs = None
    else:
        derivs = myargs[2]

    if (canUpdate and tops.get('box01',False)) or \
       (canUpdate and (tops.get('box') is not None)) or \
       (tops.get('transform') is None and tops.get('box') is None):
        # then we have to create the shift and scale
        # we'll make these so that the transformed points fall
        # in [0,1]^N or in tops['box']
        val_min = np.min(samples,0)
        val_max = np.max(samples,0)
        # we need to be careful if val_min=val_max
        # in this case, shift = 0, scale = 1
        shift = np.zeros(np.shape(val_min))
        scale = np.zeros(np.shape(val_min))
        n = np.size(val_min)
        boxbds = tops.get('box',np.vstack((np.zeros((1,n)),np.ones((1,n)))))
        if type(boxbds)!= np.array:
            boxbds_tmp = np.ones((2,n))
            boxbds_tmp[0] *= boxbds[0]
            boxbds_tmp[1] *= boxbds[1]
            boxbds = boxbds_tmp
        for ii, (vmin, vmax, bd) in enumerate(zip(val_min,val_max,boxbds.T)):
            if vmin==vmax:
                shift[ii] = 0.
                scale[ii] = 1.
            else:
                scale[ii] = (bd[1]-bd[0])/(vmax-vmin)
                shift[ii] = vmin - bd[0]/scale[ii]
        tops['transform'] = np.vstack((shift,scale))

    # regardless of whether args=[D,R,Rprime] or args=[D], we perform the same
    # transformation on D, Rprime
    ss = tops['transform']
    shift = ss[0,:]
    scale = ss[1,:]
    samples -= shift
    samples *= scale
    if derivs is not None:
        derivs /= scale # note: shifting does not affect the derivatives
    

    if len(args)==1:
        # just returning samples'
        return samples
    else:
        # return samples' and values' (where values'=values)
        return samples, myargs[1], derivs

def T_scale01_domain_range(inverse, canUpdate, tops, *args):
    """
    This shifts and scales the domain to the [0,1]^d1 hypercube;
    Also shift and scales the range to the [0,1]^d2 hypercube
    
    transformation_options:
    * 'method':'scale01_domain_range'
    * 'transform domain': a 2Xd np.array, call it T
        The first row is the shift, the second is the scale ---
        given a point p in the domain, compute p' as
            p' = (p-T[0,:])*T[1,:]
        T can be supplied by the user
    * 'transform range': similar to 'transfer domain', but for the range
    * 'box01': a boolean, defaulted to False
        If True, the transform T is updated automatically when canUpdate=True
        Given the set of all samples in D, let vmax be the element-wise maximum
        of all samples and vmin be the element-wise minimum
        Then
            T[0,:] = vmin          # the shift
            T[1,:] = 1/(vmax-vmin) # the scaling
        If vmax=vmin for some elements, then the shift is 0 and the scale is 1
    """
    # first make copies of *args so that nothing changes in place
    myargs = [ a.copy() if a is not None else None for a in args ]

    if inverse:
        # return the inverse of args = [D', R',Rprime']
        samples = myargs[0]
        values  = myargs[1]
        derivs  = myargs[2]
        # shift, scale domain
        ssdom = tops['transform domain']
        shift = ssdom[0,:]
        scale = ssdom[1,:]
        samples /= scale
        samples += shift
        if derivs is not None:
            derivs *= scale
        # shift, scale range
        ssran = tops['transform range']
        shift = ssran[0,:]
        scale = ssran[1,:]
        values /= scale
        values += shift
        return samples, values, derivs

    samples = myargs[0]
    if len(args)==1:
        # then we're transforming the locations at which we interpolate
        derivs = None
    else:
        values = myargs[1]
        derivs = myargs[2]

    def get_shift_scale(data):
        val_min = np.min(data,0)
        val_max = np.max(data,0)
        # we need to be careful if val_min=val_max
        # in this case, shift = 0, scale = 1
        shift = np.zeros(np.shape(val_min))
        scale = np.zeros(np.shape(val_min))
        n = np.size(val_min)
        for ii, (vmin, vmax) in enumerate(zip(val_min,val_max)):
            if vmin==vmax:
                shift[ii] = 0.
                scale[ii] = 1.
            else:
                scale[ii] = 1/(vmax-vmin)
                shift[ii] = vmin
        return np.vstack((shift,scale))

    if (canUpdate and tops.get('box01',False)) or \
       (tops.get('transform domain') is None):
        # then we have to create the shift and scale
        # we'll make these so that the transformed points fall
        # in [0,1]^N
        tops['transform domain'] = get_shift_scale(samples)
        tops['transform range']  = get_shift_scale(values)
        

    # transform samples and derivs (if derivs was an input)
    ssdom = tops['transform domain']
    shift = ssdom[0,:]
    scale = ssdom[1,:]
    samples -= shift
    samples *= scale
    if derivs is not None:
        derivs /= scale # note: shifting does not affect the derivatives

    if len(args)==1:
        # just returning samples'
        return samples
    else:
        # return samples', values', derivs'
        # already calculated samples', derivs' above.  do it for values'
        ssran = tops['transform range']
        shift = ssran[0,:]
        scale = ssran[1,:]
        values -= shift
        values *= scale
        return samples, myargs[1], derivs
