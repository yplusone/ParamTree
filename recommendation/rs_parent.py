


import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from warnings import warn
import importlib
import random
import itertools

### import plotting
from .plotting import plotting

### import general functions
from .general_functions import make2d, remove_point_dims

### import the transformation functions
from .transformations import T_ID, T_scale01, T_scale01_domain_range
namesT = [ 'ID', 'scale01', 'scale01_domain_range' ]
funcsT = [ T_ID, T_scale01, T_scale01_domain_range ]
mymapT = dict(zip(namesT,funcsT))


class RS_Parent(plotting):

    def __init__(self, *args, **kwargs):
        """
        INPUT:
        - kind: a string in {'poly','RBF'} (as of last update, anyway)
        - *args: any arguments to help set up the interpolation
        - **kwargs:
          may include 'model_options' and 'transform_options' as keyword args
        """
        self.samples = None
        self.values  = None
        self.derivs  = None
        self.KFVdata = None
        self.model   = None
        self.mops    = None
        self.tops    = None

        # get the input model and transform options
        # (NOTE: dict.get(a,b) is bad ---
        #  even if a in dict.keys(), Python still evaluates b)
        if 'model_options' in kwargs:
            self.mops = kwargs['model_options']
        else:
            self.mops = self.get_default_mops(*args,**kwargs)
        if 'transform_options' in kwargs:
            tops = kwargs['transform_options']
        else:
            tops = self.get_default_tops(*args,**kwargs)
        self.setTransformFunc(tops) # this just makes sure that tops is correct

    def get_info(self):
        def get_name(obj):
            try:
                return obj.__name__
            except:
                return str(obj)
        def get_nice_dict_string(adict):
            kv = sorted([ (str(k),get_name(v)) for k,v in adict.items() ])
            string = '_'*100+'\n{'
            for k,v in kv:
                string += '\n'+k+':'+v+','
            return string + '\n}'
        return "MODEL OPTIONS:\n"+get_nice_dict_string(self.mops)+\
               "\nTRANSFORMATION OPTIONS:\n"+get_nice_dict_string(self.tops)

    def get_default_mops(self):
        # a function that returns the default model options
        raise Exception("get_default_mops hasn't been defined yet.")

    def get_default_tops(self):
        # a function that returns the default transformation options
        raise Exception("get_default_tops hasn't been defined yet.")

    def make_model(self):
        # a function that creates and returns a model used for evaluating the
        # response surface
        raise Exception("make model hasn't been defined yet.")

    def perform_interp(self, locs):
        # locs is a 2-d array of samples
        # this function determines the response surface value at these locations
        # return these values
        raise Exception("perform_interp hasn't been defined yet.")

    ############################################################################

    def setTransformFunc(self, tops):
        t = tops.get('method',None)
        if t in mymapT.keys():
            self.transform = mymapT[t]
            self.tops = tops
        else:
            msg = 'unrecognized transformation method = '+str(t)
            warn(msg)

    def addSamples(self, samples, values, derivs=None):
        # add samples and their values to the state
        if self.model is not None:
            samples, values, derivs =\
                self.transform(False,False,self.tops,samples,values,derivs)
        if self.samples is None:
            self.samples = samples.astype(float)
            self.values  = values.astype(float)
            if derivs is not None:
                self.derivs  = derivs.astype(float)
        else:
            self.samples = np.vstack((self.samples,samples))
            self.values  = np.vstack((self.values,values))
            if derivs is not None:
                self.derivs  = np.vstack((self.derivs,derivs))

    def buildModel(self,**kwargs):
        tops = self.tops
        ### transformation stuff
        # since we're building the model, the transformation can be updated
        # so, first untransform all the samples
        strue = self.getSamples()
        vtrue = self.getValues()
        dtrue = self.getDerivs()
        # make sure we're still using the correct transform function
        self.setTransformFunc(tops)
        # now, we go ahead and transform the points
        self.samples, self.values, self.derivs =\
                self.transform(False,True,tops,strue,vtrue,dtrue)
        ### build model stuff
        results = self.make_model(**kwargs)
        return results

    def interp(self, locs, **kwargs):
        # return the interpolation of locs
        locs = make2d(locs)
        locs = locs.astype(float)
        locs = self.transform(False,False,self.tops,locs)
        ests = self.perform_interp(locs,**kwargs)
        locs, ests, _ = self.transform(True,False,self.tops,locs,ests,None)
        return ests

    def interpDeriv(self, locs, delta=0.001):
        # return the numerical derivative at each of locs
        if len(np.shape(locs))==1:
            locs = np.reshape(locs,(1,-1))
        derivs = np.zeros(np.shape(locs))
        derivs[:,:] = self.interp(locs)
        for d in range(np.size(locs,1)):
            locs[:,d] -= delta
            derivs[:,d:d+1] -= self.interp(locs)
            derivs[:,d] /= delta
            locs[:,d] += delta
        return derivs

    def getSamples(self):
        if self.model is not None:
            s, v, d = self.transform(True,False,self.tops,
                                     self.samples,self.values,self.derivs)
            return s
        else:
            return self.samples

    def getValues(self):
        if self.model is not None:
            s, v,d  = self.transform(True,False,self.tops,
                                     self.samples,self.values,self.derivs)
            return v
        else:
            return self.values

    def getDerivs(self):
        if self.model is not None:
            s, v,d  = self.transform(True,False,self.tops,
                                     self.samples,self.values,self.derivs)
            return d
        else:
            return self.derivs

    def has_derivatives(self):
        return self.derivs is not None

    def getSamplesTransformed(self):
        return self.samples

    def getValuesTransformed(self):
        return self.values

    def getDerivsTransformed(self):
        return self.derivs

    def resetSVD(self):
        self.samples = None
        self.values  = None
        self.derivs  = None    
    
