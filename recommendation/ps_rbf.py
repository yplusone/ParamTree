"""

Copyright (C) 2014 Cornell University

Radial Basis Function interpolation in d-dimensional domain, r-dimensional range

Call with either

    rs = RBF(phi='thinplate', phi_var=1, polyorder=1)
or
    rs = RBF('RBF',model_options,transform_options)

model_options:
    A dictionary with the following keys ---

    * 'phi': a string specifying the type of RBF
    * 'phi_var': any parameters necessary for the corresponding RBF type
      The following options are available:
      phi                   equation                   phi_var
      'linear'              P(r) = c*r                 c
      'Gaussian'            P(r) = exp(-(eps*r)**2)    eps
      'multiquadratic'      P(r) = sqrt(1+(eps*r)**2)  eps
      'inverse_quadratic'   P(r) = 1/(1+(eps*r)**2)    eps
      'thinplate'           P(r) = r**2 * ln(r)        None
      'compact'             P(r) = {v    if r>x        [x,v,phi2,phi_var2]
                                   {P(r) if r<=x,
                                   {  where P(r) is specified by phi2, phi_var2
                                   
    * 'polyorder': an integer specifying the order of the polynomial tail
      this should either be 1 or 2
    
transformation default:
    scale01_domain_range, to automatically scale the samples to the the [0,1]^d
    hypercube and scale the range to [0,1]

"""

from .rs_parent import RS_Parent
from .general_functions import *
import numpy as np
import scipy.spatial as scs
from time import time
from sklearn.preprocessing import MinMaxScaler

class RBF(RS_Parent):

    def __init__(self,phi='inverse_quadratic',phi_var=1,polyorder=1,**kwargs):
        super().__init__(phi,phi_var,polyorder,**kwargs)
        self.mops['model requires validation set'] = False



    def get_default_mops(self,*args,**kwargs):
        phi       = args[0]
        phi_var   = args[1]
        polyorder = args[2]
        mops = {'phi':phi,
                'phi_var':phi_var,
                'polyorder':polyorder}
        return mops

    def get_default_tops(self,*args,**kwargs):
        tops ={'method':'scale01_domain_range',
               'box01':True}
        return tops

    def make_model(self):
        samples = self.samples
        values  = self.values
        mops    = self.mops
        
        M = np.size(samples,0)
        N = np.size(samples,1)
        # get rid of any point dimensions
        # (they can screw up the matrix and make it singular)
        shape = np.shape(samples)
        samples = remove_point_dims(samples)
        if np.shape(samples)==shape:
            self.point_dims = False
        else:
            self.point_dims = True
     
        # make the distance matrix
        d = scs.distance_matrix(samples,samples)
        # and apply phi
        d = self.phiOfX(d,mops['phi'],mops['phi_var'])

        # make the polynomial matrix
        p = make_poly(samples,mops['polyorder'])
        pdim = np.size(p,1)

        # assemble the matrix
        full = np.zeros((pdim+M,pdim+M))
        full[:M,:M] = d
        full[M:,:M] = p.T
        full[:M,M:] = p
        # make the b vector
        b = np.vstack((values,np.zeros((pdim,1))))

        # solve
        Apinv = np.linalg.pinv(full)

        self.model = Apinv.dot(b)


    def perform_interp(self, locs):
        mops = self.mops
        # returns values at locs based on knots, values, and model
        # locs is a np.mat, each row is a vector specifying a single location
        
        # get rid of any point dimensions
        if self.point_dims:
            r = remove_point_dims(self.samples,locs)
            samples = r[0]
            locs    = r[1]
        else:
            samples = self.samples
        
        # apply phi to (knots-locs)
        d = scs.distance_matrix(locs,samples)
        phiOfD = self.phiOfX(d,mops['phi'],mops['phi_var'])
        # get the polynomial tail
        p = make_poly(locs, mops['polyorder'])

        # get the values
        return np.dot(np.hstack((phiOfD,p)),self.model)
        
    def fit(self,x,y):
        # self.scaler.fit(x)
        # x = self.scaler.transform(x)
        self.xcolumns = list(x.columns)
        self.bounds = np.zeros((2,len(self.xcolumns)))
        for index,col in enumerate(self.xcolumns):
            self.bounds[0,index] = min(x[col])
            self.bounds[1, index] = max(x[col])
        y = np.array(y).reshape(-1,1)
        self.addSamples(np.array(x),y)
        self.buildModel()


    @staticmethod
    def phiOfX(amat, phi,phi_var):
        # amat is a np matrix, each element is distance R
        # apply phi to each element, return phi(mat) of same size
        if phi=='linear':
            return amat*phi_var
        if phi=='Gaussian':
            # then kind_options=[Gaussian exponent]
            return np.exp(-np.power((phi_var*amat),2))
        if phi=='multiquadratic':
            return np.power(1+np.power(amat*phi_var,2),0.5)
        if phi=='inverse_quadratic':
            return np.power(1+np.power(amat*phi_var,2),-1)
        if phi=='thinplate':
            iszero = (amat==0)
            tmp = 1.0*amat + 1.0*iszero
            tmp = np.multiply(np.power(amat,2),np.log(tmp))
            return np.multiply(tmp, (1-iszero) ) 
        if phi=='compact':
            tmp = amat.copy()
            ones = (tmp>=phi_var[0])
            tmp = self.phiOfX(tmp, phi_var[2], phi_var[3])
            tmp[ones] = phi_var[1]
            return tmp
        if phi=='cubic':
            return np.power(amat,3)
        return "Unrecognized phi = "+str(phi)

if __name__ == '__main__':
    
    import numpy as np
    
    ### some testing things to see it works
    print("Testing for RBF: ")
    tops = {'method':'scale01',
            'box':[[0,0],[2,2]]} # shift/scale points to [0,2]^2
                                 # see transform.py for more info
    test = RBF('linear',1,1)#,transform_options=tops)
    samples = np.array( [[1,2],[3,4]])
    vals = np.array( [[1], [2]])
    test.addSamples(samples,vals)
    moresamples = np.array([[10,8.2]])
    morevals = np.array([[5.61]])
    test.addSamples(moresamples,morevals)
    print("Here are the current samples")
    print(test.getSamples())
    print("Here are the current values")
    print(test.getValues())  
    print("Building model")
    test.buildModel()
    test.plot_2d(axes=[0, 1],kind='contour')
    print("Testing model: ")
    print("value of samples[0] should be ",vals[0]," calculated as ",test.interp(samples[0]))
    print("value of samples[1] should be ",vals[1]," calculated as ",test.interp(samples[1]))
    print("value of moresamples should ",morevals," calculated as ",test.interp(moresamples))
    print("At some other random points: ")
    print("value at [2,3] is: ",test.interp(np.array([[2,3]])))
    print("value at [10,0], [0,10] are: ",test.interp(np.array([[10,0],[0,10]])))


    ### TIMING
    import matplotlib.pyplot as plt
    from time import time
    
    d = 2
    neval = 10000

    nstart = int(np.ceil(np.log2(d*3)))+3
    locs  = np.random.rand(neval,d)
    tbuild = []
    teval  = []
    nvals  = []
    for n in [i for i in range(nstart,max(nstart+5,10))]:
        print('n: ',n)
        npts = 2**n
        samples = np.random.rand(npts,d)
        values  = np.random.rand(npts,1)
        test = RBF()
        test.addSamples(samples,values)
        t1 = time()
        test.buildModel()
        t2 = time()
        test.interp(locs)
        t3 = time()
        nvals.append(n)
        tbuild.append(t2-t1)
        teval.append((t3-t2)/neval)

    plt.figure(1)
    plt.subplot(211)
    plt.title('build time')
    plt.ylabel('log2(time [s])')
    plt.plot(nvals,np.log2(tbuild),'bo-')
    plt.subplot(212)
    plt.title('average evaluation time')
    plt.ylabel('log2(time [ms])')
    plt.xlabel('log_2(samples)')
    plt.plot(nvals,np.log2([t*1000 for t in teval]),'rx-')
    plt.show()
