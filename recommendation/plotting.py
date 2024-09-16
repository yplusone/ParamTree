"""
some plotting functions
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

class plotting:

##    def get_domain_dim(self):
##        return np.size(self.getSamples(),1)

    def make_samples(self, axes, bounds, center, npts):
        """
        make a set of samples
        axes is an ordered list
        bounds is a list of (lower,upper) bounds
        center is an array, the center of the remaining dimensions
            (None if there are no other dimensions)
        npts is a list
        """
        nsamp = np.prod(npts)
        dim   = self.get_domain_dim()
        nax   = len(axes)
        axsamplist = [ np.arange(b[0],b[1]+1/2/n,(b[1]-b[0])/(n-1)) \
                       for (b,n) in zip([bds for bds in bounds.T],npts) ]
        axsamples = np.zeros((nsamp,nax))
        for i, row in enumerate(itertools.product(*axsamplist)):
            axsamples[i] = row
        if center is None:
            return axsamples,axsamplist
        samples = np.tile(center,(nsamp,1))
        for (a,axs) in zip(axes,axsamples.T):
            samples = np.insert(samples, a, axs, 1)
        return samples, axsamplist

    def get_bounds(self, axes):
        bounds = np.zeros((2,len(axes)))
        s = self.getSamples()
        for i, a in enumerate(axes):
            L = np.min(s,0)[a]
            U = np.max(s,0)[a]
            D = U-L
            bounds[0,i] = L#-D/5.
            bounds[1,i] = U#+D/5.
        return bounds

    def plot_1d(self, axis, center=None, bounds=None, n=200,
                show=True, ax=None):
        """
        plot a 1 dimensional slice along axis from bounds[0] to bounds[1] with
        n points in between
        the other dimensions are held at center
        """
        if ax is None:
            f, ax = plt.subplots(1)
        if bounds is None:
            bounds = self.get_bounds(axes=axis)
        samples, axsamplist = self.make_samples([axis], bounds, center, [n])
        values  = self.interp(samples)
        ax.plot(axsamplist[0], values[:,0])
        if show:
            plt.show()
        return f, ax

    def plot_2d(self, axes, center=None, bounds=None, n=[100,100],
                show=True, ax=None, f=None,kind='contourf'):
        """
        plot a 2 dimensional slice along axes from bounds
        """
        if ax is None:
            f = plt.figure() 
            ax = plt.axes(projection='3d')
        if bounds is None:
            bounds = self.get_bounds(axes=axes)
        samples, axsamplist = self.make_samples(axes, bounds, center, n)
        values = self.interp(samples)
        X,Y = np.meshgrid(axsamplist[0],axsamplist[1])
        Z = np.reshape(values,np.shape(X))
        ax.plot_surface(X, Y, Z, cmap='rainbow')
        # if kind=='contourf':
        #     im = ax.contourf(X,Y,Z)
        # elif kind=='contour':
        #     im = ax.contour(X,Y,Z)
        # plt.colorbar(im)
        if show:
            plt.show()
        return f, ax
        
            

    def get_domain_dim(self):
        return 5

if __name__=='__main__':
    a = plotting()
    print(a.make_samples([0,3,4],np.array([[5,6],[10,11],[20,21]]).T,
                         [0.,1], [2,2,3]))
    

