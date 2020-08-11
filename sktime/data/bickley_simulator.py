import numpy as np
from scipy.integrate import solve_ivp

class BickleyJet(object):
    '''
    Implementation of the Bickley jet based on "A Spectral Clustering Approach to Lagrangian Vortex Detection"
    by A. Hadjighasem, D. Karrasch, H. Teramoto, and G. Haller.
    '''
    def __init__(self):
        # set parameters
        self.U0 = 5.4138 # units changed to 10^6 m per day
        self.L0 = 1.77   # in 10^6 m
        self.r0 = 6.371  # in 10^6 m
        self.c = np.array([0.1446, 0.205, 0.461])*self.U0
        self.eps = np.array([0.075, 0.15, 0.3])
        self.k = np.array([2, 4, 6]) / self.r0
        
    def generateTrajectories(self, m):
        '''
        
        Parameters
        ----------
        m : int
            Number of particles.
            
        Returns
        -------
        Z : (2, 401, m) ndarray
            Trajectories for m uniformly distributed test points in Omega = [0, 20] x [-3, 3].
        '''
        T0 = 0
        T1 = 40
        nT = 401 # number of time points, TODO: add as parameter?
        T = np.linspace(T0, T1, nT) # time points
        X = np.vstack(( 20*np.random.rand(m), 6*np.random.rand(m) - 3 )) # m randomly sampled points in Omgea

        Z = np.zeros((2, nT, m))
        for i in range(m):
            sol = solve_ivp(self.rhs, [0, 40], X[:, i], t_eval=T)
            sol.y[0, :] = np.mod(sol.y[0, :], 20) # periodic in x-direction
            Z[:, :, i] = sol.y
        return Z
 
    def rhs(self, t, x):
        c = self.c
        eps = self.eps
        k = self.k
        f = np.real(eps[0] * np.exp(-1j*k[0]*c[0]*t) * np.exp(1j*k[0]*x[0]) \
                  + eps[1] * np.exp(-1j*k[1]*c[1]*t) * np.exp(1j*k[1]*x[0]) \
                  + eps[2] * np.exp(-1j*k[2]*c[2]*t) * np.exp(1j*k[2]*x[0]))
        df_dx = np.real(eps[0] * np.exp(-1j*k[0]*c[0]*t)*1j*k[0] * np.exp(1j*k[0]*x[0]) \
                      + eps[1] * np.exp(-1j*k[1]*c[1]*t)*1j*k[1] * np.exp(1j*k[1]*x[0]) \
                      + eps[2] * np.exp(-1j*k[2]*c[2]*t)*1j*k[2] * np.exp(1j*k[2]*x[0]))
        
        sech_sq = self._sech(x[1]/self.L0)**2
         
        return np.array([ self.U0*sech_sq + 2*self.U0*np.tanh(x[1]/self.L0)*sech_sq*f,  self.U0*self.L0*sech_sq*df_dx ])

    def _sech(self, x):
        '''
        Hyperbolic secant.
        '''
        return 1 / np.cosh(x)


#%%------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.cluster.vq import kmeans2
    import matplotlib.pyplot as plt
    
    import d3s.kernels as kernels
    import d3s.algorithms as algorithms
    
    m = 5000
    sys = BickleyJet()
    Z = sys.generateTrajectories(m)
    nT = Z.shape[1]
    
    #%% plot particles
    for i in range(0, nT, 5):
        plt.figure(1); plt.clf()
        plt.scatter(Z[0, i, :], Z[1, i, :])
        plt.xlim(0, 20); plt.ylim(-3, 3)
        plt.pause(0.01)
    
    #%% apply kernel CCA to detect coherent sets
    X = Z[:,  0, :] # particles at time T0
    Y = Z[:, -1, :] # particles at time T1
    
    sigma = 1
    k = kernels.gaussianKernel(sigma)
    
    evs = 9 # number of eigenfunctions to be computed
    d, V = algorithms.kcca(X, Y, k, evs, epsilon=1e-3)
    
    #%% plot eigenfunctions
    for i in range(evs):
        plt.figure()
        plt.scatter(X[0, :], X[1, :], c=V[:, i])
    plt.show()
    
    #%% k-means of eigenfunctions
    c, l = kmeans2(np.real(V), 7)
    plt.figure()
    plt.scatter(X[0, :], X[1, :], c=l)
    plt.show()
