from __future__ import absolute_import
from __future__ import print_function
__author__ = 'noe'

import time
import numpy as np
from .. import moments

def genS(N):
    """ Generates sparsities given N (number of cols) """
    S = [10, 90, 100, 500, 900, 1000, 2000, 5000, 7500, 9000, 10000, 20000, 50000, 75000, 90000]  # non-zero
    return [s for s in S if s <= N]


def genX(L, N, n_var=None, const=False):
    X = np.random.rand(L, N)  # random data
    if n_var is not None:
        if const:
            Xsparse = np.ones((L, N))
        else:
            Xsparse = np.zeros((L, N))
        Xsparse[:, :n_var] = X[:, :n_var]
        X = Xsparse
    return X


def genY(L, N, n_var=None, const=False):
    X = np.random.rand(L, N)  # random data
    if n_var is not None:
        if const:
            Xsparse = -np.ones((L, N))
        else:
            Xsparse = np.zeros((L, N))
        Xsparse[:, :n_var] = X[:, :n_var]
        X = Xsparse
    return X


def reftime_momentsXX(X, remove_mean=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        s_ref = X.sum(axis=0)  # computation of mean
        if remove_mean:
            X = X - s_ref/float(X.shape[0])
        C_XX_ref = np.dot(X.T, X)  # covariance matrix
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def mytime_momentsXX(X, remove_mean=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        w, s, C_XX = moments.moments_XX(X, remove_mean=remove_mean)
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def reftime_momentsXXXY(X, Y, remove_mean=False, symmetrize=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        sx = X.sum(axis=0)  # computation of mean
        sy = Y.sum(axis=0)  # computation of mean
        if symmetrize:
            sx = 0.5*(sx + sy)
            sy = sx
        if remove_mean:
            X = X - sx/float(X.shape[0])
            Y = Y - sy/float(Y.shape[0])
        if symmetrize:
            C_XX_ref = np.dot(X.T, X) + np.dot(Y.T, Y)
            C_XY = np.dot(X.T, Y)
            C_XY_ref = C_XY + C_XY.T
        else:
            C_XX_ref = np.dot(X.T, X)
            C_XY_ref = np.dot(X.T, Y)
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def mytime_momentsXXXY(X, Y, remove_mean=False, symmetrize=False, nrep=3):
    # time for reference calculation
    t1 = time.time()
    for r in range(nrep):
        w, sx, sy, C_XX, C_XY = moments.moments_XXXY(X, Y, remove_mean=remove_mean, symmetrize=symmetrize)
    t2 = time.time()
    # return mean time
    return (t2-t1)/float(nrep)


def benchmark_moments(L=10000, N=10000, nrep=5, xy=False, remove_mean=False, symmetrize=False, const=False):
    #S = [10, 100, 1000]
    S = genS(N)

    # time for reference calculation
    X = genX(L, N)
    if xy:
        Y = genY(L, N)
        reftime = reftime_momentsXXXY(X, Y, remove_mean=remove_mean, symmetrize=symmetrize, nrep=nrep)
    else:
        reftime = reftime_momentsXX(X, remove_mean=remove_mean, nrep=nrep)

    # my time
    times = np.zeros(len(S))
    for k, s in enumerate(S):
        X = genX(L, N, n_var=s, const=const)
        if xy:
            Y = genY(L, N, n_var=s, const=const)
            times[k] = mytime_momentsXXXY(X, Y, remove_mean=remove_mean, symmetrize=symmetrize, nrep=nrep)
        else:
            times[k] = mytime_momentsXX(X, remove_mean=remove_mean, nrep=nrep)

    # assemble report
    rows = ['L, data points', 'N, dimensions', 'S, nonzeros', 'time trivial', 'time moments_XX', 'speed-up']
    table = np.zeros((6, len(S)))
    table[0, :] = L
    table[1, :] = N
    table[2, :] = S
    table[3, :] = reftime
    table[4, :] = times
    table[5, :] = reftime / times

    # print table
    if xy:
        fname = 'moments_XXXY'
    else:
        fname = 'moments_XX'
    print(fname + '\tremove_mean = ' + str(remove_mean) + '\tsym = ' + str(symmetrize) + '\tconst = ' + str(const))
    print(rows[0] + ('\t%i' * table.shape[1])%tuple(table[0]))
    print(rows[1] + ('\t%i' * table.shape[1])%tuple(table[1]))
    print(rows[2] + ('\t%i' * table.shape[1])%tuple(table[2]))
    print(rows[3] + ('\t%.3f' * table.shape[1])%tuple(table[3]))
    print(rows[4] + ('\t%.3f' * table.shape[1])%tuple(table[4]))
    print(rows[5] + ('\t%.3f' * table.shape[1])%tuple(table[5]))
    print()


def main():
    LNs = [(100000, 100, 10), (10000, 1000, 7), (1000, 2000, 5), (250, 5000, 5), (100, 10000, 5)]
    for L, N, nrep in LNs:
        benchmark_moments(L=L, N=N, nrep=nrep, xy=False, remove_mean=False, symmetrize=False, const=False)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=False, remove_mean=False, symmetrize=False, const=True)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=False, remove_mean=True, symmetrize=False, const=False)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=False, remove_mean=True, symmetrize=False, const=True)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=False, symmetrize=False, const=False)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=False, symmetrize=False, const=True)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=False, symmetrize=True, const=False)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=False, symmetrize=True, const=True)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=True, symmetrize=False, const=False)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=True, symmetrize=False, const=True)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=True, symmetrize=True, const=False)
        benchmark_moments(L=L, N=N, nrep=nrep, xy=True, remove_mean=True, symmetrize=True, const=True)


if __name__ == "__main__":
    main()