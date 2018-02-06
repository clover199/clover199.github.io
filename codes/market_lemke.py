from __future__ import division, print_function, absolute_import
import numpy as np
from itertools import combinations
# import matplotlib.pyplot as plt

epsilon = 1e-8
np.set_printoptions(precision=4, suppress=True)

def FindNextVertex(M, b, v, e):
    """
        Given polyhedron defined by Mx<=b,
        find the next vertex along edge e from vertex v
    """

    # index of all planes cut edge e
    alpha = M.dot(e)
    index = np.argwhere( alpha>0 ).ravel()

    if index.shape[0]==0:
        return None

    beta = min( ( b[index] - M[index,:].dot(v) ) / alpha[index] )
    return v + beta*e


def FindEdges(M, b, v, verbose=False):
    """
        Find directions of complementary edges from given vertex v.
        Edges are labeled by index of relaxed constraints.
        input:  M:  numpy array of dimension (m, n)
                b:  numpy array of dimension (m,)
                v:  numpy array of dimension (n,)
        output: dirs:  dict
    """

    m, n = M.shape

    # index of tight inequalities of the vertex
    tight = np.argwhere( np.abs(M.dot(v)-b) < epsilon ).ravel()
    unq, cnt = np.unique(tight%n, return_counts=True)

    if verbose:
        temp = -np.ones([2,n])
        temp.ravel()[tight] = tight
        print("tight conditions of the vertex:\n", temp)

    if np.linalg.matrix_rank(M[tight,:])!=n:
        raise ValueError("Not a vertex")

    if unq.shape[0]==n:
        print("solution vertex", v)
        raise ValueError("This is a solution vertex")

    if unq.shape[0]!=n-1:
        raise ValueError("Vertex not complementary")

    dirs = {}
    for index in combinations(tight, n):
        unq, cnt = np.unique(np.array(index)%n, return_counts=True)
        if unq.shape[0]==n-1:    # should satisfy complementary conditions
            if np.abs(np.linalg.det(M[index,:]))>epsilon:    # should be linearly independent
                k = unq[np.where(cnt==2)][0]
                ds = -np.linalg.inv(M[index,:])
                dir1 = ds[:, list(index).index(k)]
                if all(M[tight,:].dot(dir1)<epsilon):    # should satisfy all inqualities
                    key = set(tight)-set(np.argwhere(np.abs(M.dot(dir1))<epsilon).ravel())
                    dirs[tuple(np.sort(list(key)))] = dir1
                dir2 = ds[:, list(index).index(k+n)]
                if all(M[tight,:].dot(dir2)<epsilon):    # should satisfy all inqualities
                    key = set(tight)-set(np.argwhere(np.abs(M.dot(dir2))<epsilon).ravel())
                    dirs[tuple(np.sort(list(key)))] = dir2
    if verbose:
        if tight.shape[0]>n:
            print("Degeneracy!")
        print("Directions:")
        for key, val in dirs.items():
            print(key, val)
    return dirs


def Next(M, b, v, verbose=False):
    """
        Find the next vertex of polyhedron Mv<=b from the current vertex v.
        k is index of edge pointing to current vertex v.
        input:  M:  numpy array of dimension (m, n)
                b:  numpy array of dimension (m,)
                v:  numpy array of dimension (n,)
        output: v:  numpy array of dimension (n,)
    """

    m, n = M.shape

    # find direction of the out-going edge
    dirs = FindEdges(M, b, v, verbose)

    # choose direction with smallest z value if degeneracy exists
    min_key = None
    min_val = epsilon
    for key, val in dirs.items():
        if val[-1]<min_val:
            min_key = key
            min_val = val[-1]
    if min_key is None:
        raise ValueError("Cannot find directions with non-positive z value")
    dire = dirs[min_key]
    if verbose:
        print("choose direction:", min_key, dire)

    return FindNextVertex(M, b, v, dire)


def solve(A, q, verbose=False):
    """
        This solves the market LCP.
        input:  A:  numpy array of dimension (n,n)
                q:  numpy array of dimension (n,)
                k:  integer from 0 to n-1, default to be 0
        output: y:  numpy array of dimension (n,)
                    return None if encounter a ray
    """
    n = A.shape[0]

    # Write all linear constraints with one matrix
    M = np.vstack([A, -np.identity(n)])[:-1,:]    # last one duplicates z>=0
    b = np.hstack([q, np.zeros(n-1)])

    # starting vertex
    y = np.zeros(n)
    y[n-1] = -np.min(q)

    count = 0    # store the number of pivotings
    while y[n-1]>epsilon:    # while z is not zero
        count += 1
        if verbose:
            print("\n----Current vertex----\n", y)
        y = Next(M, b, y, verbose)
        if y is None:
            return None
    print("\nNumber of pivotings:", count)
    return y[:n]


def ExchangeMarket(W, U, verbose=False):
    """
        Find market equilibrium of the exchange market.
        input:  W:  numpy array of dimension (m,n).
                U:  numpy array of dimension (m,n). Utility
        output: x:  numpy array of dimension (m,n). Allocation
                P:  numpy array of dimension (n,). Price
    """

    m, n = U.shape

    # rescale U so that S_j=1
    S = np.sum(W, axis=0)
    U = U * S
    W = W / S

    # form LCP with parameters v = {f_ij, p_j, lambda_i, z}
    A13 = np.zeros([m*n,m])
    for i in range(m):
        for j in range(n):
            A13[i*n+j][i] = U[i][j]
    A = np.vstack([np.hstack([np.zeros([m*n,m*n]), -np.tile(np.eye(n),[m,1]), A13, np.zeros([m*n,1])]), \
                   np.hstack([np.kron(np.ones(m), np.eye(n)), -np.eye(n), np.zeros([n,m]), np.zeros([n,1])]), \
                   np.hstack([-np.kron(np.eye(m), np.ones(n)), W, np.zeros([m,m]), -np.ones([m,1])]), \
                   np.hstack([np.zeros(m*n+m+n), -np.ones(1)])])
    q = np.hstack([np.ones(m*n+n), -np.sum(W, axis=1), np.zeros(1)])

    v = solve(A, q, verbose)
    if v is None:
        raise ValueError("secondary ray")

    # get x and P from v
    P = v[m*n:m*n+n]+1
    x = v[:m*n].reshape([m,n]) / P

    return x*S, P/S

def FisherMarket(M, U, S, verbose=False):
    """
        Find market equilibrium of the Fisher linear market.
        input:  M:  numpy array of dimension (m,). Budget
                U:  numpy array of dimension (m,n). Utility
                S:  numpy array of dimension (n,). Goods
        output: x:  numpy array of dimension (m,n). Allocation
                P:  numpy array of dimension (n,). Price
    """

    m, n = U.shape

    # rescale U so that S_j=1
    U = U * S

    # transform to exchange marekt
    W = np.tile(M.reshape([-1,1]), (1,n)) / np.sum(M)

    x, P = ExchangeMarket(W, U, verbose)
    P = P / np.sum(P) * np.sum(M)
    return x*S, P/S

if __name__=='__main__':
    print("\nTest Fisher Linear Market")
    M = np.array([10,10])
    U = np.array([[1,0],[2,1]])
    S = np.array([1,1])
    print("\nBudget M:\n", M)
    print("Utility matrix U:\n", U)
    print("Number of goods S:\n", S)
    x, p = FisherMarket(M, U, S, verbose=True)
    print("\nAllocation x:\n", x)
    print("Price P:\n", p)
