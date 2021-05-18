# Quant Tool Kit 

import pprint
from numpy.lib.npyio import ndfromtxt
import scipy
import scipy.linalg
import numpy as np
from math import exp, sqrt 
def LU_Decomposition(A):
    # Will find the LU decomposition of a square matrix
    P, L, U = scipy.linalg.lu(A)
    return(P, L , U)

def cholesky_Decomposition(A, lower=True):
    # Note that cholesky decomposition assumes matrix is hermitian and positive defintite
    # we can find the lower L triangler matrix.
    L = scipy.linalg.cholesky(A,lower=lower)
    return(L)

def QR_Decomposition(A,mode="full", pivoting=False):
    if pivoting != True:
        Q,R = scipy.linalg.qr(A, mode =mode , pivoting=pivoting)
        return(Q,R)
    else:
        Q,R,P = scipy.linalg.qr(A, mode =mode, pivoting=pivoting)
        return(Q,R,P)

def qr_rank(A, tol=None):
    Q,R,P = QR_Decomposition(A,pivoting=True)
    tol = np.max(A) * np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return (Q[:, :rnk].conj())
    

def qr_null(A, tol=None):
    Q,R,P = QR_Decomposition(A,pivoting=True)
    tol = np.max(A) * np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return (Q[:, rnk:].conj())

def mc_simulation(args):
    pass

# option tree takes in a starting stock price, strike price, time to expiry, risk free rate,
# volatility sigma, and the number of steps to price the option based on opttype
# option_tree(float32, float32,int,float32, float32, int, int) -> float32
# The code will compute the Blacck scholes ootion value using the binomial tree for european options 
# The code is based on the works found in CS 335 LEC 001 Fall 2018 by the univeristy of Waterloo
# Based on Instructor YuYing Li of Cheriton School of Computer Science in University of WAterloo.

def option_tree(S0=100,strike=None,Time=1, rf = 0.025, sigma =0.3, opttype=0, N_steps=10000):
    
    if strike == None:
        raise ValueError("Strike is missing and option cannot be evaluted")
    # Dividing number of steps  for tree pricing
    delt = Time/N_steps
    
    mu = exp(sigma * sqrt(delt))
    # down director
    d = 1/mu
    # updirector multiple
    a = exp(rf*delt)
    # From risk-free probability measure
    rf_prob = (a - d)/( mu - d )
    
    # First one must find the payoff at time of termination
    # Using Ito's Derivation to find the we can find the termination
    # value of the underyling.
    mu_array = np.array(list(range(0,N_steps)))
    d_array = np.array(list(range(N_steps,0,-1)))
    
    W = S0 *  d * d_array * mu* mu_array
    def payoff(S, K, opttype= 0):
        if(opttype ==0):
            W = max(S-K,0)  
        else:
            W = max(strike - S , 0)
    W = np.array([[payoff(S1,strike) for S1 in W]])
        
    ### Once we have obtained the payoffs of the options at termination
    ### we have to propogate backward to have the price today 
    ### This part will require a for loop but binary tree operations can 
    ### can be applied as well
    ### Apply Black Scholes SDE, we can find the risk free valuation at time 
    ### i in 1 to N_step at time = 1
    
    for i in range(N_steps,1, -1):
        # risk free growth rate rfgr
        rfgr =  exp(- rf * delt)
        
        top = W[2:i+1]
        bottom = W[1:i]
        
        W = rfgr * (rf_prob * top + (1 - rf_prob) * down)
        
    value = W[0]

    return(value)
        



    
