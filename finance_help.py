
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def EMVmodel(mu, sigma):
    n = np.shape(mu)[0]
    ell = np.ones(n)
    ellh = ell.T
    sigma_inv = np.linalg.inv(sigma)
    temp1 = np.matmul(sigma_inv, ellh)
    temp2 = np.matmul(temp1,ellh)
    h0 = (temp2.item()**-1) * temp1
    temp3 = np.matmul(sigma_inv , mu.T)
    #print(temp3)
    temp4 = np.matmul(ell,  temp3.T)
    h1 = np.subtract(temp3 , temp4.item() * h0)
    alpha0 = np.matmul(mu, h0.T)
    alpha1 = np.matmul(mu, h1.T)
    beta0 = np.matmul(np.matmul(h0,sigma),h0.T)
    beta2 = np.matmul(np.matmul(h1,sigma),h1.T)
    return(alpha0,alpha1,beta0,beta2,h0,h1)


def EMVmodelplot(tlow, thigh, tinc, alpha0, alpha1, beta0,beta2):
   # if isinstance(alpha0, "matrix):
    alpha0  = alpha0.item()
    #if isinstance(alpha1, "matrix"):
    alpha1 = alpha1.item() 
    beta0 = beta0.item()
    beta2 = beta2.item()
    t = np.arange(tlow, thigh,tinc)#.tolist() # if you want a list object
    mup = alpha0  + t * alpha1
    muplow = alpha0 - t * alpha1
    sigma2p = beta0 + t**2 * beta2
    plt.plot(sigma2p,mup, label = "Positive Portfolio")
    plt.xlabel("Portfolio variance")
    plt.ylabel("Portfolio Expected Return mu")
    plt.plot(sigma2p , muplow, label = "Negative Portfolio")
    plt.title("Efficient Frontier: Mean - Variance Space")
    plt.legend() 
    plt.show()