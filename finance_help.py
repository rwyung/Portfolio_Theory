
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sample_mu = np.array([1.1, 1.2, 1.3 ])
# sample_sigma = np.matrix("0.01 0 0; 0 0.05 0; 0 0 0.07")
# r = 1.02
# t_low, thigh, tinc = 0 , 1 , 0.01

def checkdata(Sigma,tol):
    '''
    Check data will check whether Sigma is Positive Definite up to a certain threshold it will also
    Check for symmetry and thus will consequently check for inversability.
    '''
    Donald = True
    while Donald:
        if np.linalg.norm(Sigma - Sigma.T, 2) > tol:
            print("Error has Occured:  The Co-variance matrix is not symmetric")
            Donald = False
        if min(np.linalg.eig(Sigma)) < tol :
            print("Error has Occured: The Co-Variance Matrix is not Postive Definite")
            Donald = False
    return(Donald)
    
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
## For testing Purposes

# a0,a1,b0,b2,h0,h1 = EMVmodel(sample_mu, sample_sigma)
# a0,a1,b0,b2,h0,h1 = EMVmodel(sample_mu, sample_sigma)

def EMVmodelplot(tlow, thigh, tinc, alpha0, alpha1, beta0,beta2,show=True):
   # if isinstance(alpha0, "matrix):
    alpha0  = alpha0.item()
    #if isinstance(alpha1, "matrix"):
    alpha1 = alpha1.item() 
    beta0 = beta0.item()
    beta2 = beta2.item()
    t = np.arange(tlow, thigh,tinc)#.tolist() # if you want a list object
    mup = alpha0  + t * alpha1
    muplow = alpha0 - t * alpha1
    sigma2p = beta0 + np.square(t) * beta2
    plt.plot(sigma2p,mup, label = "Positive Portfolio")
    # plt.xlabel("Portfolio variance")
    # plt.ylabel("Portfolio Expected Return mu")
    plt.plot(sigma2p , muplow, label = "Negative Portfolio")
    # plt.title("Efficient Frontier: Mean - Variance Space")
    # 
    plt.legend() 

    # if show == True:
    #     plt.show()
    # else:
    #     pass
    # else:
    #     pass

def EFMSDplot(tlow,thigh,tinc,alpha0,alpha1, beta0, beta2):
    t = np.arange(start=tlow,step=tinc,stop=thigh)
    mup = alpha0 + t * alpha1
    muplow = alpha0 - t * alpha1
    stddevp = (beta0 + t**2 * beta2)**0.5
    plt.plot(stddevp, mup)
    plt.plot(stddevp, muplow)


def CMLplot(tlow,thigh,tinc,alpha0, alpha1, beta0,beta2,mu,Sigma,r):
    '''
    tlow is the lowest t risk tolerence, thigh is the hightest risk tolerence
    tinc is the increments to change risk tolerence, alpha0,alpha1,beta0,beta2 are components
        derived from the Efficient Frontier usually of Integers and Matrix Objects
    mu and Sigma are expected return arrays and the coveriance matrix of mixed assets  
    r is the risk free rate
    '''
    
    risk_free =  r + 1
    #self.plot_EMV(tstart=tlow,thigh=thigh,tinc=tinc)
    #EMVmodelplot(tlow, thigh, tinc, alpha0, alpha1, beta0,beta2,show=False)
    alpha0  = alpha0.item()
    alpha1 = alpha1.item() 
    beta0 = beta0.item()
    beta2 = beta2.item()
    EFMSDplot(tlow=tlow, thigh=thigh, tinc=tinc,alpha0=alpha0, alpha1=alpha1, beta0= beta0, beta2 = beta2  )
    n = np.shape(mu)[0]
    ell = np.ones(n)
    #ell_t = ell.T
    sigma_inv = np.linalg.inv(Sigma)
    print(sigma_inv)
    excess = (mu - risk_free * ell)
    print(excess)
    coeff = np.matmul(np.matmul(excess.T, sigma_inv),excess).item()
    print(coeff)
    coeffsqrt = coeff**0.5
    #lowerdf = np.matmul(np.matmul(ell,sigma_inv), ell_t)).item()
    tm = 1/(np.matmul(np.matmul(ell,sigma_inv), excess)).item()
    print(tm)
    xm = tm * np.matmul(sigma_inv,excess.T)
    t = np.arange(start=tlow, stop=tm,step=tinc**2)
    print(t)
    mup = risk_free + t * coeff
    print("mup is: ", mup)
    mup = np.asarray(mup).reshape(-1)
    sigmap = coeffsqrt * t
    print(sigmap)
    sigmap = np.asarray(sigmap).reshape(-1)
    print(np.shape(mup))
    print(np.shape(sigmap)) 
    
    plt.plot(sigmap,mup)
    ### Up end after intersection
    t =  np.arange(start=tm,stop=3*tm, step= tinc**2)
    mup = risk_free + t * coeff
    sigmap = t * coeffsqrt
    plt.plot(sigmap,mup)
    plt.xlabel = 'Portfolio Standard Deviation'
    plt.ylabel = 'Porfolio Mean'
    plt.show()