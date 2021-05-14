# Convex Portfolio Optimization:

import numpy as np
from numpy.linalg.linalg import _qr_dispatcher
import pandas as pd
from numpy import matmul
from numpy.linalg import inv
from finance_help import * 
from utils import *
import scipy.linalg 
 
def prepare_risk(port):
    if port is None:
        print("Missing Portfolio")
    
    print("Preparing data for Cost Function...")
 
    e_return_normalized= np.power(port.expected_returns, 1/252) - 1
    out = e_return_normalized - port.indicators
    print("Preparation complete...")
    return(out.to_numpy() )

def solve_convex_constraint(port=None,constraint=None , content=None):
    if (constraint is None) or (constraint is None) or (content is None):
        print("You are missing one of the arguments")
        return(None)
    
    A1 = prepare_risk(port)
    n_constraint = constraint.shape[1]
    m_content  = len(content)
    
    #A1 = port.indicators.copy().to_numpy()
    
    if (m_content < n_constraint): #### Under-Determined Linear Systems
        Q , R = QR_Decomposition(constraint.T, mode="economic")
        
        #TODO correct the rank search inorder to speed up n flops .
        ### First Section solves linear constraints 
        
        Null_kernel = qr_null(constraint.T)
        #Rank_kernel = qr_rank(constraint)

        v  = matmul(inv(R.T),content)
        x_0 = matmul(Q, v)

        print("A Viable Solution is:", "\n", x_0)
        ### Second part constructs QR decomp solution for convex solutions
        
        Abar = A1 @ Null_kernel
        print("Estimated Abar matrix: \n", Abar)
        bbar = A1 @ x_0
        
        Qbar,Rbar= QR_Decomposition(Abar, mode="economic")
        # Rank_kernel_bar = qr_rank(A_bar)
        w_star = Qbar.T @ bbar
        x_star = x_0 + Null_kernel @ w_star
        port.allocation  = x_star 
        return(x_star)    
        
    elif (m_content > n_constraint): #### Over-Determined Linear Systems
        print("OverDetermined System Detected...\n")
        print("Using QR Decompostion for solving")
        Q,R = QR_Decomposition(constraint, mode="economic")
        
        #TODO correct the rank search inorder to speed up n flops .
        ### First Section solves linear constraints 
        
        Null_kernel = qr_null(constraint)
        #Rank_kernel = qr_rank(constraint)

        v  = matmul(Q.T,content)
        x_0 = matmul(inv(R), v)

        ### Second part constructs QR decomp solution for convex solutions
        Q_bar, R_bar = QR_Decomposition(port.T, mode="economic")
        b_bar = scipy.linalg.solve(R_bar.T, content.T)
        
        
        #Qbar,Rbar= QR_Decomposition(A_bar, mode="economic")
        # Rank_kernel_bar = qr_rank(A_bar)
        w_star =  matmul(Q_bar.T, b_bar)
        x_star = x_0 + matmul(Null_kernel, w_star)

        return(x_star)    
    else: #### Square Matrix 
        print("Square contrains detected using LU DECOMPOSITION \n")
        print("Starting computation")
        
def print_allocation(port):
    if port.allocation is None:
        print("Please run solve_convex_constraint, first to get allocation...")
    

    for equity in port.equities:
        ind = port.equities.index(equity)
        weight = port.allocation[ind]
        Money =  weight * port.notional
        print(f"{equity}: {weight:.4f} : {Money:.2f}")
        
        
            
        

