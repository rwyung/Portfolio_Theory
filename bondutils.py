import numpy as np
from CurveObj import CurveObj

def PVCF(Term, Cashflow, df,freq):

    PV =  Cashflow * np.power((1+df * (1/freq)), (-Term *( 1/freq)))
    return(PV)

def CustomCashflowPV(Terms,Cashflows,df,freq, *kwargs):
    # TODO 
    # 1. Check if Term length is equal to number of Cashflows:
    # 2. Check if df is constant or list or numpy array of discounting factors
    # 
    Correct = False
    if not(isinstance(Terms, (np.ndarray))):
        Terms = np.array(Terms)
    # Check if Cashflows are the Same length
    if isinstance(Cashflows, (float,int)):
            Cashflows = [Cashflows for _ in range(0,len(Terms))]
            Cashflows = np.array(Cashflows)
            Correct = True
    if len(Terms) == len(Cashflows):
        Correct = True
    else:        
        print("Number of Terms is not equal to number of Cashflows...")
        return({"Error": "4001","Description": {"NumTerms": len(Terms), "numCashflow": len(Cashflows)}})
    #  check if df is constant, if true make array full 
    if isinstance(df, (int, float)):
        discount_factor = [df for i in range(0, len(Terms))]
        discount_factor = np.array(df)
    else:
        discount_factor = np.array(df)    
    if Correct:
        print(Terms)
        print("------------------")
        print(Cashflows)
        print("------------------")
        print(discount_factor)
        print("------------------")
        print(freq)
        result = PVCF(Terms, Cashflows, discount_factor,freq)
        return(result.sum())
    else:
        return()




def swap_val(Principal,Terms, fixed_leg,fltleg,freq,*kwargs):
    """
        swap_val: double, double, double, listofdoubles, freq, **[Cashflows, prinicpal exchange]
        swap_val consumes Principal amount, list of terms,  fixed leg coupon and list of future float rate forward rates and retursn the PV of the swap
    """
    output = {"Fixed": 0, "Float": 0}

    Error = 0
    
    if Principal == 0:
        return()
    
    if len(Terms) != len(fltleg):
        Error =  1 
    
    ## TODO run interpoloation of fltleg to be able to avoid Error Code
    
    if Error:
        return("Error has Occured")
    
    else:
        fixed_rates = np.array([fixed_leg for _ in range(len(Terms))])
        
        
        output["Fixed"] = Fixed
        output["Float"] = Float 
    