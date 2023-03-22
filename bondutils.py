import numpy as np
from CurveObj import CurveObj
import pandas as pd
import math


def PVCF(Term, Cashflow, df,freq):

    PV =  Cashflow * np.power((1+df * (1/freq)), (-Term *( 1/freq)))
    return(PV)

def CustomCashflowPV(Terms,Cashflows,df,freq, **kwargs):
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

    debug = kwargs["debug"] if "debug" in kwargs else False
    if Correct:
        if debug:
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

def CustomFixedCoupon(Terms,Prinicpal, fixed_rate,daycount,reset_freq, discount,valuation_term):
    
    #TODO:
    # Case 1 If valuation term is after terms only select Terms and prinicpal beyond that valuation term. Set opening balance to the sum of
    # the prinicpal payments up to that point
    # Case 2 all terms are less than valuation term. Simply return 0
    # Case 3 if valuation is before the start date of the cust
    if valuation_term < Terms[0]:
        pass
    elif valuation_term > Terms[-1]: 
        pass
    else:
        pass
         

def CSFixed_w_Prepay_LIQ(Terms, Principle, Prepayment,liquidation, fixed_rate, daycount, reset_freq, discount, valuation_term): 
    """
        CustomScheduleFixed: numpy.array, numpy.array, numpy.array, double, str, double, numpy.array, datetime.datetime) -> double
        CustomScheduleFixed consumes array of terms prinicipal and prepayment cashflows, daycount convention
            and the payment frequency, discount curve and valuation_data and returns the Present Value of the Cashflows  

        Note ensure that the first term has principle, prepayment, liquidation is set to 0 at term 0

        Prepayment, and liquidiation rates are not presented in percentages  like CPR or SMM.
    """
    
    
    #TODO change from valuation term to valuation date comparison
    # For now check conditions of valuation term going beyond last term in Terms

    if valuation_term > Terms[-1]:
        print('Valuation Term is beyond Maturity. Valuation returns 0')
        return(0)
    # TODO check if length fo terms is equal to Prinicpal Terms
    
    temp_total_prinicpal = Principle + Prepayment + liquidation 
    temp_total_prinicpal_schedule = temp_total_prinicpal.cumsum()

    # using numpy standard sum functions we can determine the opening balance of the instrument
    Opening_Balance = temp_total_prinicpal.sum()
    
    # This displays the opening balance for each period
    Opening_Balance_schedule = Opening_Balance - temp_total_prinicpal_schedule

    
    # TODO write conversions for daycount and freq and include accrued interest
    # Note my model assumes that the accrual will rely on the last coupon observed from the forward curve observed 
    # on the valaution_term date.
    # Since it is flat, the numbers by off compared to more conventional methodologies.
    # Things to include
    # 1.Accured in terest
    # 2.Proper consideration of the total Opening balance. Liquidation refers to defaulted assets which are subjected to recovery
    #   Thus the opening balance may not neccesarily be the sum of liquidation prepayment and scheduled principal. 
    #   Rather it is the expected sudden loss of the collateral which in turn  reduces the balance. 
    
    def check_daycount(day_count, reset_freq):
        # check type day_count
        if not(isinstance(day_count, str)):
            period  = 365/reset_freq
            out = (period, 1/365)
        else:
            day_count_temp = day_count.lower
            if day_count_temp == "act/365":
                period = 365/reset_freq
                out = (period,1/365)
            elif day_count_temp == "30/360":
                period = 360 / reset_freq
                out =  (period,1/360)
            elif day_count_temp == "30/365":
                period = 360/reset_freq
                out = (period, 1/365)
            else:
                period = 365.25 / reset_freq
                out = (period, 1/365.25)
        return out
    # extract day count convention    
    daycount_metric = check_daycount(day_count= daycount, reset_freq= reset_freq)
    
    Interest_schedule = Opening_Balance_schedule * fixed_rate * daycount_metric[0] * daycount_metric[1]
    
    Total_CF = Interest_schedule + Principle

    output = {"Terms": Terms, "Opening Balance": Opening_Balance_schedule,
              "Prinicpal": Principle,
                "Prepayment": Prepayment, "Liquidation": liquidation, "Interest": Interest_schedule}
    # Use custom cashflow to value the deal.
    PV = CustomCashflowPV(Terms,Total_CF,discount, reset_freq)
    
    

    return({"Present Value": PV, "cashflow":pd.DataFrame(output)})
#TODO include section to calculate all the prepayment and principal payments and interest cashflows and liquidation payments.


# Swaps section
def non_amortizing_swap_val(Principal,Terms, fixed_leg,fltleg,df,freq,**kwargs):
    """
        non_amortizing_swap_val: double, double, double, listofdoubles, freq, **[Cashflows, prinicpal exchange]
        non_amortizing_swap_val consumes Principal amount, list of terms,  fixed leg coupon and list of future float rate forward rates and retursn the PV of the swap
        non
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
    
    ## Fixed leg cashflows
    fixed_CF = Principal * np.array([fixed_leg for _ in range(len(Terms))])
    
    ## Float leg cashflows
    if isinstance(fltleg,(int, float)):
        float_CF = Principal * np.array([fltleg for _ in range(len(Terms))])
    elif isinstance(fltleg, list):
        float_CF = Principal * np.array(fltleg)
    else:
        float_CF = Principal * fltleg

    output["Fixed"] = fixed_CF
    output["Float"] = float_CF
    # TODO add Prinicpal Exchange Logic prinicpal_exchange
    # In general, if prinicpal is exchanged then we can simplify the result to a Bond for Bond transaction
    pay_fixed = kwargs["pay_fixed"] if "pay_fixed" in kwargs else True
    
    valuation = 0 
    fixed_result = CustomCashflowPV(Terms,fixed_CF,df,freq)
    float_result = CustomCashflowPV(Terms,float_CF,df,freq)
    if pay_fixed:   
        valuation = fixed_result - float_result

    else:
        valuation = -(fixed_result - float_result)
    
    return(valuation)

# Risk Measures and quick calculations.
# In the following section of code we will define common risk measures and calculations used in valuation.
# I.e Duration, WAL, Convexity, dollar convexity, Accrual Interest, Vega, Theta


# CUSTOM Model Risk metrics
def CustomFixedDuration(Terms,Cashflows,df,freq, steps=1e-5, **kwargs):
    df_up = df + steps
    df_down = df - steps

    P0 = CustomCashflowPV(Terms,Cashflows,df,freq)
    P_up = CustomCashflowPV(Terms,Cashflows,df_up,freq)
    P_down = CustomCashflowPV(Terms, Cashflows,df_down,freq)
    return((P_down - P_up)/ (2*steps*P0))

def CustomFixedConvexity(Terms,Cashflows,df,freq,steps=1e-5, **kwargs):
    df_up = df + steps
    df_down = df - steps

    P0 = CustomCashflowPV(Terms,Cashflows,df,freq)
    P_up = CustomCashflowPV(Terms,Cashflows,df_up,freq)
    P_down = CustomCashflowPV(Terms, Cashflows,df_down,freq)

    output = (P_up  + P_down - 2*P0)/ (2 * P0 * steps)
    return(output)

def DollarDuration(duration, PV):
    return(PV * duration)

def DollarConvexity(convexity, PV):
    return(PV * convexity)

def custom_price(Terms, Prini):
    pass
# Other Functionality and tools

def bisection_search(f,low, up, target, tol=1e-10, MAXIT=100000000):
    """
        bisection_search is function that consumes a function f, a lower bound low, upper bound up and target value and tolerance:
        bisection_search: function, float, float, float, float => float
        
        Consider f(x) = x + 2 Then
        bisection_serach(f,-5,5, 0) => 2
    """
    output = "Method Failed"
    N  = 1 
    mid_point = 0
    sign = lambda x: math.copysign(1,x)
    while(N <= MAXIT): 

        mid_point = (low + up)/2
        
        f_low = f(low)
        f_up = f(up)
        f_midpoint = f(mid_point)
        g = f_midpoint - target
        g_up = f_up - target
        g_low =f_low - target
        #print(f"f lower is  {f_low}")
        #print(f"f up is: {f_up}")
        #print(f"midpoint is: {mid_point}")

        if g == 0 or ((up - low)/2 <  tol):
            output = mid_point
            break
        else:
            N +=1
            if sign(g_low) == sign(g_up):
                break
            elif sign(g_low) == sign(g): 
                low = mid_point
            elif sign(g_up) == sign(g):
                up = mid_point
    print((output, f(mid_point)))
    return output          


        


    