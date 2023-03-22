#run_tests.py
import scipy
import numpy as np
from CurveObj import CurveObj
from bondutils import *
import matplotlib.pyplot as plt
from test_unit import test_equal
terms_list = [1/365, 5/365, 10/365,1/12,3/12,6/12,9/12,1.,3.,5.,7.,10.,15.,20.,25.,30.]
rates_list = [0.025, 0.0252,0.02535,0.0255,0.0259, 0.0295, 0.0315,0.0334,0.0318,0.0308, 0.0305,0.0304,0.031, 0.0314,0.0305,0.0301]

target_terms_3 = np.sort(np.random.default_rng().uniform(0,30,100))

test_Curve_obj = CurveObj(terms=terms_list, rates=rates_list, name="Test 1", type_curve="zeros")
target_terms_smooth = np.arange(0,25,0.0001)
test_1 = test_Curve_obj.interpolateCurve(target_terms=[7/365, 15/365,1.5, 2.5, 3.5, 4.5, 6.5] ,type = "Cubic Spline", keep_og_terms=False)
test_2 = test_Curve_obj.interpolateCurve(target_terms=[7/365, 15/365,1.5, 2.5, 3.5, 4.5, 6.5] ,type = "linear", keep_og_terms=False)

test_3 = test_Curve_obj.interpolateCurve(target_terms=target_terms_smooth ,type = "Cubic Spline", keep_og_terms=False)
test_4 = test_Curve_obj.interpolateCurve(target_terms=target_terms_3 ,type = "Cubic Spline", keep_og_terms=False)
test_5 = test_Curve_obj.interpolateCurve(target_terms=target_terms_3, type = "linear", keep_og_terms=False)
#test_5 = test_Curve_obj.interpolateCurve(target_terms=[7/365, 15/365,1.5, 2.5, 3.5, 4.5, 6.5] ,type = "Cubic Spline", keep_og_terms=False)

# print("Test 1 Results {}".format(test_1))
# print("Test 2 Results {}".format(test_2))

# plt.plot(test_1["Terms"], test_1["Rates"], label="Cubic Spline")
# plt.plot(test_5["Terms"], test_5["Rates"], label="linear")
# #plt.plot(test_3["Terms"], test_3["Rates"], label="Cubic Spline #2")
# plt.plot(test_4["Terms"], test_4["Rates"],label="Cubic Spline #3")
# plt.plot(terms_list, rates_list,label="True Curve")
# plt.legend()
# plt.show()

# print("Test 3 Plot function Forward Curve")
# test_forward_1= test_Curve_obj.convert_to_forward_curve(target_terms = None,days_forward=1/65,copy=True)
# print(test_forward_1.name)
# print(test_forward_1.terms)
# print(test_forward_1.rates)
# test_forward_1.plotCurve()

# test bondutils.py functions.

term = 1
freq = 2 
Cashflow = 100 
r_1 =  0.05

test_7 = PVCF(term, Cashflow=Cashflow,df=r_1,freq=2)

term_array = np.arange(0,10+0.5,0.5)

test_8 = CustomCashflowPV(term_array,Cashflows=Cashflow,df=r_1, freq=freq)
print("Cashflow Amount_test7 is: {}".format(test_7))
print("Test 8 Results: ",test_8)


Principal = 10e6
#r_2 = [0.07 for _ in range(len(term_array))]
r_2 = np.random.normal(0.05, 0.0025, len(term_array))
df = [0.03 for _ in range(len(term_array))] # indexed to a seperate risk free rat
test_9 = non_amortizing_swap_val(Principal, term_array, r_1,r_2,df,freq=2)
print(f"Swap was valued at {test_9:0.2f}")

prinicpal_vector = np.random.normal(10000, 100, len(term_array))
prepay_vector = np.random.normal(200, 10, len(term_array))
liquidation = np.random.normal(10, 1, len(term_array))

test_10 = CSFixed_w_Prepay_LIQ(term_array, prinicpal_vector, prepay_vector, liquidation,0.05,"30/360", 2,df, 2)
test_10_output = test_10["cashflow"]
print("Present value test 10: " , test_10["Present Value"])
print(test_10_output.head())

def dee(x):
    return(2*x+1)
test_bisect = bisection_search(dee, -2, 2, 0)
test_bisect_2 = bisection_search(dee,0,2, 0)

print(test_equal("test 11", test_bisect, -0.5))
print(test_equal("test 12", test_bisect_2, "Method Failed"))

test_13 = CustomFixedDuration(term_array,Cashflows=Cashflow,df=r_1, freq=freq,steps=1e-5)
test_14 = CustomFixedConvexity(term_array,Cashflows=Cashflow,df=r_1, freq=freq,steps=1e-5)
print("Duration Custom Cashflow: ", test_13)
print("Convexity Custom Cashflow: ", test_14)


