#run_tests.py
import scipy
import numpy as np
from CurveObj import CurveObj
import matplotlib.pyplot as plt
terms_list = [1/365, 5/365, 10/365,1/12,3/12,6/12,9/12,1.,3.,5.,7.,10.,15.,20.,25.,30.]
rates_list = [0.025, 0.0252,0.02535,0.0255,0.0259, 0.0295, 0.0315,0.0334,0.0318,0.0308, 0.0305,0.0304,0.031, 0.0314,0.0305,0.0301]

test_Curve_obj = CurveObj(terms=terms_list, rates=rates_list, name="Test 1", type_curve="zeros")
target_terms_smooth = np.arange(0,25,0.0001)
test_1 = test_Curve_obj.interpolateCurve(target_terms=[7/365, 15/365,1.5, 2.5, 3.5, 4.5, 6.5] ,type = "Cubic Spline", keep_og_terms=False)
test_2 = test_Curve_obj.interpolateCurve(target_terms=[7/365, 15/365,1.5, 2.5, 3.5, 4.5, 6.5] ,type = "linear", keep_og_terms=False)

test_3 = test_Curve_obj.interpolateCurve(target_terms=target_terms_smooth ,type = "Cubic Spline", keep_og_terms=False)
#test_4 = test_Curve_obj.interpolateCurve(target_terms=[7/365, 15/365,1.5, 2.5, 3.5, 4.5, 6.5] ,type = "Cubic Spline", keep_og_terms=False)
#test_5 = test_Curve_obj.interpolateCurve(target_terms=[7/365, 15/365,1.5, 2.5, 3.5, 4.5, 6.5] ,type = "Cubic Spline", keep_og_terms=False)

print("Test 1 Results {}".format(test_1))
print("Test 2 Results {}".format(test_2))

plt.plot(test_1["Terms"], test_1["Rates"], label="Cubic Spline")
plt.plot(test_2["Terms"], test_2["Rates"], label="linear")
plt.plot(test_3["Terms"], test_3["Rates"], label="Cubic Spline #2")
plt.plot(terms_list, rates_list,label="True Curve")
plt.legend()
plt.show()

test_forward_1= test_Curve_obj.convert_to_forward_curve(target_terms = None,days_forward=1/65,copy=True)
print(test_forward_1.name)
print(test_forward_1.terms)
print(test_forward_1.rates)
test_forward_1.plotCurve()

