
from typing import OrderedDict
import collections
from scipy import interpolate
import numpy as np
import string 
import random
import matplotlib.pyplot as plt
class CurveObj:
    
    def __init__(self,terms,rates,name,type_curve):
        self.terms = terms
        self.rates = rates
        self.name = name
        self.type_curve = type_curve
    
    #TODO test and validate this function
    def order_terms(self):
        temp = self.terms
        # Works with numpy 
        if isinstance(self.terms, list):
            self.terms = temp.sort()
        elif isinstance(self.terms, dict):
            # Works with pandas dictionaries 
            self.terms = collections.OrderedDict(temp)
        
        print("Ordered Terms ... Complete!")
    
    def __add__(self, OtherCurve):
        # TODO if otherCurve is a constant, then apply parallel shift on curve
        # If OTher Curve is CursveObj
        #   Then:
        #       1.  If Curve is the same length then add parallel
        #       2.  If Curve has different term structure Then add the terms that match
        #           a. If needed make append function to include terms not on curve
        #           b. Return the shortest term structure list with added curves.
        
        lofrates = self.rates
        
        if isinstance(OtherCurve, list):
            pass
        else:
            self.rates = lofrates + OtherCurve 

        return(CurveObj(terms=self.terms,rates=self.rates, name=self.name))


    
    def interpolateCurve(self,target_terms: list ,type: str, keep_og_terms=False):
        if type == "Cubic Spline":
            interpolate_kind = "spline"
        elif type =="linear":
            interpolate_kind = "linear"
        elif type =="B Spline":
            interpolate_kind = "b-spline"
        else:
            print("Type is unknown: Defaulting to linear regression")
            interpolate_kind =type
        
        # get interpolation object
        if("linear" in interpolate_kind):
            f = interpolate.interp1d(self.terms, self.rates, kind = interpolate_kind)
        elif (interpolate_kind == "spline"):
            f = interpolate.CubicSpline(self.terms, self.rates, bc_type = 'natural')
        else:
            # to do add b sline
            pass
            
        
        interpolated_rates = f(target_terms)
        return({"Terms": target_terms, "Rates":  interpolated_rates, "Interpolator": f}) 

    
    def convert_to_forward_curve(self, target_terms = None,days_forward=1/65,copy=False):
        """
            convert_to_forward_curve(CurveObj, listofdoubles) :=> CurveObj(self, listofdoubles, listofdoubles, str ,str)
            convert_to_forward_curve takesa a CurveObj and a list of double data type terms and returns a forward curve for desired target_terms with days forward or 
            years forward equivalent.            
        """

        # TODO add unit test
        # 1. Take a zero cruve rates and linearly interpolate for desired target terms
        # 2. Once all target terms are find with interpolated zero rates, we then convert each 0 rate to respective rates
        # additional we can add a argument that changes frequency of the rates used./
        ######
        
        def ForwardRate(df1: float, df2: float):
            # Subject to change
            interpo_dict = self.interpolateCurve(target_terms =self.terms,type = "Cubic Spline", keep_og_terms=False)
            interpo_dict["Interpolator"]([df1,df2])
            result  = interpo_dict["Rates"][1] * ( df2 / (df2- df1)) - interpo_dict["Rates"][0] * (df1 / (df2 - df1))
            return result
        
        def codeGen(chars =string.ascii_uppercase + string.digits, N = 6):
            return ''.join(random.choice(chars) for _ in range(N))

        if target_terms is None:
            target_terms = self.terms[:-1]
        len_target_terms = len(target_terms)
        vector_of_ForwardRates = [0 for i in range(0, len_target_terms)]
        
        # Now we loop through the rates
        for j in range(0, len_target_terms):
            vector_of_ForwardRates[j] = ForwardRate(df1=days_forward, df2 = days_forward + target_terms[j])
        
        if copy:
            return(CurveObj(terms=target_terms, rates=vector_of_ForwardRates, name="Forward Curve "+(codeGen(N=6)),type_curve ="Forward Curve"))
        else: 
            return({"Terms": target_terms, "Rates": vector_of_ForwardRates})

    def convert_zero_to_forward(self,copy=False):
        """
            convert_zero_to_forward: CurveObj, Bool ->  CurbeObj
            convert_zero_to_forward helps convert zero  rates into a forward curve
        """

        if self.type_curve != "zero":
            return self

        terms = self.terms
        
        rates = self.rates
        
        
        
    def plotCurve(self):

        plt.plot(self.terms, self.rates, label= self.name)
        #plt.plot(test_2["Terms"], test_2["Rates"], label="linear")
        #plt.plot(test_3["Terms"], test_3["Rates"], label="Cubic Spline #2")
        #plt.plot(terms_list, rates_list,label="True Curve")
        plt.legend()
        plt.show()

                    
# def codeGen(chars =string.ascii_uppercase + string.digits, N = 6):
#     return ''.join(random.choice(chars) for _ in range(N))            
        
            
            
        



