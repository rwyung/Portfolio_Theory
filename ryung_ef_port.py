# Efficient Frontier tool Box

import yfinance as yf
import pandas as pd
import numpy as np
import csv
import time
from scipy import stats
from finance_help import EMVmodel, EMVmodelplot, CMLplot, checkdata
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# The Portfolio Optimizer Function attempts to return a .json or python dictionary
# that pertains to the weigthings to certain assets assuming provided there exists a solution.

# This is an applicationof the theoryies developed in Portfolio Optimization by
# Michael J. Best 
def extract_data(equities,length = "2y"):
    def label_data(df,equity = "Null"):
        df['ticker'] = equity
        # df['pct_change'] = df['close']
        return(df)
    tickers = [label_data(yf.Ticker(equity).history(length), equity) for equity in equities]
    data = pd.concat(tickers)
    data.to_csv("sample.csv")
    return(data)

def get_indicators(df, companies = None):
    # At the moment it will count the number of equities in the Portfolio Object and generate the daily returns of the asset
    # based off of the opening and closing stock prices.
    
    if companies is None:
        return(0)
    # if have more than 1 company in the portfolio then we can generate each companies returns and concat the pd.Series Objectes
    # at the end
    if type(companies) is list:
        num_comp = len(companies)
        output = [0 for _ in  range(num_comp)]
        i = 0
        for company in companies:
            temp = df[df['ticker']== company].copy()
            name_pct = "pct_change" + "_" + company
            temp[name_pct] =  (temp['Close'] - temp["Open"])/temp['Open']
            output[i] = temp[name_pct]
            i+=1
    # If the Portfolio only contains 1 company then it is simply finding 1 return and returning that data frame
    elif type(companies) is str:
        output  = [0]
        temp = df[df['ticker']== companies].copy()
        name_pct = "pct_change" + "_" + companies
        temp[name_pct] =  temp['close'].pct_change().copy()
        output[0] = temp[name_pct]
    else:
        print("Invalid Input: make sure list of companies is contained in a list")
        return(None)

    return(pd.concat(output, axis= 1))

# Covariance_matrix takes in a dataframe; supposedly of daily returns and attempts to find the covaraince matrix of the returns.           
def covariance_matrix(df): 
    #df =df.set_index('Date')
    covariance_matrix_df = df.cov()
    return(covariance_matrix_df)

class Portfolio:
    def __init__ (self, name, notional, equities, risk_free, sharp_ratio, beta):
        self.name = name
        self.notional = notional
        self.equities = equities
        self.risk_free = risk_free
        self.sharp_ratio = sharp_ratio
        self.beta = beta
        self.historical = extract_data(self.equities)
        self.indicators = get_indicators(self.historical,self.equities)    
        self.covariance_matrix_df = covariance_matrix(self.indicators)  
        self.expected_returns = None
        self.portfolio_variance = self.indicators.var()
        self.EMV_components = None
    #def print(self,i):
    def description(self):
        return f"{self.name} is holding with {self.notional} notional which has a {self.sharp_ratio} sharp ratio."

    def risk_level(self):
        return(f"{self.name} portfolio contains {self.equities} which may be risky")
        
    def __str__(self):
        return(f"{self.name} Portfolio")

    def add(self,equities):
        if type(self.equities) is list:
            self.equities.append(equities)
        else:
            self.equities = [self.equities].extend(equities)

    def extract_basic_equity(self, time_interval = "2y"):
        try:
            self.portfolio_numbers_df = extract_data(self.equities,length=time_interval)
            return(self.portfolio_numbers_df)
        #return(extract_data(self.equities,length=time_interval))
        except AttributeError:
            return("Error Occured during extraction")
    def covariance_matrix(self):
        # Covariance consumes Portfolio class structure and returns the covariance matrix
        # of the respective assets in the portfolio. At the moment one would have to create a large list
        # of Tickers  and take the daily covariance of the portfolio.
        temp = extract_data(self.equities)
        temp = get_indicators(temp, self.equities)
        temp = covariance_matrix(temp)
        self.covariance_matrix_df = temp
        del temp
        return(self.covariance_matrix_df)



    def expected_return(self, method="arthemtic"):
        # Expected_return is a Portfolio method that takes in an str argument method that will Consider the method of
        # calculating the mean return.
        # Expected_return Portfolio str :=> pandas.DataFrame
        if method =="arthemtic":
            # this method will generate each equities expected return based on the arthemtic mean of returns
            temp = extract_data(self.equities)
            temp = get_indicators(temp,self.equities)
            #temp = temp + 1
            mean_return = temp.mean()
            annualized = mean_return.apply(lambda x: x + 1)
            annualized = annualized.apply(lambda x: x**365)
            #annualized = annualized.apply(lambda x: x - 1)
            self.expected_returns = annualized.to_numpy()            
            return(self.expected_returns)
            
        elif method == "geometric":
            #Only works for periods where all equities 
            temp = extract_data(self.equities)
            temp = get_indicators(temp,self.equities)
            temp = temp.fillna(0)
            temp = temp + 1
            mean_return = stats.mstats.gmean(temp,axis=0)
            #mean_return = mean_return - 1
            mean_return = (mean_return ** 365)
            self.expected_returns = mean_return
            return(self.expected_returns)
            
            # #np.exp(np.log(df.prod(axis=1))/df.notna().sum(1))
            # mean_return = np.exp(np.log(temp.prod(axis=0))/temp.notna().sum(1))
            # annualized = mean_return + 1
            # annualized = annualized ** 12
            # self.expected_returns = annualized
            # return(self.expected_returns)
        else:
            pass

    def get_EMV_components(self):
        
        if self.expected_returns is None:
            self.expected_return(method="geometric")
            #("print Please run Portfolio.expected_return() first to initate expected returns")
        
        a0,a1,b0,b2,h0,h1 = EMVmodel(self.expected_returns,self.covariance_matrix_df.to_numpy())
        self.EMV_components = {'a0': a0, 'a1': a1, 'b0': b0, 'b2': b2, 'h0': h0, 'h1': h1}
        
        return(self.EMV_components)

    def plot_EMV(self, tstart = 0,thigh = 1, tinc = 0.01):
        if self.EMV_components is None:
            self.EMV_components = self.get_EMV_components
            try:
                EMVmodelplot(tlow=tstart, thigh= thigh, tinc=tinc, alpha0= self.EMV_components['a0'], alpha1=self.EMV_components['a1'],
                beta0= self.EMV_components['b0'], beta2= self.EMV_components['b2'])
            except Exception as e:
                print("Error has occured: {e}".format(e=e))
            else:
                print("Process has completed...")
        else:
            try:
                EMVmodelplot(tlow=tstart, thigh= thigh, tinc=tinc, alpha0= self.EMV_components['a0'], alpha1=self.EMV_components['a1'],
                beta0= self.EMV_components['b0'], beta2= self.EMV_components['b2'])
            except Exception as e:
                print("Error has occured: {e}".format(e=e))
            else:
                print("Process has completed...")

    def plot_returns(self, start="2019-01-01", end="2020-01-01"):
        if self.expected_returns is None:
            self.expected_return()
        temp = self.historical.reset_index()
        fig = go.Figure(data=go.Ohlc(x=temp['Date'],
                    open=temp['Open'],
                    high=temp['High'],
                    low=temp['Low'],
                    close=temp['Close']))
        fig.show()
        del temp


    def variance_portfolio(self, method = "annually"):
        ## variance portfolio consumes a portfolio class and a str called method and returns the variance of each resepctive 
        ## equity in the portfolio 
        ## variance 
        temp = extract_data(self.equities)
        temp = get_indicators(temp,self.equities)
        variance = temp.var().to_numpy()
        if method == "annually":
            self.variance = variance * 365
        elif method == "daily":
            self.variance = variance
        elif method == "weekly":
            self.variance = variance * 7
        else:
            self.variance = variance * 180
            
        return(self.variance)

    def plot_CML(self, tlow=0,thigh=1, tinc=0.01):
        if self.EMV_components is None:
            self.EMV_components = self.get_EMV_components
            self.expected_return()
            try:
                CMLplot(tlow=tlow, thigh= thigh, tinc=tinc, alpha0= self.EMV_components['a0'], alpha1=self.EMV_components['a1'], beta0= self.EMV_components['b0'], beta2= self.EMV_components['b2'],mu = self.expected_returns, Sigma = self.covariance_matrix_df, r = self.risk_free)
            except Exception as e:
                print("Error has occured: {e}".format(e=e))
            else:
                print("Process has completed...")
        else:
            try:
                CMLplot(tlow=tlow, thigh= thigh, tinc=tinc, alpha0= self.EMV_components['a0'], alpha1=self.EMV_components['a1'],
                beta0= self.EMV_components['b0'], beta2= self.EMV_components['b2'],mu = 1+ self.expected_returns,Sigma = self.covariance_matrix_df, r= self.risk_free)
            except Exception as e:
                print("Error has occured: {e}".format(e=e))
            else:
                print("Process has completed...")
    
# class equities:
#     def __init__():
#         super().__init__(self,nme,notional,equities,risk_free,sharp_ratio, beta)

################ Ending Prices for each equite.








#data = yf.Tickers("spce tsla spy")

### if raw data is desired then we simply use the information provided from the porfolio class #TODO implement class download to .csv
#data = yf.download("SPCE TSLA NVDA DDOG TD.to", period="2y",prepost=False, interval="1d")
#.to_csv("finance_data.csv")

#list(data.columns)