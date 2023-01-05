# Portfolio_Theory

## GOAL

The goal for the following scripts is to use historical data extracted from yahoo finance using the yfinance module and apply 
basic portfolio theory at a click of a button develop an efficient frontier curve based on the variance and expected return methodology

The techniques in hopes will also address things like gradient descient for quadractic programing functions

## TImeline

THis section outlines the time line of the project in which we will later base our valuations on

### phase 1 efficeint frontier(passive investing)

1. Ability to download data

2. Ability to format data to find covariance table perferably as a numpy array / matrix

3. Apply the Efficient frontier without the notion of a risk free asset only bounded by a finite n assets for a certain time period.

4. Include a risk free asset perferably one that can extracted via the yfinance or self defined

### Phase 2 Financial Analysis AutoMation()

1. Extract financial accounting data
2. Automate simple models like Gordan Growth 1 period models
3. Add scenerio Analysis aka changes in equity costs, sga growth rates, risk free rate measures, beta proxies, market proxies
4. Apply divided, FCFF, residual income , to netbook vlaue methods, and show the difference in evaluation method
    + the difference here is that we will default to None since if a company has no Dividends then we cant apply DCF models. 
5. NLP: Pass a report and extract using fuzzy logic to extract key metrics. etc... 
6. Get the script to also predict future sga, FCFF and rebalance portfolio that way.

### PHASE 3 Machine Learning (DEEP LEARNING Methods) 2023 project

1. Extract Tweets from stocktwits and twitter to develope an retail analyst approach.
2. Time series Predictor
3. Option checker-> checks the option chains for increased volume and points to how the market percieves future change.
