import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime

class Oilsensibiltiy(QCAlgorithm):

    def Initialize(self):
        
        self.SetStartDate( 2002 , 1, 1)       # Set Start Date
        self.SetEndDate( 2020 , 10, 10)
        self.SetCash(100000)                # Set Strategy Cash

        self.lookback = 61                 # Length(days) of historical data
        self.weights_long,self.weights_short = pd.DataFrame(),pd.DataFrame()      # Pandas data frame (index: symbol) that stores the weight
        self.Portfolio.MarginModel = PatternDayTradingMarginModel()
        self.AGG = self.AddEquity("AGG", Resolution.Daily).Symbol
        self.nextLiquidate = self.Time   # Initialize last trade time
        self.rebalance_days = 30
        
        self.UniverseSettings.Resolution = Resolution.Daily   # Use hour resolution for speed
        self.oil = self.AddData(QuandlOil, 'FRED/DCOILBRENTEU', Resolution.Daily).Symbol
        self.AddUniverse(self.CoarseSelection, self.SelectFine)
        self.selectedequity = 500
        self.numberOfSymbolsFine = 50
        self.Symbols_long = []
        self.Symbols_short = []
        self.zscore_keep_buy = []
        self.zscore_keep_short = []
        self.weights_long = []
        self.weights_short = []

    def CoarseSelection(self, coarse):
        
        if self.Time < self.nextLiquidate:
            return Universe.Unchanged
        
        selected = sorted([x for x in coarse if x.HasFundamentalData and x.Price > 5],
                          key=lambda x: x.DollarVolume, reverse=True)

        symbols = [x.Symbol for x in selected[:self.selectedequity ] ]

        return symbols
        
    def SelectFine(self, fine):
        
        filtered = [x.Symbol for x in fine if  x.AssetClassification.MorningstarSectorCode == 309]
        
        self.Symbols_long = filtered[:self.numberOfSymbolsFine]
        
        self.Symbols_short = filtered[-self.numberOfSymbolsFine:]
        
        return self.Symbols_long + self.Symbols_short

    def GetWeights_Buy(self, history , crudeoil_history):
 
        crudeoil_history = np.log(crudeoil_history/crudeoil_history.shift(1)).dropna()
        
        history = history.dropna(axis=1)
        
        sample = np.log(history/history.shift(1)).dropna()
        
        crudeoil_history.index = sample.index
        
        zscore = self.ZscoreGrade(sample,crudeoil_history)
        
        zscore_buy = zscore[zscore>1.25].dropna(axis=1)
            
        zscore_keep = zscore[zscore>0.50].dropna(axis=1)
        
        L = len(zscore_buy.columns)
        
        try :
            weights = (zscore_buy * (1 / L)/zscore_buy).iloc[0,:].sort_values()
        except:
            weights = pd.DataFrame()
        
        return weights,zscore_keep,L
     
        
    def GetWeights_Sell(self, history , crudeoil_history,L):
        
        crudeoil_history = np.log(crudeoil_history/crudeoil_history.shift(1)).dropna()
        
        history = history.dropna(axis=1)
        
        sample = np.log(history/history.shift(1)).dropna()
        
        crudeoil_history.index = sample.index
        
        zscore = self.ZscoreGrade(sample,crudeoil_history)
        
        zscore_short = zscore[zscore<-1.25].dropna(axis=1)
            
        zscore_keep = zscore[zscore<-0.50].dropna(axis=1)
        
        try :
            weights = (zscore_short * (-1 / L)/zscore_short).iloc[0,:][:L]
        except:
            weights = pd.DataFrame()
        
        return weights,zscore_keep
        
        
    def ZscoreGrade(self,sample, factors) :
        
        factors = sm.add_constant(factors)
        
        # Train Ordinary Least Squares linear model for each stock
        
        OLSmodels = {ticker: sm.OLS(sample[ticker], factors).fit() for ticker in sample.columns}
        
        # Get the residuals from the linear regression after PCA for each stoc
        
        resids = pd.DataFrame({ticker: model.resid for ticker, model in OLSmodels.items()})
        
        #Get the OU parameters 
        
        shifted_residuals = resids.cumsum().iloc[1:,:]
        
        resids = resids.cumsum().iloc[:-1,:]
        
        resids.index = shifted_residuals.index
        
        OLSmodels2 = {ticker: sm.OLS(resids[ticker],sm.add_constant(shifted_residuals[ticker])).fit() for ticker in resids.columns} 
        
        # Get the new residuals
        
        resids2 = pd.DataFrame({ticker: model.resid for ticker, model in OLSmodels2.items()})
        
        # Get the mean reversion parameters 
        
        a = pd.DataFrame({ticker : model.params[0] for ticker , model in OLSmodels2.items()},index=["a"])
    
        b = pd.DataFrame({ticker: model.params[1] for ticker , model in OLSmodels2.items()},index=["a"])
        
        e = (resids2.std())/(252**(-1/2))
    
        k = -np.log(b) * 252
        
        #Get the z-score
        var = (e**2 /(2 * k) )*(1 - np.exp(-2 * k * 252))
    
        num = -a * np.sqrt(1 - b**2)
    
        den = ( 1-b ) * np.sqrt( var )
    
        m  = ( a / ( 1 - b ) )
    
        zscores= num / den # zscores of the most recent day
    
        return zscores
        
    def OnData(self, data):
        
        history_long = self.History(self.Symbols_long, self.lookback, Resolution.Daily).close.unstack(level=0)
        
        new_look_back  = len(history_long)
        
        crudeoil_history = self.History(QuandlOil,self.oil , 300, Resolution.Daily).droplevel(level=0)
        
        crudeoil_history = crudeoil_history[~crudeoil_history.index.duplicated(keep='last')].iloc[-new_look_back:]

        self.weights_long,self.zscore_keep_buy,L = self.GetWeights_Buy(history_long,crudeoil_history)
        
        #history_short = self.History(self.Symbols_long, self.lookback, Resolution.Daily).close.unstack(level=0)
        
        self.weights_short,self.zscore_keep_short = self.GetWeights_Sell(history_long,crudeoil_history,L)
        self.Debug(self.weights_short)
        
        
        for holding in self.Portfolio.Values:
            if holding.Symbol in self.zscore_keep_short.index or holding.Symbol in self.zscore_keep_buy.index or holding.Symbol == self.AGG :
                continue
            if holding.Invested:
                self.Liquidate(holding.Symbol)
                
        for symbol, weight in self.weights_short.items():
            self.Debug(symbol)
            self.SetHoldings(symbol,0.75*weight)

        for symbol, weight in self.weights_long.items():
            self.SetHoldings(symbol,0.75*weight)
        
        if self.Time < self.nextLiquidate:
            return 
        
        self.SetHoldings('AGG', 0.70 )
        
        self.nextLiquidate = self.Time + timedelta(self.rebalance_days)

    def OnSecuritiesChanged(self, changes):

        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol, 'Removed from Universe')
        
class QuandlOil(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = 'Value'
