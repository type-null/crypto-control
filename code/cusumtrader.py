"""
Control chart based Trading strategies for crypto market.

Course project - Data Analytics and System Monitoring 2021 Fall

Team: Weihang, Babak
Date: 2021-12-09

"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from datetime import timedelta
from pmdarima.arima import auto_arima
from matplotlib.dates import DateFormatter
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from sklearn.linear_model import LinearRegression


class CusumTrader:
    """
    Add comments later ...
    """
    def __init__(self, coin='', doc=0, ra=False):
        """
        Current dataset includes 10 crypto coins. More coin data can be downloaded from CoinCompare using its API.
        Bitcoin, Ethereum, Solana, Ripple, Binance Coin, Tether, Cardano, Polkadot, Terra, Dogecoin.
        """
        warnings.filterwarnings('ignore')
        self.results = {'Return CUSUM No Fee':{}, 'Return CUSUM No Fee 46':{}, 'Return CUSUM':{}, 'Return CUSUM 46':{}, 
                'Residual CUSUM No Fee':{}, 'Residual CUSUM No Fee 46':{}, 'Residual CUSUM':{}, 'Residual CUSUM 46':{}, 
                'Return EWMA No Fee':{}, 'Return EWMA':{}, 'Residual EWMA No Fee':{}, 'Residual EWMA':{}, 'Benchmark':{},
                'Risk Adjusted CUSUM No Fee':{}, 'Risk Adjusted CUSUM':{}}
                
        self.funcHelp = """ Instructions:
        Use `.trade()` method to simulate all models at once.
        Use `.runModel(modelName)` method to run a specific model. 
            For example, `.runModel('Residual CUSUM No Fee', feature='residualFeature', method='montgomery', fee=0)`.
        
        modelName can be:
            'Return CUSUM No Fee', 'Return CUSUM No Fee 46', 'Return CUSUM', 'Return CUSUM 46', 'Residual CUSUM No Fee', 
            'Residual CUSUM No Fee 46', 'Residual CUSUM', 'Residual CUSUM 46', 'Return EWMA No Fee', 'Return EWMA',
            'Residual EWMA No Fee', 'Residual EWMA', 'Benchmark'
        You need to select proper `feature`, `method`, `fee` values for each model.

        Model's results are stored in a dictionary attribute `results`. You can access specific model's result by indexing.
            For example, `.results[model]['signal plot']` will show the price plot with buy and sell signals.
        Keys in `.results` includes:   
            'signals':          trading signal table for this model 
            'signal plot':      price plot with buy and sell signals
            'backtest ratios':  key metrics from backtest
            'backtest df':      raw result table after backtest
          For CUSUM models:
            'score':            k, h score table
            'score plot':       k, h score plots
            'k', 'h':           k, h values
            'CUSUM plot':       CUSUM plot which trading signals based on
          For EWMA models:
            'EWMA plot':        EWMA plot which trading signals based on
        """
        if doc:
            print(self.funcHelp)
        if ra:
            self.securities = pd.read_excel("../data/biostocks.xlsx", parse_dates=True, index_col="Dates")
            self.stocks = self.securities.loc[:, ['NBI Index', 'SPSIBI Index', 'MRNA US Equity']]
            self.traindf, self.testdf = self.trainTestSplit(self.stocks)
            #14 risk adjusted cusum no fee
            self.runModel('Risk Adjusted CUSUM No Fee', feature='regResidual', method='montgomery', fee=0)
            #15 risk adjusted cusum
            self.runModel('Risk Adjusted CUSUM', feature='regResidual', method='montgomery')
        else:
            self.data = pd.read_csv("../data/crypto_1y.csv", parse_dates=True, index_col="time")
            self.traindf, self.testdf = self.trainTestSplit(self.data)
            self.coin = coin
            #13 benchmark
            self.createBenchmark()

    def trade(self):
        # 1 return based cusum no fee
        self.runModel('Return CUSUM No Fee', feature='return', method='montgomery', fee=0)
        # 2 return based cusum no fee 46
        self.runModel('Return CUSUM No Fee 46', feature='return', method='46', fee=0)
        # 3 return based cusum
        self.runModel('Return CUSUM', feature='return', method='montgomery')
        # 4 return based cusum 46
        self.runModel('Return CUSUM 46', feature='return', method='46')
        # 5 residual based cusum no fee
        self.runModel('Residual CUSUM No Fee', feature='residualFeature', method='montgomery', fee=0)
        # 6 residual based cusum no fee 46
        self.runModel('Residual CUSUM No Fee 46', feature='residualFeature', method='46', fee=0)
        # 7 residual based cusum
        self.runModel('Residual CUSUM', feature='residualFeature', method='montgomery')
        # 8 residual based cusum 46
        self.runModel('Residual CUSUM 46', feature='residualFeature', method='46')
        # 9 return based EWMA no fee
        self.runModel('Return EWMA No Fee', feature='return', fee=0, ewma=True)
        #10 return based EWMA
        self.runModel('Return EWMA', feature='return', ewma=True)
        #11 residual based EWMA no fee
        self.runModel('Residual EWMA No Fee', feature='residualFeature', fee=0, ewma=True)
        #12 residual based EWMA
        self.runModel('Residual EWMA', feature='residualFeature', ewma=True)


    def summary(self):
        columns = ['Strategy', 'Start Time', 'End Time', 'Duration', 'Exposure Time', 
                    'Waiting Time', 'Sharpe Ratio', 'Sortino Ratio']
        self.ratiodf = pd.DataFrame(columns=columns)
        for (i, model) in enumerate(self.results.keys()):
            if self.results[model]:
                # not empty
                l1 = self.results[model]['backtest ratios'].copy()
                l1.insert(0, model)
                self.ratiodf.loc[i] = l1
        self.equityPlot = self.plotEquity()
        self.drawdownPlot = self.plotDrawdown()
        return self.ratiodf.sort_values(by='Sortino Ratio', ascending=False)


    def trainTestSplit(self, df):
        df = df.loc[df.index>='2021-05-01', :]
        return df.loc[df.index<'2021-09-01', :], df.loc[df.index>='2021-09-01', :]

    def createBenchmark(self):
        bm = self.testdf
        bm.loc[:, 'signal'] = 0
        bm.loc[bm.index[0], 'signal']=1
        bm.loc[bm.index[-1], 'signal']=-1
        bm.loc[:, 'share'] = 100 / bm.loc[bm.index[0], self.coin]
        bm.loc[:, 'equity'] = bm.loc[:, 'share'] * bm.loc[:, self.coin]
        self.results['Benchmark']['signals'] = bm
        # Backtest
        tempResult = self.backtest(self.results['Benchmark']['signals'])
        self.results['Benchmark']['backtest ratios'] = tempResult[0]
        self.results['Benchmark']['backtest df'] = tempResult[1]

    def calculateReturn(self, df):
        df.loc[:, "price"] = df.loc[:, self.coin]
        df.loc[:, "logPrice"] = np.log(df.loc[:, "price"])
        df.loc[:, "return"] = df.loc[:, "logPrice"] - df.loc[:, "logPrice"].shift(1)
    
        df.loc[df.index[0], "return"] = 0
        df.loc[df.index[0], 'signal'] = 0
        cols = ['price', 'logPrice', 'return', 'signal']
        return df.loc[:, cols]

    def calculateResidualFeature(self, df):
        std = np.std(df.loc[:, 'residual'])
        df.loc[:, 'residualFeature'] = df.loc[:, 'residual'] / std
        return df

    def getRegResidual(self, df):
        df.loc[:, "price"] = df.loc[:, 'MRNA US Equity']
        X = df.loc[:, ['NBI Index', 'SPSIBI Index']]
        y = df.loc[:, 'MRNA US Equity']
        reg = LinearRegression().fit(X, y)
        prediction = reg.predict(X)
        df.loc[:, 'regResidual'] = y - prediction
        return df

    def detectSellSignal(self, df, start, feature, method, k, h, fee):
        df.loc[df.index[start], 'cMinus'] = 0
        for i in range(start+1, len(df)):
            if method == "montgomery":
                cMinus = np.min([0, -df.loc[df.index[i-1], 'cMinus'] + df.loc[df.index[i], feature]+k])
            else:
                cMinus = np.min([0, df.loc[df.index[i-1], 'cMinus'] + df.loc[df.index[i], feature]-k])
            df.loc[df.index[i], 'cMinus'] = cMinus
        
            if cMinus <= -h:
                # sell signal appears
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'share'] = 0
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'share'] *  df.loc[df.index[i], 'price']
                df.loc[df.index[i], 'equityWithoutFee'] = df.loc[df.index[i], 'equity']
                return self.detectBuySignal(df, i, feature=feature, method=method, k=k, h=h, fee=fee)
            else:
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'share'] = df.loc[df.index[i-1], 'share']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'share'] * df.loc[df.index[i], 'price']
                df.loc[df.index[i], 'equityWithoutFee'] = df.loc[df.index[i], 'equity']
            
        return df

    def detectBuySignal(self, df, start, feature, method, k, h, fee):
        transactionFee = fee
        df.loc[df.index[start], "cPlus"] = 0
        for i in range(start+1, len(df)):
            cPlus = np.max([0, df.loc[df.index[i-1], 'cPlus'] + df.loc[df.index[i], feature]-k])
            df.loc[df.index[i], 'cPlus'] = cPlus
        
            if cPlus >= h:
                # buy signal appears
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity'] * (1-transactionFee)
                df.loc[df.index[i], 'equityWithoutFee'] = df.loc[df.index[i-1], 'equity']
                df.loc[df.index[i], 'share'] = df.loc[df.index[i], 'equity'] / df.loc[df.index[i], 'price']
                return self.detectSellSignal(df, i, feature=feature, method=method, k=k, h=h, fee=fee)
            else:
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'share'] = df.loc[df.index[i-1], 'share']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                df.loc[df.index[i], 'equityWithoutFee'] = df.loc[df.index[i], 'equity']
            
        return df

    def createTradeSignal(self, df, k, h, feature='return', method="montgomery", fee=0.5/100):
        df.loc[df.index[0], 'equity'] = 100
        df.loc[df.index[0], 'share'] = 0
        return self.detectBuySignal(df, start=0, feature=feature, method=method, k=k, h=h, fee=fee)

    def applyARIMA(self, df):
        df.loc[:, "price"] = df.loc[:, self.coin]
        autoARIMA = auto_arima(df['price'],
                           start_p=0, max_p=5,
                           d=0, max_d=2,
                           start_q=0, max_q=5)
        # print("Auto ARIMA parameters: ", autoARIMA.order)
        model = ARIMA(df['price'], order=autoARIMA.order)
        model_fit = model.fit()
        # print(model_fit.summary())
        df.loc[:, 'residual'] = model_fit.resid
        return df

    def getScore(self, feature='', method='montgomery', fee=0.5/100):
        kRange = np.arange(0.0002, 0.0023, step=0.0004)
        hRange = np.arange(0.0002, 0.0023, step=0.0004)

        if feature == "residualFeature":
            kRange = np.arange(0.02, 0.23, step=0.04)
            hRange = np.arange(0.2, 2.3, step=0.4)
            df = (self.traindf
                  .pipe(self.applyARIMA)
                  .pipe(self.calculateResidualFeature))
        elif feature == 'return':
            df = (self.traindf
                  .pipe(self.calculateReturn))
        elif feature == 'regResidual':
            df = (self.traindf
                  .pipe(self.getRegResidual))
        else:
            print("Feature not found!")
    
        outputdf = pd.DataFrame(list(product(kRange, hRange)), columns=['k', 'h'])
    
        for k in kRange:
            for h in hRange:
                trades = self.createTradeSignal(df, k=k, h=h, feature=feature, method=method, fee=fee)
                kh = (outputdf.loc[:, 'k']==k) & (outputdf.loc[:, 'h']==h)
                outputdf.loc[kh, 'finalEquity'] = trades.equity[-1]
            
        return outputdf.sort_values(by='finalEquity', ascending=False)

    def backtest(self, df):
        # Time
        startTime = df.index[0]
        endTime = df.index[-1]
        duration = endTime - startTime

        buyTime = df.loc[df['signal']==1, :].index
        sellTime = df.loc[df['signal']==-1, :].index
        numberOfCycles = np.min([len(sellTime), len(buyTime)])
        sellTimeEven = sellTime[:numberOfCycles]
        buyTimeEven = buyTime[:numberOfCycles]

        #we bought the coin and hold to find a sell signal
        exposureTime = np.sum(sellTimeEven-buyTimeEven) / numberOfCycles
        # waitingTime = ((buyTimeEven[0]-df.index[0]) 
                       # + np.sum(buyTimeEven[1:]-sellTime[:-1])) / numberOfCycles
        waitingTime = duration - exposureTime

        # Profit
        df.loc[:, 'logEquity'] = np.log(df.loc[:, 'equity'])
        df.loc[:, 'portReturn'] = df.loc[:, 'logEquity'] - df.loc[:, 'logEquity'].shift(1)
        downside = df.loc[:, 'portReturn']<0
        df.loc[downside, 'downsideReturn'] = -df.loc[downside, 'portReturn']
        df.loc[~downside, 'downsideReturn'] = 0

        # Risk
        pstd = df.loc[:, 'portReturn'].std()
        if pstd == 0:
            pstd = 1
        sharpeRatio = df.loc[:, 'portReturn'].mean() / pstd
        downstd = df.loc[:, 'downsideReturn'].std()
        if downstd == 0:
            sortinoRatio = df.loc[:, 'portReturn'].mean() / 1
        else:
            sortinoRatio = df.loc[:, 'portReturn'].mean() / df.loc[:, 'downsideReturn'].std()
        df.loc[:, 'drawdown'] = df.loc[:, 'equity'] - df.loc[:, 'equity'].cummax()
    
        return [startTime, endTime, duration, exposureTime, waitingTime, sharpeRatio, sortinoRatio], df


    def runModel(self, model, feature, method='montgomery', fee=0.5/100, ewma=False):
        print(f"Test model: {model} ...")
        if ewma:
            # EWMA
            if feature == 'residualFeature':
                self.results[model]['signals'] = (self.testdf.pipe(self.applyARIMA).pipe(self.calculateResidualFeature)
                    .pipe(self.createEWMATradeSignal, feature=feature, fee=fee))
            elif feature == 'return':
                self.results[model]['signals'] = (self.testdf.pipe(self.calculateReturn)
                    .pipe(self.createEWMATradeSignal, feature=feature, fee=fee))
            self.results[model]['EWMA plot'] = self.plotEWMA(self.results[model]['signals'])
        else:
            # Grid search for k,h
            self.results[model]['score'] = self.getScore(feature=feature, method=method, fee=fee)
            self.results[model]['score plot'] = self.plotScore(self.results[model]['score'])
            self.results[model]['k'] = self.results[model]['score'].iloc[0].k
            self.results[model]['h'] = self.results[model]['score'].iloc[0].h
            # Test on testdf
            if feature == "residualFeature":
                self.results[model]['signals'] = (self.testdf.pipe(self.applyARIMA).pipe(self.calculateResidualFeature)
                    .pipe(self.createTradeSignal, feature=feature, k=self.results[model]['k'], h=self.results[model]['h'], fee=fee, method=method))
            elif feature == 'return':
                self.results[model]['signals'] = (self.testdf.pipe(self.calculateReturn)
                    .pipe(self.createTradeSignal, feature=feature, k=self.results[model]['k'], h=self.results[model]['h'], fee=fee, method=method))
            elif feature == 'regResidual':
                self.results[model]['signals'] = (self.testdf.pipe(self.getRegResidual)
                    .pipe(self.createTradeSignal, feature=feature, k=self.results[model]['k'], h=self.results[model]['h'], fee=fee, method=method))
            self.results[model]['CUSUM plot'] = self.plotCUSUM(self.results[model]['signals'].head(200), h=self.results[model]['h'])
        self.results[model]['signal plot'] = self.plotPriceWithSignal(self.results[model]['signals'].head(200))
        # Backtest
        tempResult = self.backtest(self.results[model]['signals'])
        self.results[model]['backtest ratios'] = tempResult[0]
        self.results[model]['backtest df'] = tempResult[1]



    def detectEWMABuySignal(self, df, start, feature, mu, sigma, fee, lambdaa=0.3, L=3):
        transactionFee = fee
        df.loc[df.index[start],'z']=lambdaa*df.loc[df.index[start], feature] + (1-lambdaa)*mu
        df.loc[df.index[start],'ucl']=mu+(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*1))))
        df.loc[df.index[start],'lcl']=mu-(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*1))))
    
        for i in range(start+1, len(df)):
            
            z = lambdaa*df.loc[df.index[i], feature] +(1-lambdaa)*df.loc[df.index[i-1],'z']
            ucl = mu+(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*(i-start+1)))))
            lcl = mu-(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*(i-start+1)))))
            df.loc[df.index[i], 'z'] = z
            df.loc[df.index[i],'ucl']= ucl
            df.loc[df.index[i],'lcl']= lcl
        
            if z >= ucl:
                # buy
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity'] * (1-transactionFee)
                df.loc[df.index[i], 'share'] = df.loc[df.index[i], 'equity'] / df.loc[df.index[i], 'price']
                return self.detectEWMASellSignal(df, i+1, feature=feature, mu=mu, sigma=sigma, fee=fee)
            elif z <= lcl:
                # reset EWMA and hold
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'share'] = df.loc[df.index[i-1], 'share']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                return self.detectEWMABuySignal(df, i+1, feature=feature, mu=mu, sigma=sigma, fee=fee)
            else:
                #hold
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'share'] = df.loc[df.index[i-1], 'share']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']

        return df


    def detectEWMASellSignal(self, df, start, feature, mu, sigma, fee, lambdaa=0.3, L=3):
        df.loc[df.index[start],'z']=lambdaa*df.loc[df.index[start], feature] + (1-lambdaa)*mu
        df.loc[df.index[start],'ucl']=mu+(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*1))))
        df.loc[df.index[start],'lcl']=mu-(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*1))))
        
        for i in range(start+1, len(df)):
            
            z = lambdaa*df.loc[df.index[i], feature] +(1-lambdaa)*df.loc[df.index[i-1],'z']
            ucl = mu+(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*(i-start+1)))))
            lcl = mu-(L*sigma*np.sqrt(lambdaa/((2-lambdaa)*1)*(1-(1-lambdaa)**(2*(i-start+1)))))
            df.loc[df.index[i], 'z'] = z
            df.loc[df.index[i],'ucl']= ucl
            df.loc[df.index[i],'lcl']= lcl
        
            if z <= lcl:
                # sell
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'share'] = 0
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'share'] *  df.loc[df.index[i], 'price']
                return self.detectEWMABuySignal(df, i+1, feature=feature, mu=mu, sigma=sigma, fee=fee)
            elif z >= ucl:
                # reset EWMA and hold
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'share'] = df.loc[df.index[i-1], 'share']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'share'] * df.loc[df.index[i], 'price']
                return self.detectEWMASellSignal(df, i+1, feature=feature, mu=mu, sigma=sigma, fee=fee)
            else:
                #hold
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'share'] = df.loc[df.index[i-1], 'share']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'share'] * df.loc[df.index[i], 'price']

        return df


    def createEWMATradeSignal(self, df, feature='return', fee=0.5/100):
        df.loc[df.index[0], 'equity'] = 100
        df.loc[df.index[0], 'share'] = 0
        mu = np.mean(df.loc[:, feature])
        sigma = np.std(df.loc[:, feature])
        return self.detectEWMABuySignal(df, start=0, feature=feature, mu=mu, sigma=sigma, fee=fee)


    ### Plot functions
    def plotPriceWithSignal(self, df):
        fig = plt.figure(figsize=[16,9])
        plt.plot(df.index, df.loc[:, 'price'], linewidth=0.9)
        buy = df.loc[df['signal']==1, 'price']
        plt.scatter(buy.index, buy, c="g", marker="^", label="Buy signal")
        sell = df.loc[df['signal']==-1, 'price']
        plt.scatter(sell.index, sell, c="r", marker="v", label="Sell signal")
        plt.ylabel("Price")
        plt.xlabel("")
        plt.grid(alpha=0.4)
        plt.legend(loc='best')
        plt.close()
        return fig

    def plotCUSUM(self, df, h):
        fig = plt.figure(figsize=[16,9])
        plt.plot(df.index,df['cPlus'], linewidth=0.9, color='g', marker='h', markersize=6, label="Detecting buy signal")
        plt.plot(df.index,df['cMinus'], linewidth=0.9, color='r', marker='h', markersize=6, label="Detecting sell signal")
        plt.plot(df.index,np.array([h]*len(df.index)),color='b', label="Control limits")
        plt.plot(df.index,np.array([-h]*len(df.index)),color='b')
        plt.ylabel("CUSUM")
        plt.xlabel("")
        plt.grid(alpha=0.4)
        plt.legend(loc='best')
        plt.close()
        return fig

    def plotBeforeAndAfterARIMA(self, df):
        fig = plt.figure(figsize=[16,9])
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        autocorrelation_plot(df['price'], label="Price", ax=ax1)
        plt.grid(alpha=0.4)
        autocorrelation_plot(df['residual'], label="Residual", ax=ax2)
        plt.grid(alpha=0.4)
        plt.close()
        return fig

    def plotScore(self, df):
        fig = plt.figure(figsize=[16,9])
        plt.subplot(1,2,1)
        plt.scatter(df.loc[:,'k'], df.loc[:,'finalEquity'], label='k')
        plt.ylabel("Final Equity Value")
        plt.xlabel("$k$")
        plt.grid(alpha=0.4)
        plt.subplot(1,2,2)
        plt.scatter(df.loc[:,'h'], df.loc[:,'finalEquity'], label='h')
        plt.ylabel("Final Equity Value")
        plt.xlabel("$h$")
        plt.grid(alpha=0.4)
        plt.close()
        return fig

    def plotEWMA(self, df):
        fig = plt.figure(figsize=[16,9])
        plt.plot(df.index, df.z, label='EWMA of Return')
        plt.plot(df.index, df.ucl, c='g', label='Buy signal threshold')
        plt.plot(df.index, df.lcl, c='r', label='Sell signal threshold')
        plt.grid(alpha=0.4)
        plt.ylabel("EWMA")
        plt.legend(loc='best')
        plt.close()
        return fig

    def plotEquity(self):
        fig = plt.figure(figsize=[16,9])
        for model in self.results.keys():
            if self.results[model]:
                self.results[model]['backtest df'].loc[:,'equity'].plot(linewidth=0.9, label=model)
        plt.grid(alpha=0.4)
        plt.legend(loc='best')
        plt.ylabel('Equity')
        plt.close()
        return fig

    def plotDrawdown(self):
        fig = plt.figure(figsize=[16,9])
        for model in self.results.keys():
            if self.results[model]:
                self.results[model]['backtest df'].loc[:,'drawdown'].plot(linewidth=0.9, label=model)
        plt.grid(alpha=0.4)
        plt.legend(loc='best')
        plt.ylabel('Drawdown')
        plt.close()
        return fig