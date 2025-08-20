import time
import pandas as pd
from datetime import date
from sympy import true
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import plotly.subplots as ms
from scipy import stats
import numpy as np
from pathlib import Path  
import requests
import math as m
from termcolor import colored as cl

class StockTrader(object):
    def __init__(self, symbol:str) -> None:
        self.symbol = symbol
        self.last_intradays = pd.DataFrame()
        self.historic = pd.DataFrame()
        self.company_details = pd.DataFrame()
        self.company_info = pd.DataFrame()
        self.price = 0
        self.high_mean = 0
        self.low_mean = 0
        self.in_position = True
        self.KC_limits = []#middle, upper, lower
        self.KC_strategy_earnings = 0
        self.KC_strategy_roi = 0
        self.RSI_index = None
        self.RSI_strategy_earnings = 0
        self.RSI_strategy_roi = 0
        self.model = None
        self.forecast = pd.DataFrame()
        self.historic_data_period = None
        self.shares = []

    def obtain_intraday_data(self, fullsize=True):
        if fullsize:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.symbol}&interval=1min&outputsize=full&apikey=API_KEY'
        else:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.symbol}&interval=1min&apikey=API_KEY'
        r = requests.get(url)
        dict_stock = dict(r.json())
        df_intradays = pd.DataFrame.from_dict(dict_stock["Time Series (1min)"],orient="index")
        df_intradays.reset_index(inplace=True)
        df_intradays["1. open"] = df_intradays["1. open"].astype(float)
        df_intradays["2. high"] = df_intradays["2. high"].astype(float)
        df_intradays["3. low"] = df_intradays["3. low"].astype(float)
        df_intradays["4. close"] = df_intradays["4. close"].astype(float)
        df_intradays["5. volume"] = df_intradays["5. volume"].astype(float)
        self.last_intradays = df_intradays
    
    def obtain_historic_data(self , fechainicio:str , fechafin:str):
        try:
            # Obtener datos del simbolo desde Yahoo Finance
            datos = yf.download(self.symbol, start=fechainicio, end=fechafin)
            datos.reset_index(inplace=True) 
            self.historic = pd.DataFrame(datos)

        except Exception as e:
            print(f"Error to obtain historic data from {self.symbol}: {e}")

    def obtain_fast_info(self):
        try:
            finfo = yf.Ticker(self.symbol)
            return finfo.fast_info
        except Exception as e:
            print(f"Error to obtain historic data from {self.symbol}: {e}")
            return pd.DataFrame()
        
    def obtain_company_info(self, period="1y"):
        try:
            # if self.company_details.empty:
            #     company_details = yf.Ticker(self.symbol)
            #     self.company_details=company_details
            # else:
            #     company_details = self.company_details 
                
            #if self.historic_data_period == period:
            #    return True
            #if self.historic_data_period != None:
                # if self.historic_data_period == "5y":
                #     df_historial = self.company_info[-365:]
                # else:
                #     company_details = yf.Ticker(self.symbol)
                #     self.company_details = company_details
                #     df_historial = company_details.history(period=period)
            company_details = yf.Ticker(self.symbol)
            self.company_details = company_details
            df_historial = company_details.history(period=period)
            df_historial.reset_index(inplace=True)

            df_diff = np.array([df_historial["High"] - df_historial["Open"], df_historial["Low"] - df_historial["Open"]])
            hd = df_diff[0].mean()
            ld = df_diff[1].mean()
            #diferencia = hd - ld
            if "longName" in company_details.info:
                self.name = company_details.info['longName']
            else:
                self.name = "-----"

            self.historic_data_period = period
            self.company_info = df_historial
            
            self.high_mean = hd
            self.low_mean = ld
            self.price = df_historial["Close"]
            return True
        except Exception as e:
            print(f"Error to obtain company info from {self.symbol} in Yahoo")
            print(e)
            return False
        
    

    def get_ref_price(self):
        return self.high_mean-self.low_mean



    def is_good_candidate(self, desired_earnings:float, maxprice:float) -> bool:
        if (desired_earnings>= (self.high_mean-self.low_mean)):
            if (maxprice<= self.price):
                return True
        return False

    def buy_shares(self, quantity, price, date, comission):
        self.shares.append((quantity, price, date, comission))


    def should_buy_sell(self, price=0, trend=False) -> str:
        if trend:
            lastprice = self.company_info["Close"].iloc[-2]
            price = self.company_info["Close"].iloc[-1]
        else:
            lastprice =self.company_info["Close"].iloc[-1]
        #precio de ayer vs limites de ayer u precios de hoy vs limites de hoy
        if lastprice > self.KC_limits[2][0] and price < self.KC_limits[2][1] and self.KC_strategy_roi>0:
            return "Buy"
        elif lastprice < self.KC_limits[1][0] and price > self.KC_limits[1][1]:
            return "Sell"
        return "None"

    #simulates trader but using alphasignal 
    def simulate_trader(self):
        self.obtain_intraday_data(False)
        for index, row in self.last_intradays.iterrows():
            print(row)
            #print(self.last_intradays)
            
            time.sleep(1)
        pass
    


    def get_prediction_prophet(self, period=1):
            #df_train = data[['Date','Close']].iloc[:int(len(data)*0.95)]
            #df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
            #df_test = data[['Date','Close']].iloc[int(len(data)*0.95):]
            #df_test = df_test.rename(columns={"Date": "ds", "Close": "y"})
            if self.company_info.empty:
                print(self.company_info)
                return None
            
            data = self.company_info

            if not data.empty:
                #print(data)
                print("#################################")
                data["Date"] = data["Date"].dt.tz_localize(None)
                print (data)
                print("#################################")
                df_train = data[['Date','Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                #df_train['ds'].dt.tz_localize(None)
                #df_train['ds'].apply(lambda x: x.replace(tzinfo=None))
                df_test = data[['Date','Close']].iloc[int(len(data)*0.95):]
                df_test = df_test.rename(columns={"Date": "ds", "Close": "y"})
                #df_test['ds'].dt.tz_localize(None)
                #df_test['ds'].apply(lambda x: x.replace(tzinfo=None))
                #modelo de prophet
                model = Prophet()
                model.fit(df_train)
                future = model.make_future_dataframe(periods=period)
                future = future[future["ds"].dt.dayofweek < 5]
                forecast = model.predict(future)
                self.model = model
                self.forecast = forecast
                return forecast
            
    # Keltner Channel Index
    def get_kc(self, high, low, close, kc_lookback, multiplier, atr_lookback): 
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.ewm(alpha = 1/atr_lookback).mean()
        
        kc_middle = close.ewm(kc_lookback).mean()
        kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
        kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
        self.KC_limits = [kc_middle.iloc[-2:].to_list(), kc_upper.iloc[-2:].to_list(), kc_lower.iloc[-2:].to_list()]
        return kc_middle, kc_upper, kc_lower


    def run_kc_strategy(self, investment:float, debug=False): #Keltner Channel
        '''Docstring: Keltner Channel
        @investment is the quantity of money available for investment on the specific symbol
        @debug if true will print all transactions performed during the strategy
        return earnigs during evaluation, roi and the average of the buy/sell transactions in days'''
        #self.obtain_company_info("5y")
        df_symbol = self.company_info.iloc[:,:5]
        df_symbol['kc_middle'], df_symbol['kc_upper'], df_symbol['kc_lower'] = self.get_kc(df_symbol["High"], df_symbol["Low"], df_symbol["Close"], 20,2,10)
        #self.KC_limits = [df_symbol['kc_middle'].iloc[-1], df_symbol['kc_upper'].iloc[-1], df_symbol['kc_lower'].iloc[-1]]
        #print(df_symbol['kc_middle'].iloc[-1])
        in_position = False
        equity = investment
        
        daybought =0
        daysacum = 0
        ndays = 0
        transaction= 0
        for i in range(1, len(df_symbol)):
            if df_symbol['Close'][i-1] > df_symbol['kc_lower'][i-1] and df_symbol['Close'][i] < df_symbol['kc_lower'][i] and in_position == False:
                no_of_shares = m.floor(equity/df_symbol.Close[i])
                transaction = (no_of_shares * df_symbol.Close[i])
                equity -= transaction
                in_position = True
                if debug:
                    print(cl('BUY: ', color = 'green', attrs = ['bold']), f'{no_of_shares} Shares are bought at ${df_symbol.Close[i]} on {str(df_symbol.Date[i])[:10]}')
                daybought =df_symbol.Date[i]
            elif df_symbol['Close'][i-1] < df_symbol['kc_upper'][i-1] and df_symbol['Close'][i] > df_symbol['kc_upper'][i] and in_position == True:
                equity += (no_of_shares * df_symbol.Close[i])
                in_position = False
                if debug:
                    print(cl('SELL: ', color = 'red', attrs = ['bold']), f'{no_of_shares} Shares are sold at ${df_symbol.Close[i]} on {str(df_symbol.Date[i])[:10]}')
                daysacum+= abs((df_symbol.Date[i]-daybought).days)
                ndays += 1
        if in_position == True:
            equity+=transaction
            #equity += (no_of_shares * df_symbol.Close[i])
            #daysacum+= abs((df_symbol.Date[i]-daybought).days)
            #ndays += 1
            if debug:
                print(cl(f'\nClosing position at {df_symbol.Close[i]} on {str(df_symbol.Date[i])[:10]}', attrs = ['bold']))
            in_position = False
    
        earning = round(equity - investment, 2)
        roi = round(earning / investment * 100, 2)
        if ndays>0:#checar ndaya
            day_rate = daysacum/ndays
        else:
            day_rate = 1
        self.KC_strategy_earnings= earning
        self.KC_strategy_roi = roi    
        if debug:
            print('')
            print(cl(f'{self.symbol} BACKTESTING RESULTS:', attrs = ['bold']))
            print(cl(f'EARNING: ${earning} ; ROI: {roi}%', attrs = ['bold']))
            print(cl(f'rate in days: {day_rate}', attrs = ['bold']))
        return earning, roi, day_rate
    
    def get_RSI(self, all=True, period = 14):
        if all:
            delta = self.company_info["Close"].diff(1)
        else:
            delta = self.company_info["Close"].iloc[-(period+1):].diff(1)
        delta.dropna(inplace=True)
        positive = delta.copy()
        negative = delta.copy()
        positive[positive < 0] = 0
        negative[negative > 0] = 0
        average_gain = positive.rolling(window=period).mean()
        average_loss = abs(negative.rolling(window=period).mean())
        relative_strength = average_gain / average_loss
        RSI = 100.0 - (100.0 / (1.0+ relative_strength))
        self.RSI_index = RSI.iloc[-1]
        return RSI
    
    def run_RSI_strategy(self, investment, strategy_limit=10, debug=False):
        #self.obtain_company_info("5y")
        df_symbol = self.company_info.iloc[:,:5]
        in_position = False
        equity = investment

        RSI = self.get_RSI()
        daybought =0
        daysacum = 0
        ndays = 0   
        for i in range(1,len(RSI)):
            if  RSI[i] < strategy_limit and in_position == False:
                no_of_shares = m.floor(equity/df_symbol.Close[i])
                equity -= (no_of_shares * df_symbol.Close[i])
                in_position = True
                if debug:
                    print(cl('BUY: ', color = 'green', attrs = ['bold']), f'{no_of_shares} Shares are bought at ${df_symbol.Close[i]} on {str(df_symbol.Date[i])[:10]}')
                daybought =df_symbol.Date[i]
            elif RSI[i] > 100-strategy_limit and in_position == True:
                equity += (no_of_shares * df_symbol.Close[i])
                in_position = False
                if debug:
                    print(cl('SELL: ', color = 'red', attrs = ['bold']), f'{no_of_shares} Shares are bought at ${df_symbol.Close[i]} on {str(df_symbol.Date[i])[:10]}')
                daysacum+= abs((df_symbol.Date[i]-daybought).days)
                ndays += 1
        if in_position == True:
            equity += (no_of_shares * df_symbol.Close[i])
            daysacum+= abs((df_symbol.Date[i]-daybought).days)
            ndays += 1
            if debug:
                print(cl(f'\nClosing position at {df_symbol.Close[i]} on {str(df_symbol.Date[i])[:10]}', attrs = ['bold']))
            in_position = False

        earning = round(equity - investment, 2)
        roi = round(earning / investment * 100, 2)
        self.RSI_strategy_earnings = earning
        self.RSI_strategy_roi = roi
        if ndays >0:
            day_rate = daysacum/ndays
        else:
            day_rate = 1
        if debug:
            print('')
            print(cl(f'{self.symbol} BACKTESTING RESULTS:', attrs = ['bold']))
            print(cl(f'EARNING: ${earning} ; ROI: {roi}%', attrs = ['bold']))
            print(cl(f'DAY RATE: {day_rate}', attrs = ['bold']))
        return earning, roi, day_rate

    def get_stock_evaluation(self, budget, period="5y", debug=False):

        if not self.obtain_company_info(period):
            df_return = pd.DataFrame()
        kc_earning, kc_roi, kc_day = self.run_kc_strategy(budget, debug=debug)
        rsi_earning, rsi_roi, rsi_day = self.run_RSI_strategy(budget,debug=debug)#
        #date.today()df_return = pd.DataFrame({"date":'date', "symbol":'string' }, index=[0])
        df_return = pd.DataFrame({"symbol":'string' }, index=[0])
        df_return["symbol"] = self.symbol 
        #df_return["symbol"] = df_return["symbol"].astype(pd.StringDtype())
        print(self.symbol)
        df_return["name"] = self.name
        df_return["Close"] = self.company_info["Close"].iloc[-1]
        df_return["kc_earning"] =kc_earning
        df_return["kc_roi"] = kc_roi
        df_return["kc_day"] = kc_day
        df_return["kc_earnxday"] =  round(kc_earning/kc_day,2)
        
        df_return["kc_middle"] =  self.KC_limits[0][1]
        df_return["kc_upper"]  =  self.KC_limits[1][1]
        df_return["kc_lower"]  =  self.KC_limits[2][1]
        
        df_return["rsi_earning"] =rsi_earning
        df_return["rsi_roi"] = rsi_roi
        df_return["rsi_day"] = rsi_day
        df_return["rsi_earnxday"] =  round(rsi_earning/rsi_day,2)
        
        df_return["rsi_index"] =  self.RSI_index
        return df_return
    
if __name__ == "__main__":
    #demonio = StockTrader("BIMBOA.MX")
    #demonio = StockTrader("TSLA.MX")
    demonio = StockTrader("DELLC.MX")
    #demonio = StockTrader("GES")
    #demonio.obtain_company_info()
    #demonio.obtain_intraday_data(False)
    #demonio.simulate_trader()
    #print(type(demonio.last_intradays["1. open"][0]))
    #demonio.last_intradays["4. close"].plot()
    #demonio.obtain_company_info("5y")
    #demonio.run_kc_strategy(100000)
    #demonio.run_RSI_strategy(100000)

    #print(demonio.get_RSI(False).iloc[-1])
    #print(demonio.RSI_limits)
    #print(demonio.KC_limits)
    #demonio.load_shares("./resources/portfolio.csv")
    print(demonio.symbol)
    eval = demonio.get_stock_evaluation(100000,"5y", True)
    print(demonio.KC_limits)
    print(eval)
