#from matplotlib import axes
import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import plotly.subplots as ms
from scipy import stats
import numpy as np
from pathlib import Path  
import requests

def get_intraday_data(symbol, size="full"):
    if size=="compact":
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey=OQ6R34VPKLUVCNBF'
    else:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&outputsize=full&apikey=OQ6R34VPKLUVCNBF'
    r = requests.get(url)
    dict_stock = dict(r.json())
    df_intradays = pd.DataFrame.from_dict(dict_stock["Time Series (1min)"],orient="index")
    df_intradays.reset_index(inplace=True)
    return df_intradays

def plot_raw_data(datos, man):
    #st.write(datos.keys())
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=datos['Date'], y=datos['Open'], name="abre"))
    fig.add_trace(go.Scatter(x=datos['Date'], y=datos['Close'], name="Cierra"))
    fig.layout.update(title_text="Desempeño", xaxis_rangeslider_visible=True)
    man.plotly_chart(fig)

def plot_velitas(simbolo,man):
    fig = go.Figure(go.Candlestick(
        x=simbolo.index,
        open=simbolo['Open'],
        high=simbolo['High'],
        low=simbolo['Low'],
        close=simbolo['Close']
    ))
    man.plotly_chart(fig)

def obtener_indice(indice , fechainicio , fechafin):
    try:
        # Obtener datos del simbolo desde Yahoo Finance
        datos = yf.download(indice, start=fechainicio, end=fechafin)
        datos.reset_index(inplace=True)
        #data.reset_index(inplace=True)       
        return pd.DataFrame(datos)

    except Exception as e:
        print(f"Error al obtener el índice S&P 500: {e}")
        return pd.DataFrame()

def get_forecast_trend(data):
    
    #entrenar con el 95%
    data["Date"].dt.tz_localize(None)
    df_train = data[['Date','Close']].iloc[:int(len(data)*0.95)]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_test = data[['Date','Close']].iloc[int(len(data)*0.95):]
    df_test = df_test.rename(columns={"Date": "ds", "Close": "y"})
    
    #modelo de prophet
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    future = future[future["ds"].dt.dayofweek < 5]
    forecast = m.predict(future)
    #forecast = forecast[forecast["ds"].dayofweek() < 5]
    #slope = m.params['k'][0]
    slope, intercept, r, p, std_err = stats.linregress(range(len(forecast)), forecast["trend"])
    return True
    if slope>0:
        return True
    else:
        return False
   

def filter_stocks(val_difference, list_of_stocks, max_price=100):#, barra):
    final_table = pd.DataFrame()
    selected_keys = ["longName", 'open','previousClose', 'dayLow', 'dayHigh', 'regularMarketOpen', 'regularMarketPreviousClose', 'regularMarketDayLow', 'regularMarketDayHigh']
    total_stocks = len(list_of_stocks)
    print(total_stocks)
    for i,symbol in enumerate(list_of_stocks):
        try:
            barraprogreso.progress(i/total_stocks, text="Filtering candidates")
            
            company_details = yf.Ticker(symbol)
            
            df_historial = company_details.history(period="1y")
            df_historial.reset_index(inplace=True)
            #is_trend_up = get_forecast_trend(df_historial)
            
            #consultar pag 12 de notas stock project
            # ninguna de estas 2 es correcta
            market_diff = company_details.info['regularMarketDayHigh'] - company_details.info['regularMarketDayLow']
            #diferencia = df_historial["High"].mean() - df_historial["Low"].mean()
            #debemos calcular normalizado
            df_diff = np.array([df_historial["High"] - df_historial["Open"], df_historial["Low"] - df_historial["Open"]])
            hd = df_diff[0].mean()
            ld = df_diff[1].mean()
            diferencia = hd - ld
            #is_trend_up = get_forecast_trend(df_historial)
            print("**********")
            print(symbol, " = ", market_diff, "->", diferencia)
            #print("Trend: ", is_trend_up)
            print("Max price: ",max_price, " vs. ", company_details.info['previousClose'])
            print("**********")
            if diferencia > val_difference and company_details.info['previousClose']<=max_price:# and is_trend_up:
                print("          ^^^ SELECTED ^^^ ")
                print("****************************************")
                table_info = {key: company_details.info[key] for key in selected_keys}
                df_new = pd.DataFrame(data=table_info,columns=selected_keys, index=[0])
                df_new["Diferencia"] = diferencia       
                if final_table.empty:
                    print(" >>>> NEW <<<< ")
                    final_table = df_new
                else:
                    final_table = pd.concat([final_table,df_new],ignore_index=True)
            final_table.reindex()
            final_table.sort_values(by="Diferencia")
        except Exception as e:
            print(e)
            pass

    return final_table


@st.cache_data
def get_stock_list():
    try:
        df_simbolos = pd.read_csv("resources/symbollist.csv")
        print(df_simbolos.columns)
        #for col in df_simbolos.columns:
        #    print(col.)
        return df_simbolos[" Stock Symbol"].to_list()
    except Exception as e:
        print(df_simbolos)
        return  ['GOOG', 'AAPL', 'MSFT', 'GME']

START =  "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
#STOCKS = ['GOOG', 'AAPL', 'MSFT', 'GME']
STOCKS = get_stock_list()




#st.title("Stock Prediction")
#st.write("Gráficas")
#st.set_page_config(layout="wide")
st.sidebar.title("Options")
option = st.sidebar.selectbox("Dashboard", ("Trading Bot", "Stock Prediction", "Social media", "Api"))
st.title(option)

if option=="Stock Prediction":
    selected_stock = st.sidebar.selectbox('Select dataset for prediction', STOCKS)
    opt_visibility = st.sidebar.selectbox("View", ("Company Info", "Prediction", "", ""))
    company_details = yf.Ticker(selected_stock)
    st.title(f"{company_details.info['longName']}")
    data= obtener_indice(selected_stock, START, TODAY)
    #st.dataframe(data.tail())
   
    if opt_visibility == "Company Info":
        col1, col2 = st.columns(2)
        selected_keys = ['open','previousClose', 'dayLow', 'dayHigh', 'regularMarketOpen', 'regularMarketPreviousClose', 'regularMarketDayLow', 'regularMarketDayHigh']
        table_info = {key: company_details.info[key] for key in selected_keys}

        historial = company_details.history(period='1d')
        print(type(historial))
        for key in historial.keys():
            print(key, historial[key].values)
        diferencia = historial["Close"].values[0] - historial["Open"].values[0]
        #col2.write(f"diferencia:{diferencia}")
        col2.metric(label="Current", value = historial["Close"].values[0], delta =diferencia)
        plot_raw_data(data,col2) 
        plot_velitas(data, col2)
        #col2.write(company_details.info)
       

        col1.table(pd.DataFrame.from_dict(table_info,orient="index"))
        col1.text(f"Regular Market Open: {company_details.info['regularMarketOpen']}")
        col1.text(f"Regular Market Previous Close: {company_details.info['regularMarketPreviousClose']}")
        col1.text(f"Regular Market Day Low: {company_details.info['regularMarketDayLow']}")
        col1.text(f"Regular Market Day High: {company_details.info['regularMarketDayHigh']}")


        col1.write(historial)
        
       
        df_intraday = get_intraday_data(selected_stock)
        col2.write(df_intraday)

    if opt_visibility == "Prediction":
        n_years = st.sidebar.slider('Years of prediction:', 1, 4)
        period = n_years * 365
        
        if not data.empty:
            # Predict forecast with Prophet.
            #train = df.iloc[:int(len(df)*0.8)]
            #test = df.iloc[int(len(df)*0.8):]
            df_train = data[['Date','Close']].iloc[:int(len(data))]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
            df_test = data[['Date','Close']].iloc[int(len(data)*0.95):]
            df_test = df_test.rename(columns={"Date": "ds", "Close": "y"})

            #modelo de prophet
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            future = future[future["ds"].dt.dayofweek < 5]
            forecast = m.predict(future)
            print(forecast)
            # Show and plot forecast
            st.subheader('Forecast data')
            st.write(forecast.tail())
            st.write(future.tail())
            #slope = m.params['k'][0]
            slope, intercept, r, p, std_err = stats.linregress(range(len(forecast)), forecast["trend"])
            st.write(f'Forecast plot for {n_years} years {slope}')
            fig1 = plot_plotly(m, forecast)
            #fig.add_scatter(x=df['Date'], y=df['AAPL.Low'], mode='lines')
            fig1.add_scatter(x=df_test["ds"],y=df_test["y"], mode='lines')
            st.plotly_chart(fig1)
            
            st.write("Forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

            fig3 = m.plot(forecast)
            st.write(fig3)

        else:
            st.image("https://media.canalnet.tv/2019/10/no-se-pudo-macri-974x550.jpg")

if option=="Trading Bot":
    filter_opt = st.sidebar.button("Get Symbols")
    #df_stock_candidates = pd.DataFrame()
    filepath = Path('data/filteredstocklist.csv')  
    if filter_opt:
        barraprogreso=st.progress(0, text="Filtering candidates")
        df_stock_candidates = filter_stocks(1.25,STOCKS)#, barraprogreso)
        df_stock_candidates.reindex()
        filter_opt = False
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df_stock_candidates.to_csv(filepath)
        barraprogreso.empty() 
    else:
        try:
            df_stock_candidates = pd.read_csv(filepath)
        except:
            st.write("Please generate file")
            df_stock_candidates = pd.DataFrame()
    
    st.write(df_stock_candidates)
        #else:
        #    company_details = pd.concat([company_details,pd.DataFrame(yf.Ticker(symbol))], axis=0)
if option=="Api":
    df_day = get_intraday_data("GOOGL")

    fig = ms.make_subplots(rows=2,
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x = df_day.index,
        low = df_day['3. low'],
        high = df_day['2. high'],
        close = df_day['4. close'],
        open = df_day['1. open'],
        increasing_line_color = 'green',
        decreasing_line_color = 'red'),
        row=1,
        col=1)

    #Add Volume Chart to Row 2 of subplot
    fig.add_trace(go.Bar(x=df_day.index,
        y=df_day['5. volume']),
        row=2,
        col=1)

    #Update Price Figure layout
    fig.update_layout(title = 'Interactive CandleStick & Volume Chart',
        yaxis1_title = 'Stock Price ($)',
        yaxis2_title = 'Volume (M)',
        xaxis2_title = 'Time',
        xaxis1_rangeslider_visible = False,
        xaxis2_rangeslider_visible = True)
    st.plotly_chart(fig)