#from networkx import optimize_graph_edit_distance
import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objects as go
import plotly.subplots as ms
from datetime import date
from stock import StockTrader
from pathlib import Path  
from multiprocessing import Pool, Value
from prophet.plot import plot_plotly
from scipy import stats
import time

def resaltar(x):
    return ['background-color: lightgray']*13

def real_price(price, comission):
    return price * comission + price

# monitoring price of the symbol each timesleep(n)
# recomended to be ran with thread or multiprocessing
def run_monitoring(sym_object : StockTrader, monitor_time : int):
    counter = 0
    while(counter < monitor_time):
        df_finfo = sym_object.obtain_fast_info()
        counter += 1
        time.sleep(1)

@st.cache_data
def get_stock_list():
    try:
        df_simbolos = pd.read_csv("resources/symbollist3.csv")
        print("Obtenido:", df_simbolos.columns)
        #for col in df_simbolos.columns:
        #    print(col.)
        return df_simbolos[" Stock Symbol"].to_list()
    
    except Exception as e:
        print ("****************** ERROR *********************************")
        print(e)
        print("***********************************************")
        return  ['GOOG', 'AAPL', 'MSFT', 'GME', "TSLA"]

def get_stock_objectdict(stock_list):
    stock_dict = dict()
    for symbol in stock_list:
        stock_dict[symbol] = StockTrader(symbol=symbol)
    return stock_dict


START =  "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
STOCKS = get_stock_list()
BUDGET = 100000 # Note: modificar el budget
CURRENT_STOCK = 0
TOTAL_STOCK = 0


def process_stock(stock_it):
    try:
        #global CURRENT_STOCK
        #global TOTAL_STOCK
        #CURRENT_STOCK = CURRENT_STOCK + 1
        msg_txt = f"Generating table: {stock_it.symbol}"
        print(msg_txt)
        df_tmp = stock_it.get_stock_evaluation(100000, period="1y")
        df_tmp["Recommendation"] = stock_it.should_buy_sell(trend=True)
        #barraprogreso.progress(CURRENT_STOCK/TOTAL_STOCK, text=stock_it.name)
        return df_tmp
    except Exception as e:
        print(f"Error processing: {e}")
        return pd.DataFrame()

def activate_update(stock_sym):
    stock_dt[stock_sym].historic_data_period = None

def get_individual_recommentation(stoc_sym):
    barraprogreso = st.progress(0, text="Generating table")
    print("5y evaluation...")
    barraprogreso.progress(10, text="5y evaluation...")
    fiveyear = stock_dt[stoc_sym].get_stock_evaluation(BUDGET)
    if fiveyear.empty:
        #problem
        return
    barraprogreso.progress(33, text="5y evaluation...")
    print("1y evaluation...")
    barraprogreso.progress(43, text="1y evaluation...")
    oneyear  = stock_dt[stoc_sym].get_stock_evaluation(BUDGET, "1y")
    if oneyear.empty:
        return
    barraprogreso.progress(66, text="1y evaluation...")
    #difference = abs(fiveyear) - abs(oneyear)
    print("..DONE")
    st.title(stoc_sym)
    st.title(stock_dt[stoc_sym].name)

    ##### Prophet
    n_years = st.sidebar.slider('Years of prediction:', 1, 4)
    period = n_years * 365
    forecast = stock_dt[stoc_sym].get_prediction_prophet(period)

    if stock_dt[stoc_sym].model != None:
        fig1 = plot_plotly(stock_dt[stoc_sym].model, stock_dt[stoc_sym].forecast)
        #fig1.add_scatter(x=["ds"],y=df_test["y"], mode='lines')
        
        slope, intercept, r, p, std_err = stats.linregress(range(len(forecast)), forecast["trend"])
        st.write(f'Forecast plot for {n_years} years {slope}')
        # st.write(stock_dt[stoc_sym].)
    #####

    recom = stock_dt[stoc_sym].should_buy_sell(trend=True)

    barraprogreso.progress(80, text="Recommentation...")
    if recom=="Buy":
        st.write(f"Recommendation: :green[{recom}]")
    elif recom=="Sell":
        st.write(f"Recommendation: :red[{recom}]")
    else:
        st.write(f"Recommendation: :blue[{recom}]")
    st.metric(label="Current", value = stock_dt[stoc_sym].company_info["Close"].iloc[-1], delta =stock_dt[stoc_sym].company_info["Close"].iloc[-1]-stock_dt[stoc_sym].company_info["Close"].iloc[-2])#Delta=slope
    df_total = pd.concat([ fiveyear.iloc[:,3:], oneyear.iloc[:,3:],fiveyear.iloc[:,3:]-oneyear.iloc[:,3:]], axis=0)
    df_total = df_total.reset_index()
    df_total.style.apply(resaltar, axis=1, subset=df_total.index[-1])
    barraprogreso.progress(90, text="Predicting Future")
    st.write(df_total)
    if stock_dt[stoc_sym].model != None:
        st.plotly_chart(fig1)
        
    barraprogreso.empty()

def load_shares(filename):
    try:
        df_shares = pd.read_csv(filename)
        for el, data in df_shares.items():
            print(el)
            print(data[0])
        df_shares.set_index("date")
        return df_shares
    except Exception as e:
        print(f"Error en carga: {e}")
        return pd.DataFrame()

def get_recommendations(stock_dict):
    symbols = list(stock_dict.values())
    #print(symbols)
    df_total=0
    processed_sym= 0
    with Pool() as pool:
        dfs = pool.map(process_stock, symbols)
        processed_sym+=1
        print(processed_sym)
    df_total = pd.concat(dfs, ignore_index=True)
    return df_total

def get_recommendations2(stock_dt):
    symbols = list(stock_dt.values())
    with Pool() as pool:
        dfs = []
        with st.empty():  # Create an empty placeholder for the progress bar
            progress_bar = st.progress(0)  # Initialize progress bar
            for i, df in enumerate(pool.imap_unordered(process_stock, symbols)):
                dfs.append(df)
                # Update progress bar after each process completes
                if df.empty:
                    progress_bar.progress((i + 1) / len(symbols))
                else:
                    progress_bar.progress((i + 1) / len(symbols),text=df["symbol"].to_string()[1:])
    df_total = pd.concat(dfs, ignore_index=True)
    progress_bar.empty()
    return df_total

st.sidebar.title("Options")
option = st.sidebar.selectbox("Dashboard", ("Portfolio", "Buy & Sell", "Trading Bot"))
st.title(option)
stock_dt = get_stock_objectdict(STOCKS)

################# *****************   PORTFOLIO    ***************** #################
if option == "Portfolio":
    df_shares = load_shares("./resources/portfolio.csv")
    print(df_shares["symbol"][0])
    df_shares["price"] = pd.to_numeric(df_shares["price"])
    if(df_shares["symbol"][0] in stock_dt):
       dt_finfo = stock_dt[df_shares["symbol"][0]].obtain_fast_info()
       df_shares["bought price"] = real_price(df_shares["price"], df_shares["commision"]) #df_shares["price"] * df_shares["commision"] + df_shares["price"]
       df_shares["Total Investment"] = df_shares["bought price"] * df_shares["quantity"]
       df_shares["updated price"] = dt_finfo["lastPrice"]#to be updated evary 20 min
       df_shares["Balance"] = df_shares["quantity"]*(real_price(df_shares["updated price"], df_shares["commision"]) - real_price(df_shares["bought price"], df_shares["commision"]))
       st.write(df_shares)

################# *****************   BUY & SELL    ***************** #################
if option == "Buy & Sell":
    
    opt_mode_eval = st.sidebar.selectbox("Eval. Mode", ("Individual","All"))
    if opt_mode_eval == "Individual":
        opt_stock_item = st.sidebar.selectbox("Symbol", STOCKS)
        gen_eval_btn = st.sidebar.button("Generate Evaluation")
        if gen_eval_btn:
            gen_eval_btn = False
            activate_update(opt_stock_item)
        get_individual_recommentation(opt_stock_item)
            
    else:
        filter_opt = st.sidebar.button("Generate Evaluation")
        #to_buy_opt = st.sidebar.button("Only buy")
        filepath = Path('data/evaluatedstocklist.csv') 
        df_total =pd.DataFrame()
        if filter_opt:            

            #barraprogreso = st.progress(0, text="Generate table")
            TOTAL_STOCK = len(stock_dt)
            df_total = get_recommendations2(stock_dt)
            filter_opt = False
            filepath.parent.mkdir(parents=True, exist_ok=True)                
            #barraprogreso.empty()
            df_total.to_csv(filepath)
           
        else:
            try:
                df_total = pd.read_csv(filepath)
            except:
                st.write("Please generate file")
                df_total = pd.DataFrame()

        opt_buy_sell = st.sidebar.radio(
            "Filter Recomendation:",
            key="visibility",
            options=["All", "Buy", "Sell", "Nothing"],
        )
        st.write(opt_buy_sell)

        if opt_buy_sell!="All":
            if opt_buy_sell != "Nothing":
                df_total = df_total[df_total["Recommendation"]==opt_buy_sell]
            else:
                df_total = df_total[df_total["Recommendation"].isna()]

        
        st.write(df_total)
        
    