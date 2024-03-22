import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import linregress
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

###macd
def macd(end_date,code):
    url = f'https://api.polygon.io/v1/indicators/macd/{code}?timestamp.lte={end_date}&timespan=day&adjusted=true&short_window=7&long_window=14&signal_window=9&series_type=close&order=asc&limit=5000&apiKey=RMnfdtr9nmyTjXjgbNJeX_I5pIcowZpl'
    r = requests.get(url)
    data = r.json()

    value = []
    signal = []
    histogram = []
    time = []

    for d in data['results']['values']:
        value.append(d['value'])
        signal.append(d['signal'])
        histogram.append(d['histogram'])
        time.append(datetime.fromtimestamp(int(str(d['timestamp'])[:-3])))

    df_macd = pd.DataFrame({'time': time, 'value': value, 'signal': signal, 'histogram': histogram})
    return df_macd

###stock

def stock(end_date,code):
    url = f'https://api.polygon.io/v2/aggs/ticker/{code}/range/1/day/2017-01-01/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey=RMnfdtr9nmyTjXjgbNJeX_I5pIcowZpl'
    r = requests.get(url)
    data = r.json()

    close = []
    high = []
    low = []
    time = []

    for d in data['results']:
        close.append(d['c'])
        high.append(d['h'])
        low.append(d['l'])
        time.append(datetime.fromtimestamp(int(str(d['t'])[:-3])))

    df_stock = pd.DataFrame({'time': time, 'high': high, 'low': low, 'close': close})
    return df_stock

###ema
def ema(end_date,code):
    url = f'https://api.polygon.io/v1/indicators/ema/{code}?timestamp.lte={end_date}&timespan=day&adjusted=true&window=50&series_type=close&order=asc&limit=5000&apiKey=RMnfdtr9nmyTjXjgbNJeX_I5pIcowZpl'
    r = requests.get(url)
    data = r.json()

    value = []
    time = []

    for d in data['results']['values']:
        value.append(d['value'])
        time.append(datetime.fromtimestamp(int(str(d['timestamp'])[:-3])))

    df_ema = pd.DataFrame({'time': time, 'value': value})
    return df_ema

###rsi
def rsi(end_date,code):
    url = f'https://api.polygon.io/v1/indicators/rsi/{code}?timestamp.lte={end_date}&timespan=day&adjusted=true&window=14&series_type=close&order=asc&limit=5000&apiKey=RMnfdtr9nmyTjXjgbNJeX_I5pIcowZpl'
    r = requests.get(url)
    data = r.json()

    value = []
    time = []

    for d in data['results']['values']:
        value.append(d['value'])
        time.append(datetime.fromtimestamp(int(str(d['timestamp'])[:-3])))

    df_rsi = pd.DataFrame({'time': time, 'value': value})
    return df_rsi



start_date = st.date_input("Choose a Start Date", value=None)
end_date = str(st.date_input("Choose a End Date", value = None))
code = st.text_input("Enter the stock code", value = None)


if 'start_date' in globals() and 'end_date' in globals():
    df_stock = stock(end_date,code)
    df_ema = ema(end_date,code)
    df_rsi = rsi(end_date,code)
    df_macd = macd(end_date,code)

    data_merge = pd.merge(df_stock,df_ema, 'right')
    data_merge['ema_yr'] = data_merge['value']
    data_merge = data_merge.drop(['value'], axis = 1)

    data_merge = pd.merge(data_merge,df_rsi, 'left', on = 'time')
    data_merge['rsi'] = data_merge['value']
    data_merge = data_merge.drop(['value'], axis = 1)

    df_final = pd.merge(data_merge,df_macd, 'left')
    df_final['macd'] = df_final['value']
    df_final = df_final.drop(['value'], axis = 1)

    df_final['ema_macd'] = df_final['macd']/df_final['ema_yr']

    df_final.time = [i.date() for i in df_final.time]
    df_final = df_final[df_final['time'] >= start_date].reset_index(drop = True)

    x1 = df_final.ema_yr.tolist()
    x2 = df_final.macd.tolist()

    length = df_final.shape[0] - 10

    ema_slope = [0]*10
    macd_slope = [0]*10

    for i in range(length):
        ema_slope.append(linregress([j for j in range(10)],x1[i:i+10]).slope)
        macd_slope.append(linregress([k for k in range(10)],x2[i:i+10]).slope)

    df_final['ema_slope'] = ema_slope
    df_final['macd_slope'] = macd_slope

    df_final['ema_macd_slope_ratio'] = df_final['macd_slope']/df_final['ema_slope']

    buy_index = []
    sell_index = []
    options = []
    options2 = []

    most_recent = 0
    for i,row in enumerate(df_final.iterrows()):
        #if the overall trend is positive
        if row[1]['ema_slope'] > 0:
            #and short term is also positive
            if row[1]['macd_slope'] > 0:
                #the macd/ema slope is going to be positive
                #the short and long term is both trending upwards
                #consider buying and sell when the ema/macd slope is higher than the current
                buy_indexx, result_index = next(((i, index+i) for index, value in enumerate(df_final.ema_macd[i:i+15]) if (value >row[1]['ema_macd'])and (index+i > most_recent) ), (None, None))
                buy_index.append(buy_indexx)
                sell_index.append(result_index)

                if result_index != None:
                    most_recent = result_index
                    options.append('option 1')
                    options2.append('option 1 T')
                else:
                    options.append(None)
                    options2.append('option 1 F')

            #if the short term trend is negative
            else:
                #the macd/ema slope is going to be negative
                #in the long run, it's going up, but in short run, it's going down
                #look at RSI to see if the stock is undervalued (less than 35) and if it is buy
                if row[1]['rsi'] <= 70:
                    buy_indexx, result_index = next(((i, index+i) for index, value in enumerate(df_final.ema_macd[i:i+15]) if (value >row[1]['ema_macd'])and (index+i > most_recent)), (None, None))
                    buy_index.append(buy_indexx)
                    sell_index.append(result_index)          
                    
                    if result_index != None:
                        most_recent = result_index
                        options.append('option 2')
                        options2.append('option 2 T')
                    else:
                        options.append(None)
                        options2.append('option 2 F')
                
                else: 
                    buy_index.append(None)
                    sell_index.append(None)
                    options2.append('option 2 F (RSI)')

        else: #if the overall trend is negative
            #but the short term trend is positive
            if row[1]['macd_slope'] > 0:
                #the macd/ema slope is going to be negative
                #in the long term perspective it's downward, but short term is positive
                #look at RSI to see if the stock is overvalued (over 70) and if it is, don't buy
                if row[1]['rsi'] < 75:
                    buy_indexx, result_index = next(((i, index+i) for index, value in enumerate(df_final.ema_macd[i:i+15]) if (value >row[1]['ema_macd'])and (index+i > most_recent)), (None, None))
                    buy_index.append(buy_indexx)
                    sell_index.append(result_index)          
                    
                    if result_index != None:
                        most_recent = result_index
                        options.append('option 3')
                        options2.append('option 3 T')
                    else:
                        options.append(None)
                        options2.append('option 3 F')
                
                else: 
                    buy_index.append(None)
                    sell_index.append(None)
                    options2.append('option 3 F (RSI)')
            
            #if the short term trend is also negative
            else:
                #the macd/ema slope is going to be positive
                #the short and long term is both trending downwards
                #consider buying and selling when the ema/macd slope is higher than the current (put option)
                #but for the sake of just stocks, don't buy
                buy_index.append(None)
                sell_index.append(None)        
                options.append(None)
                options2.append('option FF')    

    amount = []
    inputd = []
    t_f = []
    dates_bought = []
    dates_sold = []

    b = [i for i in buy_index if i is not None]
    s = [i for i in sell_index if i is not None]
    o = [i for i in options if i is not None]

    for i in range(len(b)):
        amount.append(df_final.close[s[i]] - df_final.close[b[i]])
        inputd.append(df_final.close[b[i]])
        dates_bought.append(df_final.time[b[i]])
        dates_sold.append(df_final.time[s[i]])
        if df_final.close[s[i]] - df_final.close[b[i]] >0:
            t_f.append(1)
        else:
            t_f.append(0)



    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x = df_final.time, y = df_final.close, name = 'close price'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x = df_final.time , y = df_final.ema_macd, name = 'ema macd ratio'),
        secondary_y=True,
    )

    # fig.add_trace(
    #     go.Scatter(x = df_final.time , y = df_final.ema_macd_slope_ratio, name = 'ema macd slope'),
    #     secondary_y=True,
    # )
    # for date in [i for i in dates_bought]:
    #     fig.add_vline(x=date, line_width=3, line_dash="dash", line_color="green")

    # for d in [i for i in dates_sold]:
    #     fig.add_vline(x=d, line_width=1, line_dash="dash", line_color="yellow")


    st.plotly_chart(fig,theme="streamlit", use_container_width=True)

    st.dataframe(pd.DataFrame({'option':o, 'amount':amount, 't_f': t_f}).groupby(['option','t_f'])['amount'].sum())
    st.info(f"Number of Transactions: {len(amount)}")
    st.info(f"Total Number of Trading Days: {df_final.shape[0]}")
    st.info(f"Total Profit: ${np.sum(amount)}")
    st.info(f"Total Invested: S{np.sum(inputd)}")
    st.info(f"Total Returns Pct: {round(np.sum(amount)/np.sum(inputd),4)}")
    st.info(f"Average Profit: {np.sum(amount)/len(amount)}")
    st.info(f"Number of Transactions Lost: {len(t_f)-sum(t_f)},{(len(t_f)-sum(t_f))/len(t_f)}")
    st.info(f"Number of Transactions Won: {sum(t_f)},{sum(t_f)/len(t_f)}")
    
    cnt = Counter()
    for word in options2:
        cnt[word] += 1
    
    st.info(cnt)