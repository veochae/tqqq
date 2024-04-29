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
import matplotlib.pyplot as plt


######################################################################
######################################################################
#### INTERNAL FUNCTIONS
######################################################################
######################################################################

#data collection function

def data_collection(load_ticker, load_interval, load_time_unit, load_start_date, load_end_date, polygon_api_key):
    #Load data from Polygon.io
    url = f'https://api.polygon.io/v2/aggs/ticker/{load_ticker}/range/{load_interval}/{load_time_unit}/{load_start_date}/{load_end_date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}'
    r = requests.get(url)
    data = r.json()

    close = []
    time = []

    while 'next_url' in data.keys():
        for d in data['results']:
            close.append(d['c'])
            time.append(datetime.fromtimestamp(int(str(d['t'])[:-3])))

        url = data['next_url']+f"&apiKey={polygon_api_key}"
        r = requests.get(url)
        data = r.json()

    for d in data['results']:
        close.append(d['c'])
        time.append(datetime.fromtimestamp(int(str(d['t'])[:-3])))

    dff = pd.DataFrame({'time': time, 'close': close});

    dff['time'] = pd.to_datetime(dff['time'])

    return dff

#trade hour check function (internal)
def check_trading_hours(dte):
    if dte.hour >= 9 and dte.hour < 16:  # Checking if the hour is between 9 and 16 (4 PM)
        if dte.hour == 9:
            if dte.minute > 30:  # If hour is 9, check if minute is > 30
                return True
            else:
                return False
        else:
            return True
    elif dte.hour == 16 and dte.minute == 0:
        return True
    else:
        return False
    
#trade hour check function (external)
    
def check_trading_hour_external(df):
    df['is_trading_hours'] = df['time'].apply(check_trading_hours)
    return df

#buy_track calculation function

def buy_track(df):
    # Calculate the date of the day before
    df['previous_day'] = df['time'].apply(lambda x: x - pd.Timedelta(days=1))
    df['previous_day'] = pd.to_datetime(df['previous_day'])
    df['previous_day_4pm'] = df['previous_day'].apply(lambda x: x.replace(hour=16, minute=0))

    # Merge with itself to find the price of the day before at 4 pm
    merged_df = pd.merge(df, df, left_on='previous_day_4pm', right_on='time', suffixes=('', '_previous'))

    # Calculate the decrease percentage compared to the price of the day before at 4 pm
    merged_df['price_decrease_percentage'] = (merged_df['close'] - merged_df['close_previous']) / merged_df['close_previous'] * 100

    # Make a new dataframe and clean it up
    buy_track_temp = merged_df.copy()
    buy_track_temp=buy_track_temp.drop(columns=['previous_day', 'time_previous', 'is_trading_hours_previous', 'previous_day_previous', 'previous_day_4pm_previous'])

    return df, buy_track_temp

def buy_track_2(df, buy_track_temp, price_dec_pct_buy):
    # Make a new dataframe that tracks past buy signals
    buy_track = buy_track_temp[buy_track_temp['price_decrease_percentage'] <= price_dec_pct_buy]
    buy_track = buy_track[buy_track['is_trading_hours'] == True]
    df['included_in_buy_track'] = df['time'].isin(buy_track['time'])

    return df

#extracting df2
def df2_extract(df):
    df2 = df[df['is_trading_hours'] == True].reset_index(drop= True)
    return df2

#sell date calculation function
def sell_track(df2, price_inc_pct_sell, pain_tolerance, max_days_til_sold):
    sell_date_info = []

    for i, row in df2.iterrows():
        if row['included_in_buy_track']:
            sell_index, date_sold, value = next(((index+i+1, df2.time[index+i+1], value) for index, value in enumerate(df2.close[i+1:i+max_days_til_sold]) if (value >= row['close'] * (1 + (0.01 * price_inc_pct_sell))) or (value <= row['close'] * (1 + (0.01 * pain_tolerance)))), (df2.index[i:i+max_days_til_sold].tolist()[-1],df2.time[i:i+max_days_til_sold].tolist()[-1], df2.close[i:i+max_days_til_sold].tolist()[-1]))
            sell_date_info.append([date_sold,value,sell_index])
        else:
            sell_date_info.append([None,None,None])    

    df2['sell_date'] = [j[0] for j in sell_date_info]
    df2['sell_price'] = [j[1] for j in sell_date_info]
    df2['sell_index'] = [j[2] for j in sell_date_info]

    # Calculate the difference in time between 'time' and 'sell_date' and save it to a new column named 'gap'
    df2['gap'] = (df2['sell_index'] - df2.index)/78

    # Calculate the increase percentage of 'sell_price' compared to 'close' for each row
    df2['gain'] = ((df2['sell_price'] - df2['close']) / df2['close']) * 100

    df2['date'] = df2['time'].dt.date
    temp = pd.DataFrame(df2.groupby('date')['included_in_buy_track'].apply(lambda x: int(x.idxmax(skipna=False)) if any(x) else None)).reset_index(drop = True).included_in_buy_track.to_list()

    res = [True if i in temp else False for i in range(df2.shape[0]) ]

    df2['first_buy'] = res

    return df2


######################################################################
######################################################################
#### SIDEBAR
######################################################################
######################################################################



with st.sidebar.form('inputs'):
    st.header("Control Panel")
    load_start_date = str(st.date_input("Choose a Start Date", value=None))
    load_end_date = str(st.date_input("Choose a End Date", value = None))
    load_ticker = st.text_input("Enter the stock ticker", value = None)
    pain_tolerance = float(st.text_input("Stop Loss (2 Digits, Negative)", value = '-1.75'))
    max_days_til_sold = float(st.text_input("Max Days til Sold", value = '15')) * 78

    price_dec_pct_buy = []
    price_inc_pct_sell = []
    number_of_observance = int(st.text_input("Combination Count (Min = 2)", value = '4'))    

    submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state['submitted'] = True
    
if st.session_state['submitted']:
    with st.sidebar.form('inputs2'):
        st.header("Buy/Sell Pct Scenarios")
        for i in range(number_of_observance):
            st.write(f"Combination {i+1}")
            vars()[f'buy_{i}'] = float(st.text_input(f"Buy Percentage {i+1} (-)", value = '0'))
            vars()[f'sell_{i}'] = float(st.text_input(f"Sell Percentage {i+1}", value = '0'))

        submitted_2 = st.form_submit_button("Submit Scenario")

for i in range(number_of_observance):
    price_dec_pct_buy.append(vars()[f'buy_{i}'])
    price_inc_pct_sell.append(vars()[f'sell_{i}'])

if submitted_2:
    st.header(f"{load_ticker} Stock Trading Dashboard")
    progress_text = "Defining Variables"
    my_bar = st.progress(0, text=progress_text)

    # Parameters in percentage; How much % change from the prev day closing price to initiate a buy, and how much % to exit the position
    load_interval = '5'
    load_time_unit = 'minute'
    polygon_api_key = 'RMnfdtr9nmyTjXjgbNJeX_I5pIcowZpl'

    my_bar.progress(10, "Data Collection in Progress")
    df = data_collection(load_ticker, load_interval, load_time_unit, load_start_date, load_end_date, polygon_api_key)

    my_bar.progress(20, "Checking for Trading Hours (T/F)")
    df = check_trading_hour_external(df)

    my_bar.progress(30, "Checking whether in Buy Track")
    df, buy_track_temp =  buy_track(df)

    trades = []
    pcts = []
    avg_gaps = []
    med_gaps = []
    pct_25 = []
    pct_75 = []
    yrs= []
    num_t = []
    gain_t = []
    loss_t = []
    print_yrs = []

    num_col_build = round(len(price_dec_pct_buy)/2,0)

    for i in range(int(num_col_build)):
        vars()[f'col{i*2}'], vars()[f'col{(i*2)+1}'] = st.columns(2)

    for i in range(len(price_dec_pct_buy)):
        with vars()[f'col{i}']:
            st.subheader(f"Scenario {i+1}")
            vars()[f'col{i}_0'], vars()[f'col{i}_1'] = st.columns(2)
            with vars()[f'col{i}_0']:
                st.info(f"Buy = {price_dec_pct_buy[i]}")
            with vars()[f'col{i}_1']:                    
                st.warning(f"Sell = {price_inc_pct_sell[i]}")
            vars()[f'tab{i}_1'], vars()[f'tab{i}_2'], vars()[f'tab{i}_3'] = st.tabs(["Overall Stats", "Gap Distribution", "Monthly Trade"])

    for i in range(len(price_dec_pct_buy)):
        my_bar.progress((i+4)*10, f"Calculating Sell Track (PCT = {price_dec_pct_buy[i]})")
        df = buy_track_2(df, buy_track_temp, price_dec_pct_buy[i])
        df2 = df2_extract(df)
        df2 = sell_track(df2, price_inc_pct_sell[i],pain_tolerance,int(max_days_til_sold))
        
        df3 = df2[df2['first_buy'] == True].reset_index(drop = True)
        df3['year'] = pd.to_datetime(df3['time']).dt.year
        df3['dow'] = pd.to_datetime(df3['time']).dt.day_of_week

        # Group the result dataframe by year
        grouped_result = df3.groupby('year')    

        for year, group in grouped_result:
            # Count the number of rows in the result dataframe
            average_gap = round(group['gap'].mean(),2)
            median_gap = round(group['gap'].median(),2)
            percentile_25 = round(np.percentile(group['gap'],25),2)
            percentile_75 = round(np.percentile(group['gap'],75),2)
            num_trades = group.shape[0]
            group['gain_or_not'] = [True if group['sell_price'].tolist()[i] > group['close'].tolist()[i] else False for i in range(group.shape[0])]

            avg_gaps.append(average_gap)
            med_gaps.append(median_gap)
            pct_25.append(percentile_25)
            pct_75.append(percentile_75)
            num_t.append(num_trades)
            yrs.append(str(year))

            try:
                gain_t.append(group.groupby('gain_or_not').count()['time'][1])
            except Exception as e:
                print(e)
                gain_t.append(0)
            try:
                loss_t.append(group.groupby('gain_or_not').count()['time'][0])            
            except Exception as e:
                print(e)
                loss_t.append(0)
            
            
            num_trades_gap_lower_than_5 = group[group['gap'] < 1].shape[0]
            num_trades_gap_lower_than_10 = group[group['gap'] < 3].shape[0]
            num_trades_gap_lower_than_15 = group[group['gap'] < 5].shape[0]

            trades_temp =[num_trades_gap_lower_than_5,num_trades_gap_lower_than_10,num_trades_gap_lower_than_15]
            pcts+=[round(trades_temp[i] / num_trades * 100,2) for i in range(3)]
            trades += [num_trades_gap_lower_than_5,num_trades_gap_lower_than_10,num_trades_gap_lower_than_15]

            print_yrs += [str(year)]*3


        data = {'Trade Days': [1,3,5]*len(yrs),
                'Count': trades,
                'years': print_yrs,
                'pct': pcts}
        
        data = pd.DataFrame(data)

        fig = px.bar(data,
                    x = 'Trade Days', 
                    y = 'pct',
                    color = 'years',
                    hover_data=['Count'],
                    orientation='v', 
                    barmode= 'group', 
                    text = 'pct',
                    title = f'By Year Trading Days and PCT')
        
        with vars()[f'tab{i}_1']:
            st.plotly_chart(fig,theme="streamlit", use_container_width=True)   

            show_data = pd.DataFrame({'Year': yrs, 
                                    'Trade Count': num_t, 
                                    'Avg Gap': avg_gaps, 
                                    'Med Gap': med_gaps, 
                                    '25 Pct': pct_25, 
                                    '75 Pct': pct_75, 
                                    'Gain T': gain_t, 
                                    'Loss T': loss_t, 
                                    'Gain T (%)': [round((gain_t[i]/num_t[i])*100,2) for i in range(len(num_t))]})
            st.dataframe(show_data)

        with vars()[f'tab{i}_2']:
            fig2 = px.histogram(df3, 
                                x= 'gap', 
                                nbins = 30, 
                                title = 'Distribution of Time Gap between Buy and Sell',
                                labels = {'x': 'Time Gap (days)', 'y': 'Frequency'})
            
            st.plotly_chart(fig2,theme="streamlit", use_container_width=True)   

        with vars()[f'tab{i}_3']:
            fig = plt.figure(figsize=(15, 5))  # Adjust width and height as needed

            # Extract year and month from the 'time' column
            df3['year_month'] = df3['time'].dt.to_period('M')

            # Count the number of datetime rows by year and month
            year_month_counts = df3.groupby('year_month').size()

            # Plot the counts
            year_month_counts.plot(kind='bar', color='skyblue')

            # Add labels and title
            plt.title('Number of trades by Year and Month')
            plt.xlabel('Year-Month')
            plt.ylabel('Count')

            for i, bar in enumerate(plt.gca().patches):
                plt.gca().text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            str(round(bar.get_height(), 2)), ha='center', va='bottom')

            # Display the plot
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
            fig.tight_layout()  # Adjust layout to prevent clipping of labels
            st.pyplot(fig)


        trades = []
        pcts = []
        yrs = []
        num_t = []
        avg_gaps = []
        med_gaps = []
        pct_25 = []
        pct_75 = []
        gain_t = []
        loss_t = []
        print_yrs = []

    my_bar.progress(100, "Completed!")

    st.download_button(label = "Download CSV",
                       data = df3.to_csv().encode('utf-8')m
                       file_name = "df3.csv",
                       mime = 'text/csv')
