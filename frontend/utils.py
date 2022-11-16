import aiohttp
import json
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

WS_CONN_BASE = "ws://backend:8000/ws_sentiment_updates?model_name="
WS_CONN = "ws://backend:8000/ws_sentiment_updates"

def get_time():
    now = datetime.now()
    dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    return dt_string

def update_metrics(container, dict_counts):
    if "Bearish" in dict_counts:
        if 'container_bearish' in st.session_state:
            st.session_state.container_bearish.empty() #clear it
        st.session_state.container_bearish = container[0].empty()
        with st.session_state.container_bearish.container():
            st.metric(
                label="Bearish",
                value=dict_counts['Bearish'],
                delta=dict_counts['Bearish']-st.session_state.bearish,
            )
        st.session_state.bearish = dict_counts['Bearish']
    
    if "Neutral" in dict_counts:
        if 'container_neutral' in st.session_state:
            st.session_state.container_neutral.empty() #clear it
        st.session_state.container_neutral = container[1].empty()
        with st.session_state.container_neutral.container():
            st.metric(
                label="Neutral",
                value=dict_counts['Neutral'],
                delta=dict_counts['Neutral']-st.session_state.neutral,
            )
        st.session_state.neutral = dict_counts['Neutral']
        
    if "Bullish" in dict_counts:
        if 'container_bullish' in st.session_state:
            st.session_state.container_bullish.empty() #clear it
        st.session_state.container_bullish = container[2].empty()
        with st.session_state.container_bullish.container():
            st.metric(
                label="Bullish",
                value=dict_counts['Bullish'],
                delta=dict_counts['Bullish']-st.session_state.bullish,
            )
        st.session_state.bullish = dict_counts['Bullish']

def plot_sentiment(container, df):
    counts = df['label'].value_counts()
    ssum = df.groupby(['label']).sum()
    #compute averages
    scores = {'Neutral': '0', 'Bullish': '0', 'Bearish':'0'}
    avg = []
    #sometimes Series counts does not contain an attribute
    if hasattr(counts, 'Neutral'):
        scores["Neutral"] = f'{ssum.score.Neutral/counts.Neutral: .2f}'
        avg.append('score:'+scores["Neutral"])
    if hasattr(counts, 'Bullish'):
        scores["Bullish"] = f'{ssum.score.Bullish/counts.Bullish: .2f}'
        avg.append('score:'+scores["Bullish"])
    if hasattr(counts, 'Bearish'):
        scores["Bearish"] = f'{ssum.score.Bearish/counts.Bearish: .2f}'
        avg.append('score:'+scores["Bearish"])

    fig, ax = plt.subplots()
    dict_counts = counts.to_dict() #this is used in metrics
    ax.bar(dict_counts.keys(), dict_counts.values(), color=[(0.1, 0.1, 0.1, 0.1), 'palegreen', 'tomato'], edgecolor='black')
    ax.bar_label(ax.containers[0], avg)
    #Let's clear the container
    if 'container_sentiment' in st.session_state:
        st.session_state.container_sentiment.empty()
    st.session_state.container_sentiment = container.empty()
    #plot again
    with st.session_state.container_sentiment.container():
        st.pyplot(fig)
    return dict_counts

def plot_price(container):
    #Let's clear the container
    if 'container_price' in st.session_state:
        st.session_state.container_price.empty()
    st.session_state.container_price = container.empty()
    #Plot again
    with st.session_state.container_price.container():
        fig, ax = plt.subplots()
        ax.plot("date", "usd", data=st.session_state.dic_price, marker='o', alpha=0.4)
        fig.autofmt_xdate()
        st.pyplot(fig)

async def consumer_sentiment(model, cols3, cols2, status):
    WS_CONN = WS_CONN_BASE + model
    async with aiohttp.ClientSession(trust_env=True) as session:
        status.subheader(f"Connecting to {WS_CONN}")
        async with session.ws_connect(WS_CONN) as websocket:
            status.subheader(f"Connected to: {WS_CONN}")
            async for message in websocket:
                data = message.json()
                data = json.loads(data)
                #get price and remove it
                usd = data['usd']
                del data['usd']
                #Plotting sentiment
                df = pd.DataFrame(data)
                dict_counts = plot_sentiment(cols2[0], df)
                #Show metrics
                update_metrics(cols3, dict_counts)
                #Plotting prices
                st.session_state.dic_price['usd'].append(usd)
                st.session_state.dic_price['date'].append(get_time())
                df_price = pd.DataFrame(st.session_state.dic_price)
                plot_price(cols2[1])
