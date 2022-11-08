import streamlit as st
import asyncio
from utils import consumer_sentiment

st.set_page_config(
    page_title="Bitcoin Tweeter sentiment with &lt;/&gt;htmx, Websocket, FastAPI & streamlit", 
    layout="wide")

selected_model = st.selectbox(
    "Select Model", ("ElKulako/cryptobert", "BTC-finetuned")
)
#dictionary for store prices
if 'dic_price' not in st.session_state:
	st.session_state.dic_price = {'usd': [], 'date': []}
#sentiment initialization
if 'neutral' not in st.session_state:
	st.session_state.neutral = 0
if 'bullish' not in st.session_state:
	st.session_state.bullish = 0
if 'bearish' not in st.session_state:
	st.session_state.bearish = 0

#For showing connecting status
status = st.empty()
#create three columns for metrics
cols3 = st.columns(3)

# Let's define next two columns 
cols2 = st.columns(2)
cols2[0].header('Sentiment')
cols2[1].header('BTC price')

asyncio.run(
    consumer_sentiment(selected_model, cols3, cols2, status)
)