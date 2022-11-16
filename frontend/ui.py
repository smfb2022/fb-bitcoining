import streamlit as st
import asyncio
from utils import consumer_sentiment
from PIL import Image

im = Image.open("./icon.png")
st.set_page_config(
    page_title="Bitcoin Tweeter sentiment with &lt;/&gt;htmx, Websocket, FastAPI & streamlit",
    page_icon=im,
    layout="wide")

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

#create three columns for metrics
cols3 = st.columns(3)

# Let's define next two columns 
cols2 = st.columns(2)
cols2[0].header('Sentiment')
cols2[1].header('BTC price')


selected_model = st.selectbox(
    "Select Model", ("ElKulako/cryptobert", "BTC-finetuned")
)

#For showing connecting status
status = st.empty()

asyncio.run(
    consumer_sentiment(selected_model, cols3, cols2, status)
)
