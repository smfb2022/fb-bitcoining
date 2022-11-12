import os
import json
import dotenv
import asyncio
import tweepy as tw
import pandas as pd
import asyncio
import requests
from typing import List
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from fastapi import WebSocket, WebSocketDisconnect
import requests
app = FastAPI(title='Bitcoin Sentiment Analysis')

async def sentiment_generator(request):
    while True:
        if await request.is_disconnected():
            break
        tweets_with_sentiments = requests.post('http://bitcoin-model-cntr:8000/bitcoin-sentiment') 
        df = pd.DataFrame(pd.DataFrame.from_dict(tweets_with_sentiments.json()))
        table = df.to_html(index=False, justify="center", classes='styled-table', table_id="sentiment")
        yield {
            "event": "sentiment_data",
            "retry": 5000,  # miliseconds
            "data": table,  # HTML representation
        }
        await asyncio.sleep(10)  # in seconds


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))

app = FastAPI()

#server sent events
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    response = requests.get('http://bitcoin-model-cntr:8000')
    #print(response)
    context = {"request": request}
    return templates.TemplateResponse("index.html", context)

@app.get("/sentiment_updates")
async def runStatus(request: Request):
    return EventSourceResponse(sentiment_generator(request))
#SSEs end

#This can be moved to utils
def get_btcprice():
    """
    return: {'usd': 21216}
    """
    payload = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
    r = requests.get('https://api.coingecko.com/api/v3/simple/price', payload)
    return r.json()['bitcoin']

# generator for websocket
async def event_generator(model_name="bitcoin"):
    """
    returns json
    """
    # get the predictions
    tweets_with_sentiments = requests.post("http://bitcoin-model-cntr:8000/bitcoin-sentiment?triton_model_name="+model_name) 
    sentiments = tweets_with_sentiments.json() #model.predict(loadTweets.get_tweets())
    # get latest price
    usd = get_btcprice()
    #Merge both dictionaries
    d = {**sentiments, **usd}
    #print(d)
    yield json.dumps(d)

@app.websocket("/ws_sentiment_updates")
async def websocket_endpoint(websocket: WebSocket, model_name="bitcoin"):
    await manager.connect(websocket)
    try:
        while True:
            async for data in event_generator(model_name):
                await websocket.send_json(data)
                await asyncio.sleep(10)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


