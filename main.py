from fastapi import FastAPI
import pandas
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
import requests
app = FastAPI(title='Bitcoin Sentiment Analysis')

async def sentiment_generator(request):
    while True:
        if await request.is_disconnected():
            break
        tweets_with_sentiments = requests.post('http://bitcoin-model-cntr:8000/bitcoin-sentiment') #model.predict()
        df = pandas.DataFrame(pandas.DataFrame.from_dict(tweets_with_sentiments.json()))
        table = df.to_html(index=False, justify="center", classes='styled-table', table_id="sentiment")
        yield {
            "event": "sentiment_data",
            "retry": 5000,  # miliseconds
            "data": table,  # HTML representation
        }
        await asyncio.sleep(10)  # in seconds


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    response = requests.get('http://bitcoin-model-cntr:8000')
    print(response)
    context = {"request": request}
    return templates.TemplateResponse("index.html", context)

@app.get("/sentiment_updates")
async def runStatus(request: Request):
    return EventSourceResponse(sentiment_generator(request))

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0", debug=True)



