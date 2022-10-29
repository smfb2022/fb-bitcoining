from fastapi import FastAPI
from sentiment import BitcoinSentiment

# Set triton url path
triton_url = 'triton:8000'

model = BitcoinSentiment(triton_url)

app = FastAPI(title='bitcoin-model')

#The bitcoin-sentiment endpoint receives number of twitter posts to analyze and returns the posts and sentiments
@app.post("/bitcoin-sentiment", tags=["Analysis"])
async def sentiment(num_tweets: int = 10):

     #We run the model to get the tweets and analyze them
     tweets_with_sentiments = model.predict(num_tweets=num_tweets)
    
     #We encode the sentiments before returning it
     tweets_with_sentiments = tweets_with_sentiments.to_dict()

     return tweets_with_sentiments


@app.get("/", tags=["Health Check"])
async def root():
     return {"message": "Ok"}

