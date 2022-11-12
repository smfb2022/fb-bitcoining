from fastapi import FastAPI
from sentiment import BitcoinSentiment

# Set triton url path
triton_url = 'triton:8000'

model = BitcoinSentiment(triton_url)

app = FastAPI(title='bitcoin-model')

#The bitcoin-sentiment endpoint receives number of twitter posts to analyze and returns the posts and sentiments
@app.post("/bitcoin-sentiment", tags=["Analysis"])
async def sentiment(triton_model_name: str = 'bitcoin-model'):

     #print(f'triton_model_name {triton_model_name}')
     #We run the model to get the tweets and analyze them
     tweets_with_sentiments = model.predict(triton_model_name=triton_model_name, model_version='1')
    
     #We encode the sentiments before returning it
     tweets_with_sentiments = tweets_with_sentiments.to_dict()

     return tweets_with_sentiments


@app.get("/", tags=["Health Check"])
async def root():
     return {"message": "Ok"}

