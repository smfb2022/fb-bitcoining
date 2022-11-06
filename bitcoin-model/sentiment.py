import logging
import pandas as pd
import shutil
from utils.logging import getLogger
from classifier import build_crypto_sentiment_analyzer, TritonBitcoinSentiment
from utils.io import load_yaml
from utils.load import LoadTweets
import mlflow
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("run.log"), logging.StreamHandler()],
)

class BitcoinSentiment():
    def __init__(self, triton_url='triton:8002'):
        
        # create logging
        self.logger = getLogger("tweet sentiment")
        self.logger.propagate = True
        self.triton_url = triton_url

        #load config
        self.config_dict = load_yaml()

        # create sentiment classifer
        self.btc_analyzer = build_crypto_sentiment_analyzer(self.config_dict["model_name"])
        
        # create twitter data loader
        self.dl = LoadTweets(self.config_dict, self.logger)

        # triton inference server
        self.tis = TritonBitcoinSentiment(triton_url)



    def predict(self, num_tweets = 10):

        with mlflow.start_run(nested=True):
            mlflow.log_metric('Accuracy', 1)
            print('Logged Metric')

        # get tweets and predict sentiments
        posts = self.dl.get_tweets(num_tweets)
        
        if (self.config_dict['inference'] == 'triton'):
            df = self.tis.run_inference(posts)
            #print(df)
        else:
            preds = self.btc_analyzer(posts)
            df = pd.DataFrame(preds)
            df.insert(0, "tweets", posts, True)
            #print(df)

        return df








