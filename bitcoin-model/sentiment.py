import logging
import pandas as pd
import shutil
from utils.logging import getLogger
from classifier import build_crypto_sentiment_analyzer, TritonBitcoinSentiment
from utils.io import load_yaml
from utils.load import LoadTweets
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



    def predict(self, triton_model_name="bitcoin-model", model_version='1'):

        #print(f'triton_model_name {triton_model_name}')
        # get tweets and predict sentiments
        posts = self.dl.get_tweets(self.config_dict['num_tweets'])
        
        if (self.config_dict['inference'] == 'triton'):
            df = self.tis.run_inference(posts, triton_model_name=triton_model_name)
            #print(df)
        else:
            preds = self.btc_analyzer(posts)
            df = pd.DataFrame(preds)
            df.insert(0, "tweet", posts, True)
            #print(df)

        return df








