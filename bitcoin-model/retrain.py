import os
from os import listdir
from os.path import isfile, join
#import dotenv
import mlflow
import numpy as np
import pandas as pd
import requests
import tweepy as tw
import glob
from datasets import load_dataset
import evaluate
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, TextClassificationPipeline,
                          Trainer, TrainingArguments)

from utils.io import load_yaml

os.environ["TOKENIZERS_PARALLELISM"] = "true"

#let's load the params
config_dict = load_yaml('./config/btc-config.yaml')
model_name = config_dict['model_name']
sentiment = {'bearish':0, 'neutral':1, 'bullish':2}
input_dir = config_dict['input_dir']
output_dir = config_dict['output_dir']
epochs = int(config_dict['epochs'])
#Load env variables
#dotenv.load_dotenv(dotenv.find_dotenv())
# set mlflow env variables
#os.environ['MLFLOW_EXPERIMENT_NAME'] = "mlflow-trainer_cryptobert"
#os.environ['MLFLOW_FLATTEN_PARAMS'] = "1"
#mlflow.set_experiment("batch_serving")

class LoadTweets:
  def __init__(self, config_dict):
    bearer_token = os.getenv('BEARER_TOKEN')
    self.client = tw.Client(bearer_token,  return_type=requests.Response, wait_on_rate_limit=True)
    self.query = config_dict['v2_query']

  def get_tweets(self, max_tweets=10):
    tweets = self.client.search_recent_tweets(query=self.query, 
                                              tweet_fields=['text'], max_results=max_tweets).json()['data']
    return [tweet['text'] for tweet in tweets]
    
  def load_dataset(self, data_files, get_tokens_function, seed=12):
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")
    print(dataset)
    tokenized_dataset = dataset.map(get_tokens_function, batched=True)
    train = tokenized_dataset["train"].shuffle(seed=seed)
    test = tokenized_dataset["test"].shuffle(seed=seed)
    validation = tokenized_dataset.get("validation")
    if validation:
      return train, test, validation.shuffle(seed=seed)
    else:
      return train, test, None

class RoBERTa_sentiment():
    def __init__(self, name: str):
        self.model_name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels = 3)
        self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)

    def predict(self, tweets):
        preds = self.pipe(tweets)
        df = pd.DataFrame(preds)
        df.insert(0, "tweet", tweets, True)
        return df
    
    def get_token(self, row):
        return self.tokenizer(row["tweet"], padding="max_length", truncation=True, max_length=256)



model = RoBERTa_sentiment(model_name)
LoadData = LoadTweets(config_dict)


train_files=glob.glob(input_dir+'/train/'+'*.csv')
test_files=glob.glob(input_dir+'/test/'+'*.csv')
data_files={"train": train_files, "test": test_files}

train, test, _ = LoadData.load_dataset(data_files, model.get_token)

data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer, return_tensors='pt', max_length=256,
                                         padding='max_length')

metric = evaluate.load("accuracy")

# We can add more metrics here
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#mlflow.autolog()

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir = True,
    num_train_epochs=epochs,
    save_total_limit = 1,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    load_best_model_at_end=True,
    learning_rate = 5e-5,
    warmup_steps=500,
    logging_steps=500,
)

trainer = Trainer(
    model=model.model, 
    args=training_args, 
    train_dataset=train, 
    eval_dataset=test,
    tokenizer=model.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
mlflow.set_experiment(config_dict["mlflow_expt"])
with mlflow.start_run():
    mlflow.set_tags(config_dict)
    mlflow.autolog()
    trainer.train()

    trainer.save_model()
    mlflow.log_artifacts("output", artifact_path=config_dict["mlflow_artifact"])
#mlflow.end_run()
