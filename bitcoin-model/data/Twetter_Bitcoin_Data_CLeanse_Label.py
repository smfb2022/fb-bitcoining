from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#model.save_pretrained(MODEL)

# Initialize reader object: df_reader
#df = pd.read_csv('Bitcoin_tweets_1000_1.csv', chunksize=100)

# Initialize an empty dictionary: counts_dict
counts_dict = {}

for chunk in pd.read_csv('chunk9.csv', chunksize=20000):
    Tweet_list=[]
    for rows in chunk['text']:
#        print(len(counts_dict))
#        text_1 = "Bitcoin may fall because of dollar being strong"
            text = preprocess(rows)
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            maxindex = int(np.argmax(scores))
#           print(maxindex)
#            with open("bit-twt.csv", "w") as text_file:
#                print(f'{text}||{config.id2label[maxindex]}||{maxindex}', file=text_file)
            Tweet_list.append([text, config.id2label[maxindex], maxindex])
#           print(f'{text}||{config.id2label[maxindex]}||{maxindex}')
    df = pd.DataFrame(Tweet_list, columns = ['Tweets','Sentiment','Score'])
    df.to_csv("chunk9_out.csv")
#return counts_dict
#print(counts_dict)

