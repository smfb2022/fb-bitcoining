import torch
from pathlib import Path
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import boto3
import os
import sys
from mlflow.tracking import MlflowClient
import yaml


def load_yaml(config_path = '../bitcoin-model/config/btc-config.yaml'):
    """_summary_
    Returns:
        _type_: _description_
    """
    #config_path = './config/btc-config.yaml'
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    return params

config_dict = load_yaml()

# model structure from the current directory "."
# trition-model
#   ├── bitcoin-model
#   │   ├── 1
#   │   │   └── model.pt
#   │   └── config.pbtxt
# models should  be bitcoin-model
model_name = "model.pt"
config_name = "config.pbtxt"
root_path = config_dict["root_path"]
model_path = root_path + "1/"
model_filepath = model_path + model_name
config_filepath = root_path + config_name
s3_bucket = config_dict["s3_bucket"]


model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
sentence = "Bitcoin #BTC is going Up.  it is great"
sentence1 = "Bitcoin #BTC is Down. Bad"
labels =["Bearish", "Neutral", "Bullish"]
inputs = tokenizer.batch_encode_plus([sentence, sentence1],
                                    return_tensors='pt', max_length=256,
                                    truncation=True, padding='max_length'
                                    )
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

input_ids.shape, attention_mask.shape


class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self):
        super(PyTorch_to_TorchScript, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,  num_labels = 3)
    def forward(self, data, attention_mask=None):
        return self.model(data, attention_mask)[0]

def save_bitcoin_model():
    # after trace it will save the model in cwd
    pt_model = PyTorch_to_TorchScript().eval()
    traced_script_module = torch.jit.trace(pt_model, (input_ids, attention_mask), strict=False)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    traced_script_module.save(model_filepath)


    configuration = """
    name: "bitcoin-model"
    platform: "pytorch_libtorch"
    max_batch_size: 32
    input [
    {
        name: "input__0"
        data_type: TYPE_INT32
        dims: [ 256 ]
      } ,
    {
        name: "input__1"
        data_type: TYPE_INT32
        dims: [ 256 ]
      }
    ]
    output {
        name: "output__0"
        data_type: TYPE_FP32
        dims: [ 3 ]
      }
    """
    with open(config_filepath, 'w') as file:
        file.write(configuration)

def upload_to_s3():
  s3_client = boto3.client('s3')
  response = s3_client.list_buckets()
  for bucket in response['Buckets']:
    print(f'{bucket["Name"]}')

def download_files(bucket_name, path_to_download, save_as=None):
  s3_client = boto3.client('s3')
  object_to_download = path_to_download
  s3_client.download_file(bucket_name, object_to_download, save_as)

def upload_file(file_name, bucket, store_as=None):
  if store_as is None:
    store_as = file_name
  s3_client = boto3.client('s3')
  s3_client.upload_file(file_name, bucket, store_as)

def download_mlflow_artifact(runid, artifact, local_path):
    client = MlflowClient()
    client.download_artifacts(runid, artifact, local_path)

if __name__ == "__main__":
  
  mlflow.set_experiment(config_dict["mlflow_expt"])
  runid_provided = False
  #mlflow_runid = "8dc049f234324330992b68ea3e36a8f3"
  mlflow_artifact = config_dict["mlflow_artifact"]
  local_path = '.' 
  mlflow_runid = model_name
  if (len(sys.argv)) == 2:
       mlflow_runid = sys.argv[1]
       model_name = local_path+"/"+mlflow_artifact
       runid_provided = True
  
  with mlflow.start_run():
      if runid_provided == True:
        download_mlflow_artifact(mlflow_runid, mlflow_artifact, local_path)
      save_bitcoin_model()
      mlflow.log_param("hugging face model", mlflow_runid)
      mlflow.set_tags(config_dict)
      upload_file(model_filepath, s3_bucket, store_as= model_filepath)
      upload_file(config_filepath, s3_bucket, store_as= config_filepath)


