
<h1 align="center" id="heading">Bitcoin Sentiment Capstone</h1>

<b>week7 Experiment Tracking Setup for Data and Model Iterations</em></b>

# Introduction
Bitcoin attracts a lot of attention based on the social media networks.  The project is trying to learn MLOPs principes by fine-tuning a pre-trained model for better sentiment analysis of a bitcoin related tweer.  The eventual application of a model is to find the correlation between bitcoin price fluctuations against social media tweets.
sentiments.

* https://github.com/smfb2022/fb-bitcoining

* https://dagshub.com/fb.capstone/fb-bitcoining

# Performance Report
- Week4 deliverable returns the specified number of bitcoin related tweets along with its sentiments and score.  It can be tested using fast API, docker locally or on EC2. 
- Week5 Model Serving - Implementing Triton Serving of a Hugging Face Model on EC2 instance
- Week7 Experiment Tracking Setup for Data and Model Iterations: original model has been hosted in the triton inference server, and version controlled by DVC and resides in the S3 bucket fb-bitcoin-capstone.  Additional data collected and cleansed for retraining is also version controlled by dvc and resides in dvc bucket.  retraining to create an alternate model with the cleansed data is in progress.  The user interface is a web page that displays the latest 10 tweets using SSE on port 8001(server side events with a polling of 20 secs.).  

# Pre-Trained Model
https://huggingface.co/ElKulako/cryptobert

# Dataset
https://huggingface.co/datasets/ElKulako/stocktwits-crypto (Nov 1 Update)


# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
    Download from Git : 
    - https://github.com/smfb2022/fb-bitcoining
2.	Software dependencies
    - Listed in requirements.txt
3.	Latest releases
4.	API references
    - https://docs.tweepy.org/en/v4.0.1/
    - https://docs.tweepy.org/en/v4.0.1/client.html


# Build and Test
TODO: Describe and show how to build your code and run the tests.
- Create the following the folder from which docker-compose resides
    * Create a .env file with a twitter developer account BEARER_TOKEN="xxxx" in the project folder.
    * Create an .aws.env file with aws access parameters
    * run "aws config" configure access parameters

- Uploading Triton server model.
    * In bitcoin-model/conversion folder run "python cryptobert_2_triton_tracking_batch.py" to generate the trition model in the following structure and automatically upload to the S3 bucket: fb-bitcoin-capstone
        *  trition-model
        *  ├── bitcoin-model'
        *  │   ├── 1
        *  │   │   └── model.pt
        *  │   └── config.pbtxt


![Data Pipeline](./pictures/DataPipeline.png)


- Data in dvc 
    The bitcoin tweets for training is in the data folder
    * Install DVC with (pip3 install dvc)
    * Install DVC with (pip3 install dvc[s3])
    * Initialize DVC (dvc init) in your repo
    * Add s3 remote (dvc remote add -d storage s3://fb-bitcoin-capstone/data/)
    * copy data folder to the same level as bitcoin-model
    * dvc add data/
    * git add data.dvc .gitignore (and any other files)
    * git commit -m "data_<version>"
    * dvc push
    * git push

- Triton model in dvc
* cp triton-model folder a triton-model folder same level as conversion (one level up) 
* dvc add triton-model/
* git add triton-model.dvc .gitignore (and any other files)
* git commit -m <xxx>
* dvc push
* git push
* note: update example: dvc remove model.dvc/ dvc add model


- Retraining 
    * In bitcoin-model/folder run "python retraining.py".  Data is already in the right place to retrain.  It already incorporates all the MLflow logging.
      - Use "dvc pull" command in the repo folder to get data and make sure the data folder in in bitcoin-model/data for retraining

- EC2 installation
    * https://github.com/FourthBrain/MLO-4/tree/main/assignments/week-4#readme 
    * Triton inference server, the main app, and the bitcoin-sentiment app were all launched in seperate containers using the right requirements.txt, dockerfile, and docker-compose.yaml
     using "docker-compose --file docker-compose.yaml up --build" command. 

- Dagshub integration
    * https://dagshub.com/fb.capstone/fb-bitcoining
        * The model and data are reflected in the view because of dvc integraton (from git)
        * There is also one experiment tracked - to track the generation of the triton inference model (by using data logger and pushing logs/ folder into git)


# Montoring and Observability

1.0 Stepup Steps

- Create a new EC2 instance
- Install Docker and Git
- Go download the github repository from https://github.com/Einsteinish/Docker-Compose-Prometheus-and-Grafana
- Update the prometheus.yml to reflec the docker metrics for triton server and Triton inference mertics
- Run the docker compose comamnd to bring up the services
- Login to Prometheus & Grafana UI

1.1 Created a EC2 instance to stand the monitorting and observability components:

- Promentheus
- Grafana
- cAdvisor
- AlertManager
- NodeExporter
- Caddy

2. Configured Prometheus to ingest metrics from Triton Inference Server

3. Created Triton Inference Panels in Grafana dashboard

- Success Request for 30 Seconds
- Success Request Per Minute
- Avereage queue time per Request
- Failure Per 30 seconds

Going to use failure alert for triggering retrianing
We wanted to use BOXKITE to show the data drift through KL Divergence metric or KS-Test
We are working to monitor Triton Inference docker containers metrics


# References
* Triton Server Conversion
1. https://medium.com/nvidia-ai/how-to-deploy-almost-any-hugging-face-model-on-nvidia-triton-inference-server-with-an-8ee7ec0e6fc4
2. NVIDIA Triton meets ArangoDB Workshop: https://www.youtube.com/watch?v=vOIm7Hibgdo&t=1958s
3. Deploying an Object Detection Model with Nvidia Triton Inference Server: https://www.youtube.com/watch?v=1ICVRk6bdss

* MLflow Serving on S3 bucket
1. Serving on S3 bucket: https://www.youtube.com/watch?v=osYRsBVId-A
2. Serving on Dagshub: https://dagshub.com/docs/integration_guide/mlflow_tracking/

* Dagshub Tracking
1. https://dagshub.com/DAGsHub-Official/dagshub-docs/src/11be98003d59a24e045c155b1ccfff036289e58a/docs/feature_guide/git_tracking.md


# Misc Useful Cmds
TODO:
* delete all docker container: docker rm $(docker ps -a -q)
* delete all docker images with biton in name: docker images -a | grep bitcoin | awk '{print $3}' | xargs docker rmi -f
* cleanout docker images, containers etc: docker system prune -a
* aws s3 ls s3://fb-bitcoin-capstone --recursive --human-readable --summarize to view the model and other contents
* Test Transformers installation:
    python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"

# Contribute
TODO: Explain how other users and developers can contribute to make your code better.
1. The model still needs to be used to correlate to bitcoin prices

