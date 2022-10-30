# fb-bitcoining
Capstone Project


install dvc
install dvc[s3]
new bucket created fb-bitcoin-capstone

commands
Install DVC with (pip3 install dvc)
Install DVC with (pip3 install dvc[s3])
Initialize DVC (dvc init) in your repo
Add s3 remote (dvc remote add -d storage s3://fb-bitcoin-capstone/data/)
copy data folder to the same level as bitcoin-model
dvc add data/
git add data.dvc .gitignore (and any other files)
git commit -m "data"
dvc push
git push
(aws configure)



copy contents of s3:///gb-bitcoin-capstone/data folder to bitcoin-model/data for retraining





Sticking the original Triton server in the git repo
Go to conversion folder
python cryptobert_2_triton_tracing_batch.py
model is created in triton-mdoel/base
if s3 permissions are set it automatical load to aws s3://fb-bitcoin-capstone/triton-model/base/
aws s3 ls s3://fb-bitcoin-capstone --recursive --human-readable --summarize to view the model
cp triton-model folder to the model folder same level as conversion (one level up) to stick it into dvc / s3
git add triton-model.dvc .gitignore (and any other files)

dvc remove / dvc add
dvc remove model.dvc
dvc add model/

dvc push
git commit -m "message"
git push

https://dagshub.com/DAGsHub-Official/dagshub-docs/src/11be98003d59a24e045c155b1ccfff036289e58a/docs/feature_guide/git_tracking.md# fb-bitcoining
Capstone Project
