# frontend/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt

COPY ui.py ./ui.py
COPY utils.py ./utils.py
COPY icon.png ./icon.png

RUN TMPDIR=~/tmp/ pip3 install --cache-dir=~/tmp/ -r requirements.txt

ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
