# backend/Dockerfile
FROM python:3.8.13
WORKDIR /app
COPY . .
RUN TMPDIR=~/tmp/ pip3 install --cache-dir=~/tmp/ -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]