FROM python:3.11-slim
WORKDIR /srv
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/srv/hf-cache
ARG MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ENV MODEL_NAME=${MODEL_NAME}
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# Download weights at build time so they never live in git.
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='${MODEL_NAME}')"
COPY app ./app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=12 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
