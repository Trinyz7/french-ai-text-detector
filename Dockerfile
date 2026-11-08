FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/data/hf_cache
# WITH_LLM=1 installe torch (version CPU) + transformers pour la perplexité.
ARG WITH_LLM=1
COPY requirements.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$WITH_LLM" = "1" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-llm.txt; \
    fi
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
