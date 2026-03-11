FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV MAX_PREVIEW_ROWS=20

RUN mkdir -p /data

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data_preprocessor_server.py .

RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app && \
    chown -R mcpuser:mcpuser /data

USER mcpuser

CMD ["python", "data_preprocessor_server.py"]
