FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml VERSION ./
COPY serve.py vib_messages.py vib_state.py pad_schema.py battery.py occ.py ./
COPY models/e5s/ ./models/e5s/

RUN pip install --no-cache-dir onnxruntime tokenizers numpy fastapi uvicorn pydantic

EXPOSE 18973

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "18973", "--workers", "1"]
