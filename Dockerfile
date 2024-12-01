FROM python:3.11-slim

# Imposta la directory di lavoro nel container.
WORKDIR /app

# Copia i file necessari nel container.
COPY requirements.txt .
COPY main.py .
COPY on_request_pipeline.py .
COPY machine_learning_functions.py .
COPY feature_engineering_functions.py .
COPY infoManager.py .
COPY connections_functions.py .

# Installa le dipendenze Python.
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta 8000 per FastAPI.
EXPOSE 8000

# Comando per avviare l'applicazione FastAPI.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
