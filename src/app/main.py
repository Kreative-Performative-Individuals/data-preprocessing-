from fastapi import FastAPI, Query
from datetime import datetime
from typing import Optional
from src.app.on_request_pipeline import get_request
from src.app.real_time.publisher import KafkaPublisher
from src.app.real_time.request import KPIStreamingRequest, KPIValidator
from src.app.real_time.message import RealTimeKPI
import os
from dotenv import load_dotenv
from src.app.connections_functions import get_datapoint
from src.app.dataprocessing_functions import cleaning_pipeline
import uvicorn

load_dotenv()

KAFKA_TOPIC_NAME = os.getenv("KAFKA_TOPIC_NAME")
KAFKA_SERVER = os.getenv("KAFKA_SERVER")
KAFKA_PORT = os.getenv("KAFKA_PORT")
BATCH_SIZE = 1000

app = FastAPI()

publisher = KafkaPublisher(
    topic=KAFKA_TOPIC_NAME,
    port=KAFKA_PORT,
    servers=KAFKA_SERVER,
)


def start():
    print("Starting the data preprocessing API...")
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8003, reload=True)


@app.on_event("startup")
async def startup_event():
    # Placeholder for startup operations (add logic if needed)
    pass


@app.get("/")
def root():
    return {"message": "Data preprocessing in progress."}


@app.get("/get_request")
def get_request_callback(
        machine_name: str = Query(..., description="Name of the machine"),
        asset_id: str = Query(..., description="ID of the asset"),
        kpi: str = Query(..., description="Key Performance Indicator"),
        operation: str = Query(..., description="Type of operation"),
        timestamp_start: datetime = Query(..., description="Start timestamp in ISO format"),
        timestamp_end: datetime = Query(..., description="End timestamp in ISO format"),
        transformation: Optional[str] = Query(None, description="Transformation type (e.g., 'S', 'T')"),
        forecasting: bool = Query(False, description="Enable forecasting (true/false)"),
):
    result = get_request(machine_name, asset_id, kpi, operation, timestamp_start, timestamp_end, transformation,
                         forecasting)

    return result


@app.post("/real-time-session")
async def real_time_session_callback(kpi_streaming_request: KPIStreamingRequest):
    _ = publisher.open_session()
    i = 0
    accumulated_data = []
    kpi_validator = KPIValidator.from_streaming_request(kpi_streaming_request)
    ready_request_size = kpi_validator.kpi_count

    while True:
        cleaned_data = cleaning_pipeline(get_datapoint(i))
        if kpi_validator.validate(cleaned_data):
            aggregation_column = kpi_validator.get_aggregation_from_kpi(cleaned_data["kpi"])
            real_time_kpi = RealTimeKPI.from_dictionary(cleaned_data, aggregation_column)
            accumulated_data.append(real_time_kpi)

        if ready_request_size == len(accumulated_data):
            await publisher.send(accumulated_data)
            accumulated_data = []

        i = (i + 1) % BATCH_SIZE


@app.get("/stop-real-time")
def stop_real_time_callback():
    return publisher.finalize()


@app.get("/health/")
def health_check():
    return {"status": "ok"}
