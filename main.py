from fastapi import FastAPI, Query
from datetime import datetime
from typing import Optional
from on_request_pipeline import get_request 

app = FastAPI()     

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
    result = get_request(machine_name, asset_id, kpi, operation, timestamp_start, timestamp_end, transformation, forecasting)

    return result
