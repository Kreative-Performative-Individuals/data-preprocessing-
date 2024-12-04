from fastapi import FastAPI, Query
from typing import Optional
from on_request_pipeline import get_request 
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()     

@app.get("/get_forecasting/")
async def get_data(machine_name: str = Query(...), asset_id: str = Query(...), 
                   kpi: str = Query(...), operation: str = Query(...), 
                   timestamp_start: Optional[str] = None, timestamp_end: Optional[str] = None, 
                   transformation: Optional[str] = None, forecasting: bool = False):
    try:
        json_transformed_data = get_request(
            machine_name, asset_id, kpi, operation, timestamp_start, timestamp_end, transformation, forecasting
        )
        return json_transformed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))