from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Placeholder for startup operations (add logic if needed)
    pass

@app.get("/")
def root():
    return {"message": "Data preprocessing in progress."}
