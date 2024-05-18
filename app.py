# app.py

from fastapi import FastAPI, HTTPException
# from ml_model import process_similarity
import uvicorn

app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Find Match Application"}

@app.get("/find-match")
def find_match_endpoint():
    try:
        # results = process_similarity()
        return {"similarities": "Test"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
