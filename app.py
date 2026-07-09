from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search import find_company_items
import pandas as pd

app = FastAPI(
    title="Master Dataset Semantic Search API",
    description="Semantic Search using Sentence Transformers + FAISS",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Master Dataset Semantic Search API",
        "docs": "/docs"
    }


@app.post("/search")
def search(request: SearchRequest):
    try:
        result = find_company_items(request.query)

        # If DataFrame, convert to JSON
        if isinstance(result, pd.DataFrame):
            records = result.to_dict(orient="records")
        else:
            records = result

        return {
            "query": request.query,
            "total_results": len(records),
            "results": records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
