from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search import find_company_items

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

# Request Model
class SearchRequest(BaseModel):
    query: str
    threshold: float = 0.50


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
        results = find_company_items(
            request.query,
            threshold=request.threshold
        )

        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))