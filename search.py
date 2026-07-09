import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

repo_path = snapshot_download(
    repo_id="waseem11/master",
    repo_type="model"
)

model = SentenceTransformer(
    os.path.join(repo_path, "all-MiniLM-L6-v2")
)

embeddings = np.load(
    os.path.join(repo_path, "embeddings.npy")
)

index = faiss.read_index(
    os.path.join(repo_path, "faiss.index")
)

with open(os.path.join(repo_path, "master.pkl"), "rb") as f:
    master = pickle.load(f)