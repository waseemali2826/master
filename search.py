import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

# Global variables
repo_path = None
model = None
index = None
master = None


def load_resources():
    global repo_path, model, index, master

    if model is None:
        print("Loading resources...")

        # Download repository (cached after first run)
        repo_path = snapshot_download(
            repo_id="waseem11/master",
            repo_type="model"
        )

        # Load Sentence Transformer model
        model = SentenceTransformer(
            os.path.join(repo_path, "all-MiniLM-L6-v2"),
            device="cpu"
        )

        # Load FAISS index
        index = faiss.read_index(
            os.path.join(repo_path, "faiss.index")
        )

        # Load optimized metadata DataFrame
        with open(
            os.path.join(repo_path, "master.pkl"),
            "rb"
        ) as f:
            master = pickle.load(f)

        print("Resources loaded successfully")


def find_company_items(query, top_k=5):
    load_resources()

    # Create query embedding
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Search in FAISS
    scores, ids = index.search(
        embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        item = master.iloc[idx].to_dict()
        item["similarity"] = float(score)

        results.append(item)

    return results
    
