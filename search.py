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

    if model is not None:
        return

    print("Loading resources...")

    # Download Hugging Face repository
    repo_path = snapshot_download(
        repo_id="waseem11/master",
        repo_type="model"
    )

    # Load SentenceTransformer model
    model = SentenceTransformer(
        os.path.join(repo_path, "all-MiniLM-L6-v2"),
        device="cpu"
    )

    # Load COMPRESSED FAISS index
    index = faiss.read_index(
        os.path.join(repo_path, "faiss_compressed.index")
    )

    # IMPORTANT for IVF indexes
    index.nprobe = 20

    # Load metadata
    with open(
        os.path.join(repo_path, "master.pkl"),
        "rb"
    ) as f:
        master = pickle.load(f)

    print("Resources loaded successfully")
    print("Total vectors:", index.ntotal)


def find_company_items(query, top_k=5):
    load_resources()

    # Generate query embedding
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Search
    scores, ids = index.search(
        embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        item = master.iloc[idx].to_dict()
        item["similarity"] = round(float(score), 4)

        results.append(item)

    return results
