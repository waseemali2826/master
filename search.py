import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download


repo_path = None
model = None
index = None
master = None


def load_resources():

    global repo_path, model, index, master

    if model is None:

        print("Loading resources...")

        repo_path = snapshot_download(
            repo_id="waseem11/master",
            repo_type="model"
        )


        model = SentenceTransformer(
            os.path.join(repo_path, "all-MiniLM-L6-v2"),
            device="cpu"
        )


        index = faiss.read_index(
            os.path.join(repo_path, "faiss.index")
        )


        with open(
            os.path.join(repo_path, "master.pkl"),
            "rb"
        ) as f:
            master = pickle.load(f)


        print("Resources loaded successfully")


def find_company_items(query, top_k=5):

    load_resources()


    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )


    scores, ids = index.search(
        embedding,
        top_k
    )


    results=[]


    for score, idx in zip(scores[0], ids[0]):

        item = master.iloc[idx].to_dict()

        item["similarity"] = float(score)

        results.append(item)


    return results
