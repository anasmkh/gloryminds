"""
Step 3: Embed every chunk in chunks.jsonl using Ollama's bge-m3 model,
and upload to a Qdrant Cloud collection with grade/subject/etc as
filterable payload fields.

Requires:
    ollama pull bge-m3
    pip install qdrant-client requests

Env vars required:
    QDRANT_URL       e.g. https://xxxxx.cloud.qdrant.io
    QDRANT_API_KEY

Usage:
    python embed_and_upload.py --input chunks.jsonl --collection school_materials
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()  

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024  
BATCH_SIZE = 16


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def ensure_collection(client: QdrantClient, name: str):
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        print(f"Collection '{name}' already exists, reusing it.")
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection '{name}' (dim={EMBED_DIM}, cosine distance).")


def main(input_path: Path, collection_name: str):
    qdrant_url = os.environ.get("QDRANT_URL")
    # qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_url :
        sys.exit("Missing QDRANT_URL or QDRANT_API_KEY environment variables. See script docstring.")

    chunks = load_chunks(input_path)
    print(f"Loaded {len(chunks)} chunks from {input_path}")

    client = QdrantClient(url=qdrant_url)
    ensure_collection(client, collection_name)

    uploaded = 0
    batch_points = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        try:
            vector = get_embedding(text)
        except Exception as e:
            print(f"  FAILED to embed chunk {chunk['id']} ({e}), skipping")
            continue

        payload = dict(chunk["metadata"])
        payload["text"] = text  

        batch_points.append(PointStruct(id=chunk["id"], vector=vector, payload=payload))

        if len(batch_points) >= BATCH_SIZE:
            client.upsert(collection_name=collection_name, points=batch_points)
            uploaded += len(batch_points)
            print(f"  Uploaded {uploaded}/{len(chunks)}")
            batch_points = []

    if batch_points:
        client.upsert(collection_name=collection_name, points=batch_points)
        uploaded += len(batch_points)

    print(f"\nDone. {uploaded}/{len(chunks)} chunks embedded and uploaded to '{collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/Users/apple/Desktop/psy_bot_v2/chunks.jsonl")
    parser.add_argument("--collection", default="school_materials")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    main(input_path, args.collection)