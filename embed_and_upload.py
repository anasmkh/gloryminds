import argparse
import json
import os
import sys
import time
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

MAX_RETRIES = 6
BASE_DELAY = 5  


def with_retries(fn, *args, description="operation", **kwargs):
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == MAX_RETRIES:
                break
            delay = BASE_DELAY * (2 ** (attempt - 1))
            print(f"  WARN: {description} failed ({e}). "
                  f"Retry {attempt}/{MAX_RETRIES} in {delay}s...")
            time.sleep(delay)
    raise last_exception


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=300,
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
    existing = [c.name for c in with_retries(
        client.get_collections, description="listing collections"
    ).collections]
    if name in existing:
        print(f"Collection '{name}' already exists, reusing it.")
        return
    with_retries(
        client.create_collection,
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        description="creating collection",
    )
    print(f"Created collection '{name}' (dim={EMBED_DIM}, cosine distance).")


def get_existing_ids(client: QdrantClient, collection_name: str, ids: list[str]) -> set:
    if not ids:
        return set()
    records = with_retries(
        client.retrieve,
        collection_name=collection_name,
        ids=ids,
        with_payload=False,
        with_vectors=False,
        description="checking existing chunk IDs",
    )
    return {str(r.id) for r in records}


def main(input_path: Path, collection_name: str):
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")  
    if not qdrant_url:
        sys.exit("Missing QDRANT_URL environment variable. See script docstring.")

    chunks = load_chunks(input_path)
    print(f"Loaded {len(chunks)} chunks from {input_path}")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

    try:
        ensure_collection(client, collection_name)
    except Exception as e:
        sys.exit(f"\nCould not reach Qdrant after {MAX_RETRIES} retries: {e}\n"
                  f"Nothing was uploaded yet, so it's safe to just rerun this script later.")

    print("Checking which chunks are already uploaded (for resume support)...")
    all_ids = [c["id"] for c in chunks]
    existing_ids = set()
    try:
        for i in range(0, len(all_ids), 200):
            existing_ids |= get_existing_ids(client, collection_name, all_ids[i:i + 200])
    except Exception as e:
        sys.exit(f"\nCould not reach Qdrant after {MAX_RETRIES} retries while checking existing "
                  f"chunks: {e}\nNo new upload was attempted - rerun this script later to resume.")

    chunks_to_process = [c for c in chunks if c["id"] not in existing_ids]
    print(f"  {len(existing_ids)} already present, {len(chunks_to_process)} new to embed.\n")

    uploaded = 0
    batch_points = []

    def flush_batch():
        nonlocal uploaded, batch_points
        if not batch_points:
            return
        with_retries(
            client.upsert,
            collection_name=collection_name,
            points=batch_points,
            description=f"uploading batch ({len(batch_points)} points)",
        )
        uploaded += len(batch_points)
        print(f"  Uploaded {uploaded}/{len(chunks_to_process)}")
        batch_points = []

    try:
        for chunk in chunks_to_process:
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
                flush_batch()

        flush_batch()  # upload any remainder

    except Exception as e:
        print(f"\nStopped: Qdrant became unreachable after {MAX_RETRIES} retries ({e}).")
        print(f"{uploaded}/{len(chunks_to_process)} new chunks were uploaded before the failure.")
        print("Progress is saved in Qdrant - just rerun this script later to resume.")
        sys.exit(1)

    print(f"\nDone. {uploaded}/{len(chunks_to_process)} new chunks embedded and uploaded "
          f"to '{collection_name}' ({len(existing_ids)} already present were skipped).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/Users/apple/Desktop/psy_bot_v2/chunks.jsonl")
    parser.add_argument("--collection", default="school_materials")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    main(input_path, args.collection)