import os
import sys
import csv
import io
import uuid
import time
import requests
from PIL import Image

# Add project root directory to Python path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client.models import PointStruct
from src.embeddings.fashion_embedder import FashionEmbedder
from src.vector_db.qdrant_service import QdrantService

BATCH_SIZE = 25  # Small batch size to keep payloads light
MAX_RETRIES = 3

def safe_upsert(db_service, points):
    """Upserts points with automatic retries if a network timeout occurs."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            db_service.client.upsert(
                collection_name=db_service.collection_name,
                points=points,
                wait=True
            )
            return True
        except Exception as e:
            print(f"⚠️ Upload attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)  # Pause before retrying
            else:
                print("❌ Max retries reached for this batch. Skipping...")
                return False

def run_ingestion():
    csv_path = "data/sarees.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Please place your CSV file in 'data/sarees.csv'.")
        return

    embedder = FashionEmbedder()
    db_service = QdrantService()

    print(f"Reading dataset from '{csv_path}'...")
    
    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        total_rows = len(rows)
        print(f"Found {total_rows} items to process.\n")

        current_batch = []
        uploaded_count = 0

        for idx, row in enumerate(rows, start=1):
            image_url = row.get("image_url", "").strip()
            if not image_url:
                continue

            name = row.get("Name", "Saree").strip()

            try:
                # Download image
                response = requests.get(image_url, timeout=8)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                
                # Generate embedding
                vector = embedder.embed_image(image)
                
                # Metadata payload
                sku = row.get("SKU", f"item_{idx}").strip()
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, sku + image_url))
                
                payload = {
                    "name": name,
                    "sku": sku,
                    "stock": row.get("Stock", "0").strip(),
                    "retail_price": row.get("Retail Price", "").strip(),
                    "discounted_price": row.get("Discounted Price", "").strip(),
                    "image_url": image_url,
                    "website_link": row.get("Website Link", "").strip()
                }
                
                current_batch.append(PointStruct(id=point_id, vector=vector, payload=payload))
                
            except Exception as e:
                print(f"[{idx}/{total_rows}] Skipping '{name}': Image load error ({e})")

            # Immediate upload as soon as batch size is reached
            if len(current_batch) >= BATCH_SIZE:
                print(f"Uploading batch ({len(current_batch)} items) to Qdrant Cloud...")
                if safe_upsert(db_service, current_batch):
                    uploaded_count += len(current_batch)
                    print(f"✅ Total Uploaded: {uploaded_count}/{total_rows}\n")
                current_batch = []

        # Upload remaining items in last batch
        if current_batch:
            print(f"Uploading final batch ({len(current_batch)} items)...")
            if safe_upsert(db_service, current_batch):
                uploaded_count += len(current_batch)
                print(f"✅ Total Uploaded: {uploaded_count}/{total_rows}\n")

    print(f"🎉 Complete! Successfully indexed {uploaded_count} items into Qdrant Cloud.")

if __name__ == "__main__":
    run_ingestion()