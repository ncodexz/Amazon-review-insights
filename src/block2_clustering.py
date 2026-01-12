"""
Block 2 - Product Category Clustering

This script groups products into semantic clusters based on how users
describe them in reviews. It operates at the product (ASIN) level.
"""

import json
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# ----------------------------
# Parameters
# ----------------------------

INPUT_PATH = "data/processed/reviews_with_sentiment.jsonl"
OUTPUT_PATH = "data/processed/product_clusters.jsonl"

MAX_PRODUCTS = 20000
MAX_REVIEWS_PER_PRODUCT = 20
MIN_VALID_REVIEWS = 3
MIN_TEXT_LENGTH = 30

N_CLUSTERS = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ----------------------------
# Review filtering
# ----------------------------

def is_valid_review(text: str) -> bool:
    if not text:
        return False

    if len(text.strip()) < MIN_TEXT_LENGTH:
        return False

    text_lower = text.lower()
    noise_patterns = [
        "no comment",
        "no comments",
        "item was canceled",
        "order was canceled",
        "did not watch",
        "never watched",
        "was a gift"
    ]

    return not any(p in text_lower for p in noise_patterns)


# ----------------------------
# Load reviews
# ----------------------------

def load_reviews(path: str):
    reviews = []

    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)

            if "asin" not in r or "text" not in r:
                continue

            text = r["text"].strip()
            if not text:
                continue

            reviews.append({
                "asin": r["asin"],
                "text": text
            })

    return reviews


# ----------------------------
# Build product texts
# ----------------------------

def build_product_texts(reviews):
    product_reviews = defaultdict(list)

    for r in reviews:
        asin = r["asin"]
        text = r["text"]

        if is_valid_review(text):
            if len(product_reviews[asin]) < MAX_REVIEWS_PER_PRODUCT:
                product_reviews[asin].append(text)

    # keep products with enough signal
    product_reviews = {
        asin: texts
        for asin, texts in product_reviews.items()
        if len(texts) >= MIN_VALID_REVIEWS
    }

    # optional cap
    product_reviews = dict(list(product_reviews.items())[:MAX_PRODUCTS])

    product_ids = []
    product_texts = []

    for asin, texts in product_reviews.items():
        product_ids.append(asin)
        product_texts.append(" ".join(texts))

    return product_ids, product_texts


# ----------------------------
# Clustering
# ----------------------------

def cluster_products(product_texts):
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cuda" if SentenceTransformer._target_device == "cuda" else "cpu"
    )

    embeddings = model.encode(
        product_texts,
        batch_size=32,
        show_progress_bar=True
    )

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(embeddings)

    return labels


# ----------------------------
# Save results
# ----------------------------

def save_clusters(product_ids, labels, path: str):
    with open(path, "w") as f:
        for asin, cluster_id in zip(product_ids, labels):
            record = {
                "asin": asin,
                "cluster_id": int(cluster_id)
            }
            f.write(json.dumps(record) + "\n")


# ----------------------------
# Main
# ----------------------------

def main():
    reviews = load_reviews(INPUT_PATH)
    product_ids, product_texts = build_product_texts(reviews)
    labels = cluster_products(product_texts)
    save_clusters(product_ids, labels, OUTPUT_PATH)

    print(f"Saved product clusters to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
