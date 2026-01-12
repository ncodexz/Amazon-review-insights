"""
Block 4 - End-to-End Pipeline

This script connects:
- Block 1: sentiment-inferred reviews
- Block 2: product clustering
- Block 3: insights and generation

All steps operate on the same dataset to ensure full consistency.
"""

import json
import re
from collections import defaultdict, Counter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

REVIEWS_PATH = "data/processed/Movies_and_TV_sentiment_v1.jsonl"
CLUSTERS_OUTPUT_PATH = "data/processed/product_clusters.jsonl"


# ============================================================
# REVIEW FILTER
# ============================================================

MIN_TEXT_LENGTH = 30

def is_valid_review(text: str) -> bool:
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False

    text = text.lower()
    noise_patterns = [
        "no comment", "no comments",
        "item was canceled", "order was canceled",
        "did not watch", "never watched",
        "was a gift"
    ]
    return not any(p in text for p in noise_patterns)


# ============================================================
# LOAD SENTIMENT-INFERRED REVIEWS (BLOCK 1 OUTPUT)
# ============================================================

def load_reviews(path):
    reviews = []

    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)

            if "asin" not in r or "text" not in r or "sentiment" not in r:
                continue

            if is_valid_review(r["text"]):
                reviews.append(r)

    return reviews


# ============================================================
# BLOCK 2 — PRODUCT CLUSTERING
# ============================================================

def run_clustering(reviews, n_clusters=5):
    product_reviews = defaultdict(list)

    for r in reviews:
        product_reviews[r["asin"]].append(r["text"])

    product_ids = []
    product_texts = []

    for asin, texts in product_reviews.items():
        if len(texts) < 3:
            continue

        product_ids.append(asin)
        product_texts.append(" ".join(texts[:20]))

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(
        product_texts,
        batch_size=32,
        show_progress_bar=True
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    product_clusters = dict(zip(product_ids, labels))

    with open(CLUSTERS_OUTPUT_PATH, "w") as f:
        for asin, cid in product_clusters.items():
            f.write(json.dumps({
                "asin": asin,
                "cluster_id": int(cid)
            }) + "\n")

    return product_clusters


# ============================================================
# BLOCK 3 — INSIGHTS & GENERATION
# ============================================================

def build_final_reviews(reviews, product_clusters):
    final_reviews = []

    for r in reviews:
        asin = r["asin"]
        if asin not in product_clusters:
            continue

        final_reviews.append({
            "asin": asin,
            "text": r["text"],
            "sentiment": r["sentiment"],
            "cluster_id": product_clusters[asin]
        })

    return final_reviews


def rank_products(final_reviews):
    cluster_product_reviews = defaultdict(lambda: defaultdict(list))

    for r in final_reviews:
        cluster_product_reviews[r["cluster_id"]][r["asin"]].append(r)

    cluster_product_stats = {}

    for cluster_id, products in cluster_product_reviews.items():
        stats = []

        for asin, reviews in products.items():
            sentiments = [r["sentiment"] for r in reviews]
            counts = Counter(sentiments)

            total = len(sentiments)
            neg = counts.get("negative", 0)

            if total < 5:
                continue

            stats.append({
                "asin": asin,
                "total_reviews": total,
                "negative": neg,
                "neg_ratio": neg / total
            })

        cluster_product_stats[cluster_id] = {
            "top_products": sorted(
                stats,
                key=lambda x: (-x["total_reviews"], x["neg_ratio"])
            )[:3],
            "worst_product": sorted(
                stats,
                key=lambda x: (-x["neg_ratio"], -x["total_reviews"])
            )[:1]
        }

    return cluster_product_stats


def extract_canonical_ideas(final_reviews):
    def split_into_ideas(text):
        return [
            s.strip() for s in re.split(r"[.!?]", text)
            if len(s.strip()) >= 20
        ]

    cluster_texts = defaultdict(list)
    for r in final_reviews:
        cluster_texts[r["cluster_id"]].append(r["text"])

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    cluster_canonical_ideas = {}

    for cid, texts in cluster_texts.items():
        ideas = []
        for t in texts:
            ideas.extend(split_into_ideas(t))

        ideas = ideas[:1000]
        if len(ideas) < 8:
            continue

        emb = embedding_model.encode(ideas, show_progress_bar=False)
        labels = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(emb)

        grouped = defaultdict(list)
        for idea, lab in zip(ideas, labels):
            grouped[lab].append(idea)

        canonical = [
            g[len(g)//2] for g in grouped.values() if len(g) >= 3
        ]

        cluster_canonical_ideas[cid] = canonical

    return cluster_canonical_ideas


def generate_articles(cluster_canonical_ideas, cluster_product_stats):
    client = OpenAI()
    articles = {}

    for cid, ideas in cluster_canonical_ideas.items():
        stats = cluster_product_stats.get(cid, {})
        top = stats.get("top_products", [])
        worst = stats.get("worst_product")

        ideas_text = "\n".join(f"- {i}" for i in ideas[:8])
        top_text = "\n".join(
            f"- Product {i+1}: many reviews ({p['total_reviews']}), low negatives ({p['negative']})"
            for i, p in enumerate(top)
        )
        worst_text = (
            f"- Highest negative ratio ({worst[0]['neg_ratio']:.2f})"
            if worst else "- No clear worst product."
        )

        prompt = f"""
You are writing a short, neutral recommendation article.

Overview ideas:
{ideas_text}

Top products:
{top_text}

Worst product:
{worst_text}
""".strip()

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=220,
            temperature=0.3
        )

        articles[cid] = response.output_text.strip()

    return articles


# ============================================================
# MAIN
# ============================================================

def main():
    reviews = load_reviews(REVIEWS_PATH)
    product_clusters = run_clustering(reviews)
    final_reviews = build_final_reviews(reviews, product_clusters)

    cluster_product_stats = rank_products(final_reviews)
    cluster_canonical_ideas = extract_canonical_ideas(final_reviews)

    articles = generate_articles(
        cluster_canonical_ideas,
        cluster_product_stats
    )

    for cid, text in articles.items():
        print(f"\n===== CLUSTER {cid} =====\n")
        print(text)


if __name__ == "__main__":
    main()
