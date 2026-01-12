"""
Block 3 - Insights & Content Generation

This script links product clusters with reviews, computes simple signals,
extracts canonical ideas, and generates neutral summaries using an LLM.
"""

import json
import re
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from openai import OpenAI


# ----------------------------
# Paths
# ----------------------------

REVIEWS_PATH = "data/raw/Movies_and_TV.jsonl"
CLUSTERS_PATH = "data/processed/product_clusters.jsonl"

# (NOTE: sentiment here follows the original notebook logic: star-based)
# This is kept as-is for alignment with the delivered project.


# ----------------------------
# Load cluster mapping
# ----------------------------

def load_product_clusters(path):
    product_clusters = {}

    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            product_clusters[r["asin"]] = r["cluster_id"]

    return product_clusters


# ----------------------------
# Review filtering (same as Block 2)
# ----------------------------

MIN_TEXT_LENGTH = 30

def is_valid_review(text):
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
# Build final reviews with cluster
# ----------------------------

def build_final_reviews(reviews_path, product_clusters):
    final_reviews = []

    with open(reviews_path, "r") as f:
        for line in f:
            r = json.loads(line)

            if "asin" not in r or "text" not in r or "rating" not in r:
                continue

            asin = r["asin"]
            if asin not in product_clusters:
                continue

            text = r["text"].strip()
            if not is_valid_review(text):
                continue

            rating = r["rating"]
            if rating <= 2:
                sentiment = "negative"
            elif rating == 3:
                sentiment = "neutral"
            else:
                sentiment = "positive"

            final_reviews.append({
                "asin": asin,
                "text": text,
                "sentiment": sentiment,
                "cluster_id": product_clusters[asin]
            })

    return final_reviews


# ----------------------------
# Cluster → product → reviews
# ----------------------------

def group_reviews(final_reviews):
    cluster_product_reviews = defaultdict(lambda: defaultdict(list))

    for r in final_reviews:
        cluster_product_reviews[r["cluster_id"]][r["asin"]].append(r)

    return cluster_product_reviews


# ----------------------------
# Rank products per cluster
# ----------------------------

def rank_products(cluster_product_reviews):
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

        top_products = sorted(
            stats,
            key=lambda x: (-x["total_reviews"], x["neg_ratio"])
        )[:3]

        worst_product = sorted(
            stats,
            key=lambda x: (-x["neg_ratio"], -x["total_reviews"])
        )[:1]

        cluster_product_stats[cluster_id] = {
            "top_products": top_products,
            "worst_product": worst_product[0] if worst_product else None
        }

    return cluster_product_stats


# ----------------------------
# Canonical idea extraction
# ----------------------------

def split_into_ideas(text):
    sentences = re.split(r"[.!?]", text)
    return [s.strip() for s in sentences if len(s.strip()) >= 20]


def select_canonical_idea(ideas):
    clean = []

    for s in ideas:
        lower = s.lower()

        bad_patterns = [
            "i ", "my ", "we ", "our ",
            "amazon", "prime", "shipping",
            "dvd", "blu", "disc",
            "bought", "order", "season", "episode"
        ]

        if any(p in lower for p in bad_patterns):
            continue

        if not any(
            k in lower for k in
            ["story", "acting", "characters", "plot", "series", "movie", "film"]
        ):
            continue

        if len(s) >= 35:
            clean.append(s)

    if not clean:
        return None

    clean = sorted(clean, key=len)
    return clean[len(clean) // 2]


def extract_canonical_ideas(final_reviews):
    cluster_texts = defaultdict(list)

    for r in final_reviews:
        cluster_texts[r["cluster_id"]].append(r["text"])

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    cluster_canonical_ideas = {}

    for cluster_id, texts in cluster_texts.items():
        ideas = []
        for t in texts:
            ideas.extend(split_into_ideas(t))

        ideas = ideas[:1000]
        if len(ideas) < 8:
            continue

        embeddings = embedding_model.encode(ideas, show_progress_bar=False)

        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        grouped = defaultdict(list)
        for idea, label in zip(ideas, labels):
            grouped[label].append(idea)

        canonical = []
        for group in grouped.values():
            idea = select_canonical_idea(group)
            if idea:
                canonical.append(idea)

        cluster_canonical_ideas[cluster_id] = canonical

    return cluster_canonical_ideas


# ----------------------------
# Prompt & generation
# ----------------------------

def build_cluster_prompt(cluster_id, ideas, stats):
    ideas_text = "\n".join(f"- {i}" for i in ideas[:8])

    top = stats.get("top_products", [])
    worst = stats.get("worst_product")

    top_text = "\n".join(
        f"- Product {i+1}: many reviews ({p['total_reviews']}), low negatives ({p['negative']})"
        for i, p in enumerate(top)
    )

    worst_text = (
        f"- Highest negative ratio ({worst['neg_ratio']:.2f}) with "
        f"{worst['negative']} negatives out of {worst['total_reviews']} reviews."
        if worst else "- No clear worst product."
    )

    return f"""
You are writing a short, neutral recommendation article based only on the information below.

Rules:
- Do not mention brand names or product codes.
- Do not invent facts.
- Use general language.
- Write 2 short paragraphs (6–8 sentences total).

Overview ideas:
{ideas_text}

Top products:
{top_text}

Worst product:
{worst_text}
""".strip()


def generate_articles(cluster_canonical_ideas, cluster_product_stats):
    client = OpenAI()
    articles = {}

    for cluster_id, ideas in cluster_canonical_ideas.items():
        stats = cluster_product_stats.get(cluster_id, {})
        prompt = build_cluster_prompt(cluster_id, ideas, stats)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=220,
            temperature=0.3
        )

        articles[cluster_id] = response.output_text.strip()

    return articles


# ----------------------------
# Main
# ----------------------------

def main():
    product_clusters = load_product_clusters(CLUSTERS_PATH)
    final_reviews = build_final_reviews(REVIEWS_PATH, product_clusters)
    cluster_product_reviews = group_reviews(final_reviews)

    cluster_product_stats = rank_products(cluster_product_reviews)
    cluster_canonical_ideas = extract_canonical_ideas(final_reviews)

    articles = generate_articles(
        cluster_canonical_ideas,
        cluster_product_stats
    )

    for c, text in articles.items():
        print(f"\n===== CLUSTER {c} =====\n")
        print(text)


if __name__ == "__main__":
    main()
