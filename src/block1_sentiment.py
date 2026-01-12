"""
Block 1 - Sentiment Inference Dataset Builder

This script loads a trained sentiment model and applies it to Amazon reviews
to generate a sentiment-enriched dataset (single source of truth).
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ----------------------------
# Filters
# ----------------------------

def is_no_experience(text: str) -> bool:
    text = text.lower()
    patterns = [
        "did not watch",
        "didn't watch",
        "never watched",
        "haven't watched",
        "nothing to say",
        "no comment",
        "no comments",
        "this was a gift",
        "was a gift",
        "item was canceled",
        "order was canceled",
        "arrived on time",
        "just as described"
    ]
    return any(p in text for p in patterns)


# ----------------------------
# Model loading
# ----------------------------

def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ----------------------------
# Sentiment prediction
# ----------------------------

def predict_sentiment(text: str, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()

    return {
        "label": model.config.id2label[pred_id],
        "probabilities": probs.squeeze().cpu().tolist()
    }


# ----------------------------
# Dataset builder (core logic)
# ----------------------------

def build_sentiment_dataset(
    input_path: str,
    output_path: str,
    model_path: str,
    start_line: int,
    max_reviews: int
):
    tokenizer, model, device = load_model(model_path)
    processed = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin)):
            if i < start_line:
                continue

            if processed >= max_reviews:
                break

            review = json.loads(line)

            if "asin" not in review or "text" not in review:
                continue

            text = review["text"].strip()
            if not text:
                continue

            if is_no_experience(text):
                continue

            prediction = predict_sentiment(text, tokenizer, model, device)

            record = {
                "asin": review["asin"],
                "text": text,
                "sentiment": prediction["label"],
                "probs": {
                    "negative": prediction["probabilities"][0],
                    "neutral": prediction["probabilities"][1],
                    "positive": prediction["probabilities"][2]
                }
            }

            fout.write(json.dumps(record) + "\n")
            processed += 1

    print(f"Saved {processed} classified reviews to:")
    print(output_path)


# ----------------------------
# Main
# ----------------------------

def main():
    INPUT_PATH = "data/raw/Movies_and_TV.jsonl"
    OUTPUT_PATH = "data/processed/reviews_with_sentiment.jsonl"
    MODEL_PATH = "models/sentiment_model"

    START_LINE = 30_000
    MAX_REVIEWS = 50_000

    build_sentiment_dataset(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        model_path=MODEL_PATH,
        start_line=START_LINE,
        max_reviews=MAX_REVIEWS
    )


if __name__ == "__main__":
    main()

