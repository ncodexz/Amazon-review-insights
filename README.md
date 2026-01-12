# Amazon-review-insights
Classification, Clustering and reviews
amazon-review-insights/
│
├── data/
│   ├── raw/
│   │   └── Movies_and_TV.jsonl
│   ├── processed/
│   │   └── reviews_with_sentiment.jsonl   ← Bloque 1 output
│
├── models/
│   └── sentiment_model/
│       └── distilbert/
│
├── src/
│   ├── block1_sentiment.py
│   ├── block2_clustering.py
│   ├── block3_insights.py
│
├── notebooks/
│   └── exploration.ipynb   (opcional)
│
├── requirements.txt
├── README.md
└── .gitignore
