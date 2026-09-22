import pandas as pd
from src.recommender import RecommendationEngine

df = pd.read_csv("data/final_products.csv")

engine = RecommendationEngine(df)

query = "laptop for programming with good battery"

results = engine.recommend(query, top_n=5)

print(
    results[
        [
            "name",
            "brand",
            "average_rating",
            "review_count",
            "sentiment",
            "similarity",
            "match_score"
        ]
    ].to_string(index=False)
)