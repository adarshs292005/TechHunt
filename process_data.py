import pandas as pd

from src.nlp import clean_text
from src.sentiment import analyze_sentiment


# --------------------------------------------------
# LOAD PYSPARK OUTPUT
# --------------------------------------------------

df = pd.read_csv(
    "data/processed_products.csv"
)

print("Products loaded:", len(df))


# --------------------------------------------------
# NLP TEXT CLEANING
# --------------------------------------------------

df["clean_reviews"] = df["all_reviews"].apply(
    clean_text
)


# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

df["sentiment"] = df["all_reviews"].apply(
    analyze_sentiment
)


# --------------------------------------------------
# SAVE FINAL DATASET
# --------------------------------------------------

df.to_csv(
    "data/final_products.csv",
    index=False
)


print("\n✅ NLP processing completed")
print("✅ Saved: data/final_products.csv")


# --------------------------------------------------
# CHECK RESULTS
# --------------------------------------------------

print("\nProduct-level sentiment:")

print(
    df["sentiment"].value_counts()
)


print("\nSample products:")

print(
    df[
        [
            "name",
            "average_rating",
            "review_count",
            "sentiment"
        ]
    ].head()
)