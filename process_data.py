import pandas as pd

from src.nlp import clean_text
from src.sentiment import analyze_sentiment


# ==================================================
# FILE PATHS
# ==================================================

INPUT_FILE = "data/amazon.csv"
OUTPUT_FILE = "data/final_products.csv"


# ==================================================
# LOAD DATA
# ==================================================

print("Loading Amazon dataset...")

df = pd.read_csv(INPUT_FILE)

print("Rows loaded:", len(df))
print("Columns:", len(df.columns))


# ==================================================
# CLEAN COLUMN NAMES
# ==================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


# ==================================================
# REMOVE COMPLETELY EMPTY ROWS
# ==================================================

df = df.dropna(how="all")


# ==================================================
# PRODUCT NAME
# ==================================================

df["product_name"] = (
    df["product_name"]
    .fillna("Unknown Product")
    .astype(str)
    .str.strip()
)


# ==================================================
# CATEGORY
# ==================================================

df["category"] = (
    df["category"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)


# ==================================================
# PRICE CLEANING
# ==================================================

def clean_price(value):

    if pd.isna(value):
        return 0.0

    value = str(value)

    value = (
        value
        .replace("₹", "")
        .replace(",", "")
        .strip()
    )

    try:
        return float(value)

    except ValueError:
        return 0.0


df["discounted_price"] = (
    df["discounted_price"]
    .apply(clean_price)
)

df["actual_price"] = (
    df["actual_price"]
    .apply(clean_price)
)


# ==================================================
# DISCOUNT PERCENTAGE
# ==================================================

def clean_discount(value):

    if pd.isna(value):
        return 0.0

    value = str(value)

    value = (
        value
        .replace("%", "")
        .strip()
    )

    try:
        return float(value)

    except ValueError:
        return 0.0


df["discount_percentage"] = (
    df["discount_percentage"]
    .apply(clean_discount)
)


# ==================================================
# RATING
# ==================================================

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)

df["rating"] = (
    df["rating"]
    .fillna(0)
)


# ==================================================
# RATING COUNT
# ==================================================

df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["rating_count"] = pd.to_numeric(
    df["rating_count"],
    errors="coerce"
)

df["rating_count"] = (
    df["rating_count"]
    .fillna(0)
)


# ==================================================
# PRODUCT DESCRIPTION
# ==================================================

df["about_product"] = (
    df["about_product"]
    .fillna("")
    .astype(str)
)


# ==================================================
# REVIEW TITLE
# ==================================================

df["review_title"] = (
    df["review_title"]
    .fillna("")
    .astype(str)
)


# ==================================================
# REVIEW CONTENT
# ==================================================

df["review_content"] = (
    df["review_content"]
    .fillna("")
    .astype(str)
)


# ==================================================
# COMBINED TEXT
# ==================================================

df["combined_text"] = (
    df["product_name"] + " " +
    df["category"] + " " +
    df["about_product"] + " " +
    df["review_title"] + " " +
    df["review_content"]
)


# ==================================================
# NLP TEXT
# ==================================================

print("Cleaning text...")

df["clean_text"] = (
    df["combined_text"]
    .apply(clean_text)
)


# ==================================================
# SENTIMENT
# ==================================================

print("Analyzing sentiment...")

df["sentiment"] = (
    df["review_content"]
    .apply(analyze_sentiment)
)


# ==================================================
# DISCOUNT AMOUNT
# ==================================================

df["discount_amount"] = (
    df["actual_price"] -
    df["discounted_price"]
)

df["discount_amount"] = (
    df["discount_amount"]
    .clip(lower=0)
)


# ==================================================
# REMOVE INVALID PRODUCTS
# ==================================================

df = df[
    df["product_name"].str.strip() != ""
]


# ==================================================
# SELECT FINAL DATA
# ==================================================

final_columns = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "discount_amount",
    "rating",
    "rating_count",
    "about_product",
    "review_title",
    "review_content",
    "clean_text",
    "sentiment",
    "img_link",
    "product_link"
]


df = df[
    [
        column
        for column in final_columns
        if column in df.columns
    ]
]


# ==================================================
# SAVE PROCESSED DATA
# ==================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# SUMMARY
# ==================================================

print("\n========================================")
print("       TECH HUNT DATA PROCESSING")
print("========================================")

print(
    "\nProducts:",
    len(df)
)

print(
    "Average rating:",
    round(df["rating"].mean(), 2)
)

print(
    "Average discount:",
    round(
        df["discount_percentage"].mean(),
        2
    ),
    "%"
)

print(
    "Categories:",
    df["category"].nunique()
)

print(
    "\nSentiment:"
)

print(
    df["sentiment"].value_counts()
)

print(
    "\nSaved:",
    OUTPUT_FILE
)

print(
    "\nProcessing completed successfully."
)
