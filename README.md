# 🔎 Tech Hunt

## NLP and Big Data Analytics Based Technology Product Recommendation System

Tech Hunt is a mini-project that combines **Natural Language Processing (NLP)** and **Big Data Analytics** to recommend technology products based on natural-language user queries.

The system processes product reviews using PySpark, performs NLP preprocessing and sentiment analysis, and uses TF-IDF and cosine similarity to recommend relevant products.

---

## 🚀 Features

- Natural-language product search
- PySpark-based data processing
- Product review aggregation
- NLP text preprocessing
- Sentiment analysis using VADER
- TF-IDF vectorization
- Cosine similarity
- Product recommendation scoring
- Product rating analytics
- Review-count analytics
- Rating vs review-count visualization
- Interactive Streamlit dashboard

---

## 🏗️ System Architecture

```text
Datafiniti Electronics Dataset
              ↓
       Google Colab
              ↓
          PySpark
              ↓
     Data Cleaning & ETL
              ↓
   processed_products.csv
              ↓
       NLP Processing
              ↓
     Sentiment Analysis
              ↓
      final_products.csv
              ↓
      TF-IDF Vectorization
              ↓
     Cosine Similarity
              ↓
   Recommendation Engine
              ↓
       Streamlit UI
              ↓
      Analytics Dashboard