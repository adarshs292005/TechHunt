import pandas as pd
import streamlit as st
import plotly.express as px

from src.recommender import RecommendationEngine


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Tech Hunt",
    page_icon="🔎",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .product-card {
        padding: 22px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        margin-bottom: 18px;
        background-color: rgba(128, 128, 128, 0.05);
    }

    .product-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .product-brand {
        font-size: 15px;
        opacity: 0.8;
        margin-bottom: 12px;
    }

    .product-info {
        font-size: 15px;
        margin-top: 6px;
    }

    .score-box {
        text-align: center;
        padding: 12px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
    }

    .score-number {
        font-size: 28px;
        font-weight: 700;
    }

    .score-label {
        font-size: 13px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/final_products.csv"
    )


df = load_data()


# ==================================================
# CREATE RECOMMENDATION ENGINE
# ==================================================

@st.cache_resource
def create_engine(dataframe):

    return RecommendationEngine(dataframe)


engine = create_engine(df)


# ==================================================
# HEADER
# ==================================================

st.title("🔎 Tech Hunt")

st.markdown(
    """
    ### NLP + Big Data Analytics Based Technology Product Recommendation System

    Find technology products using **natural-language search**,
    review analysis and **TF-IDF based recommendation**.
    """
)


# ==================================================
# SEARCH SECTION
# ==================================================

st.subheader("🔍 Find Your Product")

query = st.text_input(
    "Describe what you are looking for",
    placeholder="Example: wireless bluetooth speaker"
)


# ==================================================
# RECOMMENDATIONS
# ==================================================

if query:

    results = engine.recommend(
        query,
        top_n=5
    )

    st.subheader("🎯 Recommended Products")

    for _, product in results.iterrows():

        st.markdown(
            '<div class="product-card">',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(
            [4, 1]
        )

        # ------------------------------------------
        # PRODUCT INFORMATION
        # ------------------------------------------

        with col1:

            st.markdown(
                f"""
                <div class="product-title">
                    {product['name']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="product-brand">
                    🏷️ {product['brand']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="product-info">
                    📂 <b>Category:</b>
                    {product['primaryCategories']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="product-info">
                    ⭐ <b>Average Rating:</b>
                    {product['average_rating']}/5
                    &nbsp;&nbsp;&nbsp;
                    📝 <b>Reviews:</b>
                    {int(product['review_count'])}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="product-info">
                    😊 <b>Review Sentiment:</b>
                    {product['sentiment']}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ------------------------------------------
        # RECOMMENDATION SCORE
        # ------------------------------------------

        with col2:

            st.markdown(
                '<div class="score-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="score-number">
                    {product['match_score']:.1f}
                </div>

                <div class="score-label">
                    Recommendation Score
                </div>

                <br>

                <div class="score-label">
                    Text Similarity
                </div>

                <div style="font-size:20px;font-weight:600;">
                    {product['similarity']:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ==================================================
# ANALYTICS DASHBOARD
# ==================================================

st.divider()

st.header("📊 Product Analytics")


# ==================================================
# SUMMARY METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Products",
        len(df)
    )

with col2:

    st.metric(
        "Total Reviews",
        int(df["review_count"].sum())
    )

with col3:

    st.metric(
        "Average Rating",
        f"{df['average_rating'].mean():.2f}/5"
    )


# ==================================================
# RATING DISTRIBUTION
# ==================================================

st.subheader("⭐ Product Rating Distribution")

rating_counts = (
    df["average_rating"]
    .round(1)
    .value_counts()
    .sort_index()
    .reset_index()
)

rating_counts.columns = [
    "Rating",
    "Products"
]

fig_rating = px.bar(
    rating_counts,
    x="Rating",
    y="Products",
    title="Distribution of Average Product Ratings"
)

st.plotly_chart(
    fig_rating,
    use_container_width=True
)


# ==================================================
# REVIEW COUNT
# ==================================================

st.subheader("📝 Products by Review Count")

review_data = (
    df[
        [
            "name",
            "review_count"
        ]
    ]
    .sort_values(
        "review_count",
        ascending=False
    )
    .head(10)
)

fig_reviews = px.bar(
    review_data,
    x="review_count",
    y="name",
    orientation="h",
    title="Top 10 Products by Number of Reviews"
)

st.plotly_chart(
    fig_reviews,
    use_container_width=True
)


# ==================================================
# RATING VS REVIEW COUNT
# ==================================================

st.subheader("📈 Rating vs Review Count")

fig_scatter = px.scatter(
    df,
    x="review_count",
    y="average_rating",
    hover_name="name",
    title="Relationship Between Reviews and Average Rating"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Tech Hunt | NLP + Big Data Analytics Project"
)