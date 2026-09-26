import re

import pandas as pd
import streamlit as st

from src.recommender import RecommendationEngine
from src.product_classifier import detect_product_type


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Tech Hunt",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #777;
        margin-bottom: 2rem;
    }

    .product-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CLEAN PRODUCT TITLE
# =========================================================

def clean_product_title(title):

    title = str(title)

    # Remove "(Renewed)"
    title = re.sub(
        r"\(\s*renewed\s*\)",
        "",
        title,
        flags=re.IGNORECASE
    )

    # Remove standalone "Renewed"
    title = re.sub(
        r"\brenewed\b",
        "",
        title,
        flags=re.IGNORECASE
    )

    # Remove duplicate spaces
    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    return title


# =========================================================
# LOAD RECOMMENDATION ENGINE
# =========================================================

@st.cache_resource
def load_engine():

    return RecommendationEngine(
        "data/tech_products.csv"
    )


engine = load_engine()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🔎 Tech Hunt")

    st.write(
        "AI-powered technology product "
        "recommendation system."
    )

    st.divider()

    st.subheader("How it works")

    st.write(
        """
        **1. Natural Language Query**

        Describe the product you need.

        **2. Product Detection**

        The system identifies the product category.

        **3. NLP Processing**

        TF-IDF converts product text into numerical vectors.

        **4. Similarity Matching**

        Cosine similarity compares your query
        with available products.

        **5. Hybrid Ranking**

        The final recommendation score combines
        text similarity and product-type matching.
        """
    )

    st.divider()

    st.subheader("Technology")

    st.write(
        """
        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - TF-IDF
        - Cosine Similarity
        - PySpark
        """
    )

    st.divider()

    st.caption(
        "Tech Hunt • NLP + Big Data Analytics"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🔎 Tech Hunt</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find technology products using natural language.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SEARCH INPUT
# =========================================================

query = st.text_input(
    "What are you looking for?",
    placeholder=(
        "Example: bluetooth speaker for home"
    )
)


# =========================================================
# SEARCH CONTROLS
# =========================================================

col1, col2 = st.columns([3, 1])


with col1:

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )


with col2:

    st.write("")

    search_button = st.button(
        "🔍 Search",
        use_container_width=True
    )


# =========================================================
# SEARCH
# =========================================================

if search_button:

    if not query.strip():

        st.warning(
            "Please enter what you are looking for."
        )

    else:

        # ---------------------------------------------
        # Run recommendation engine
        # ---------------------------------------------

        with st.spinner(
            "Analyzing your requirements..."
        ):

            results = engine.recommend(
                query,
                top_n=top_n
            )

        # ---------------------------------------------
        # Detect category
        # ---------------------------------------------

        detected_type = detect_product_type(
            query
        )

        # ---------------------------------------------
        # Results heading
        # ---------------------------------------------

        st.divider()

        st.subheader(
            "🎯 Recommendations"
        )

        if detected_type:

            formatted_type = (
                detected_type
                .replace("_", " ")
                .title()
            )

            st.info(
                f"Detected product category: "
                f"**{formatted_type}**"
            )

        else:

            st.info(
                "No specific product category detected. "
                "Results are based on text similarity."
            )

        # ---------------------------------------------
        # No results
        # ---------------------------------------------

        if results.empty:

            st.warning(
                "No matching products were found."
            )

        # ---------------------------------------------
        # Display recommendations
        # ---------------------------------------------

        for index, row in results.iterrows():

            # =========================================
            # PRODUCT TITLE
            # =========================================

            display_title = clean_product_title(
                row["TITLE"]
            )

            st.markdown(
                f"""
                <div class="product-card">

                <h3>
                {index + 1}. {display_title}
                </h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            # =========================================
            # MATCH SCORE
            # =========================================

            if "final_score" in results.columns:

                match_score = float(
                    row["final_score"]
                )

            else:

                match_score = float(
                    row["similarity"]
                )

            # =========================================
            # PRODUCT TYPE
            # =========================================

            product_type_id = (
                row["PRODUCT_TYPE_ID"]
            )

            if pd.isna(product_type_id):

                product_type_text = "Unknown"

            else:

                product_type_text = str(
                    int(product_type_id)
                )

            # =========================================
            # METRICS
            # =========================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Match Score",
                    f"{match_score:.1%}"
                )

            with col2:

                st.metric(
                    "Product Type",
                    product_type_text
                )

            # =========================================
            # PRODUCT DESCRIPTION
            # =========================================

            description = str(
                row["DESCRIPTION"]
            ).strip()

            bullet_points = str(
                row["BULLET_POINTS"]
            ).strip()

            if description:

                with st.expander(
                    "📄 View product description"
                ):

                    st.write(
                        description
                    )

            elif bullet_points:

                with st.expander(
                    "📄 View product details"
                ):

                    st.write(
                        bullet_points
                    )

            else:

                st.caption(
                    "No additional product information available."
                )

            st.divider()


# =========================================================
# LANDING PAGE
# =========================================================

else:

    st.divider()

    st.subheader(
        "💡 Try searching for"
    )

    example_col1, example_col2, example_col3 = (
        st.columns(3)
    )

    with example_col1:

        st.info(
            """
            💻 **Laptop**

            `laptop computer`
            """
        )

    with example_col2:

        st.info(
            """
            🔊 **Speaker**

            `bluetooth speaker`
            """
        )

    with example_col3:

        st.info(
            """
            🎧 **Headphones**

            `wireless headphones`
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Tech Hunt — NLP & Big Data Analytics Project"
)