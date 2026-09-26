import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp import clean_text
from src.product_classifier import detect_product_type


class RecommendationEngine:

    def __init__(self, csv_path):

        print("================================")
        print("      TECH HUNT ENGINE")
        print("================================")

        # -----------------------------------------
        # Load dataset
        # -----------------------------------------

        print("Loading dataset...")

        self.df = pd.read_csv(csv_path)

        # -----------------------------------------
        # Handle missing text
        # -----------------------------------------

        for column in [
            "TITLE",
            "BULLET_POINTS",
            "DESCRIPTION"
        ]:

            if column in self.df.columns:

                self.df[column] = (
                    self.df[column]
                    .fillna("")
                    .astype(str)
                )

        # -----------------------------------------
        # Convert PRODUCT_TYPE_ID
        # -----------------------------------------

        self.df["PRODUCT_TYPE_ID"] = pd.to_numeric(
            self.df["PRODUCT_TYPE_ID"],
            errors="coerce"
        )

        # -----------------------------------------
        # Combine product information
        # -----------------------------------------

        self.df["PRODUCT_TEXT"] = (
            self.df["TITLE"]
            + " "
            + self.df["BULLET_POINTS"]
            + " "
            + self.df["DESCRIPTION"]
        )

        # -----------------------------------------
        # Clean text
        # -----------------------------------------

        print("Cleaning product text...")

        self.df["CLEAN_TEXT"] = (
            self.df["PRODUCT_TEXT"]
            .apply(clean_text)
        )

        # -----------------------------------------
        # TF-IDF
        # -----------------------------------------

        print("Building TF-IDF model...")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.df["CLEAN_TEXT"]
            )
        )

        print(
            "TF-IDF matrix:",
            self.tfidf_matrix.shape
        )

        print(
            "Products loaded:",
            len(self.df)
        )

        print("================================")

    # =================================================
    # PRODUCT TYPE MATCH
    # =================================================

    def product_type_match(
        self,
        title,
        product_type
    ):

        # No product type detected
        if product_type is None:
            return 0

        title = str(title)

        detected_type = detect_product_type(
            title
        )

        if detected_type == product_type:
            return 1

        return 0

    # =================================================
    # RECOMMEND
    # =================================================

    def recommend(
        self,
        query,
        top_n=5
    ):

        query = query.strip()

        # -----------------------------------------
        # Empty query
        # -----------------------------------------

        if not query:

            results = self.df.head(
                top_n
            ).copy()

            results["similarity"] = 0.0

            results["type_match"] = 0.0

            results["final_score"] = 0.0

            return results

        # -----------------------------------------
        # Detect product type
        # -----------------------------------------

        product_type = detect_product_type(
            query
        )

        print(
            f"\nQuery: {query}"
        )

        print(
            f"Detected product type: "
            f"{product_type}"
        )

        # -----------------------------------------
        # Clean query
        # -----------------------------------------

        cleaned_query = clean_text(
            query
        )

        # -----------------------------------------
        # Convert query to TF-IDF
        # -----------------------------------------

        query_vector = (
            self.vectorizer.transform(
                [cleaned_query]
            )
        )

        # -----------------------------------------
        # Cosine similarity
        # -----------------------------------------

        similarity_scores = (
            cosine_similarity(
                query_vector,
                self.tfidf_matrix
            ).flatten()
        )

        # -----------------------------------------
        # Create result dataframe
        # -----------------------------------------

        results = self.df.copy()

        results["similarity"] = (
            similarity_scores
        )

        # -----------------------------------------
        # Product type matching
        # -----------------------------------------

        results["type_match"] = results[
            "TITLE"
        ].apply(
            lambda title:
            self.product_type_match(
                title,
                product_type
            )
        )

        # -----------------------------------------
        # Hybrid recommendation score
        #
        # 70% text similarity
        # 30% product type match
        # -----------------------------------------

        results["final_score"] = (
            results["similarity"] * 0.70
            +
            results["type_match"] * 0.30
        )

        # -----------------------------------------
        # Sort by final recommendation score
        # -----------------------------------------

        results = (
            results
            .sort_values(
                by="final_score",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        return results