from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        # ==================================================
        # PRODUCT INFORMATION
        # ==================================================

        self.df["product_text"] = (
            self.df["name"].fillna("") + " " +
            self.df["brand"].fillna("") + " " +
            self.df["primaryCategories"].fillna("")
        )

        # ==================================================
        # REVIEW INFORMATION
        # ==================================================

        self.df["review_text"] = (
            self.df["clean_reviews"].fillna("")
        )

        # ==================================================
        # TF-IDF FOR PRODUCT INFORMATION
        # ==================================================

        self.product_vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.product_matrix = (
            self.product_vectorizer.fit_transform(
                self.df["product_text"]
            )
        )

        # ==================================================
        # TF-IDF FOR REVIEWS
        # ==================================================

        self.review_vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.review_matrix = (
            self.review_vectorizer.fit_transform(
                self.df["review_text"]
            )
        )

    # ==================================================
    # RECOMMENDATION
    # ==================================================

    def recommend(self, query, top_n=5):

        query = query.lower().strip()

        # ==================================================
        # PRODUCT SIMILARITY
        # ==================================================

        query_product_vector = (
            self.product_vectorizer.transform(
                [query]
            )
        )

        product_similarity = cosine_similarity(
            query_product_vector,
            self.product_matrix
        ).flatten()

        # ==================================================
        # REVIEW SIMILARITY
        # ==================================================

        query_review_vector = (
            self.review_vectorizer.transform(
                [query]
            )
        )

        review_similarity = cosine_similarity(
            query_review_vector,
            self.review_matrix
        ).flatten()

        results = self.df.copy()

        results["product_similarity"] = (
            product_similarity
        )

        results["review_similarity"] = (
            review_similarity
        )

        # ==================================================
        # KEYWORD MATCHING
        # ==================================================

        query_words = [
            word
            for word in query.split()
            if len(word) > 2
        ]

        keyword_scores = []

        for _, row in results.iterrows():

            product_text = (
                str(row["name"]) + " " +
                str(row["brand"]) + " " +
                str(row["primaryCategories"])
            ).lower()

            matches = 0

            for word in query_words:

                if word in product_text:
                    matches += 1

            if len(query_words) > 0:

                keyword_score = (
                    matches / len(query_words)
                )

            else:

                keyword_score = 0

            keyword_scores.append(
                keyword_score
            )

        results["keyword_score"] = (
            keyword_scores
        )

        # ==================================================
        # RATING SCORE
        # ==================================================

        rating_score = (
            results["average_rating"] / 5
        )

        # ==================================================
        # FINAL RECOMMENDATION SCORE
        # ==================================================

        results["match_score"] = (
            results["product_similarity"] * 55
            +
            results["keyword_score"] * 25
            +
            results["review_similarity"] * 10
            +
            rating_score * 10
        )

        # ==================================================
        # COMPATIBILITY WITH APP.PY
        # ==================================================

        # app.py expects a column called "similarity".
        # We use product similarity as the main similarity value.

        results["similarity"] = (
            results["product_similarity"]
        )

        # ==================================================
        # SORT RESULTS
        # ==================================================

        results = results.sort_values(
            "match_score",
            ascending=False
        )

        # ==================================================
        # RETURN TOP PRODUCTS
        # ==================================================

        return results.head(top_n)