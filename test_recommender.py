from src.recommender import RecommendationEngine

DATASET = "data/tech_products.csv"

print("==============================")
print("       TECH HUNT")
print("==============================")

engine = RecommendationEngine(DATASET)

while True:

    query = input(
        "\nEnter your search "
        "(or type 'exit'): "
    )

    if query.lower() == "exit":
        break

    results = engine.recommend(
        query,
        top_n=5
    )

    print("\n==============================")
    print("       RECOMMENDATIONS")
    print("==============================")

    for index, row in results.iterrows():

        print(
            f"\n{index + 1}. "
            f"{row['TITLE']}"
        )

        print(
            f"Similarity: "
            f"{row['similarity']:.2%}"
        )

    print("\n==============================")