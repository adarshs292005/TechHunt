from src.nlp import clean_text

text = """
This laptop is absolutely amazing! The performance is excellent,
but the battery life could be better.
"""

result = clean_text(text)

print("Original:")
print(text)

print("\nCleaned:")
print(result)