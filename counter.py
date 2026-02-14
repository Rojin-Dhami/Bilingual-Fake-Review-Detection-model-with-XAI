import json
import re

target_word = "competitor"   # change to the word you want to count
count = 0

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)   # assuming file contains a LIST of such objects

for review in data:
    text = review.get("review_text", "").lower()
    
    # tokenize words safely
    words = re.findall(r"\b\w+\b", text)
    
    count += words.count(target_word.lower())

print(f"'{target_word}' appears {count} times in review_text")
