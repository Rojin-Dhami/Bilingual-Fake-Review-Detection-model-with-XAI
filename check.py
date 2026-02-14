import json
from collections import Counter

def check_diversity(file_path,encoding="utf-8"):
    with open(file_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    
    # Check for repetitive sentence starters
    starters = [r['review_text'].split()[0].lower() for r in data if len(r['review_text'].split()) > 0]
    print("Top 5 Sentence Starters:", Counter(starters).most_common(5))
    
    # Check for average word length
    avg_words = sum(len(r['review_text'].split()) for r in data) / len(data)
    print(f"Average word count: {avg_words:.2f}")

# Usage:
check_diversity('data.json')