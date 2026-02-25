import re
import random
from datetime import datetime, timedelta
import pandas as pd



def get_length_type(n):
    if n <= 10:
        return "very_short"   # tighten further
    elif n <= 28:
        return "short"
    elif n <= 65:
        return "medium"
    else:
        return "long"         # catches more reviews
    



# ── SLANG WORD LIST ───────────────────────────────────────────
SLANG_WORDS = [
    # English slang
    "lol", "omg", "wtf", "tbh", "ngl", "fr", "bruh", "legit", "lit", "fire",
    "goat", "slay", "vibe", "lowkey", "highkey", "bussin", "mid", "cap", "no cap",
    "bet", "fam", "bro", "sis", "sus", "yikes", "rn", "imo", "smh", "lmao",
    # Romanized Nepali slang
    "yaar", "sathi", "dami", "ekdam", "maja", "mast", "aba", "hai", "ni",
    "tyo", "kasto", "argh", "uff", "wah", "waah", "sahi", "jhakkas",
    # Code-mixed slang
    "totally ramro", "super dami", "ekdam fire", "legit ramro", "yaar seriously",
]

# ── RATING WEIGHTS per fake_type ──────────────────────────────
RATING_WEIGHTS = {
    "generic_bot":     {1: 5,  2: 5,  3: 10, 4: 30, 5: 50},
    "positive_spam":   {1: 2,  2: 3,  3: 5,  4: 20, 5: 70},
    "negative_attack": {1: 60, 2: 25, 3: 10, 4: 3,  5: 2 },
    "neutral_bot":     {1: 10, 2: 10, 3: 50, 4: 20, 5: 10},
    # fallback for genuine reviews
    "genuine":         {1: 5,  2: 10, 3: 20, 4: 35, 5: 30},
}


def detect_slang(text):
    """Returns (has_slang: bool, slang_count: int)"""
    text_lower = text.lower()
    count = 0
    for slang in SLANG_WORDS:
        if " " in slang:
            count += text_lower.count(slang)
        else:
            count += len(re.findall(rf"\b{re.escape(slang)}\b", text_lower))
    return (count > 0), count


def generate_rating(fake_type):
    """Realistic rating based on fake_type. Falls back to 'genuine' weights."""
    weights = RATING_WEIGHTS.get(fake_type, RATING_WEIGHTS["genuine"])
    choices = list(weights.keys())
    probs   = [weights[r] / sum(weights.values()) for r in choices]
    return random.choices(choices, weights=probs, k=1)[0]


def generate_date(start="2022-01-01", end="2025-06-30"):
    """Random date between start and end."""
    s     = datetime.strptime(start, "%Y-%m-%d")
    e     = datetime.strptime(end,   "%Y-%m-%d")
    delta = (e - s).days
    return (s + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def add_feature_columns(df, text_col="review_text", fake_type_col="fake_type"):
    """
    Adds 4 new columns to any review dataframe:
      - rating      : int (1-5) based on fake_type
      - date        : str (YYYY-MM-DD) random between 2022-2025
      - has_slang   : bool
      - slang_count : int

    Parameters:
      df           : your pandas DataFrame
      text_col     : column name containing the review text
      fake_type_col: column name containing fake_type (or 'genuine')

    Returns:
      df with 4 new columns added (does not modify original)
    """
    df = df.copy()

    # rating — use fake_type if column exists, else default to genuine weights
    if fake_type_col in df.columns:
        df["rating"] = df[fake_type_col].apply(generate_rating)
    else:
        df["rating"] = [generate_rating("genuine")] * len(df)

    # date
    df["date"] = [generate_date() for _ in range(len(df))]

    # slang
    slang_results       = df[text_col].apply(detect_slang)
    df["has_slang"]     = slang_results.apply(lambda x: x[0])
    df["slang_count"]   = slang_results.apply(lambda x: x[1])

    return df