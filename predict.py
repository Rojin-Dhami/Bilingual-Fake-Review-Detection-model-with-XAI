# predict.py
import os
import warnings
import numpy as np
import torch
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
matplotlib.use("Agg")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "./models/transformers/mBERT_finetuned"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)
model.eval()
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(DEVICE)
print(f"✅ Model loaded on {DEVICE}")

# ══════════════════════════════════════════════════════════════
# RATING ANALYSIS
# ══════════════════════════════════════════════════════════════
def analyze_rating(rating, text):
    """
    Analyze if rating is consistent with review text
    Extreme ratings (1 or 5) are common in fake reviews
    """
    text_lower = text.lower()

    # Positive words
    positive_words = {
        "good", "great", "amazing", "excellent", "best",
        "awesome", "fantastic", "love", "perfect", "wonderful",
        "ramro", "mito", "dami", "ekdum", "maja"
    }
    # Negative words
    negative_words = {
        "bad", "worst", "terrible", "horrible", "awful",
        "waste", "poor", "disappointing", "never", "hate",
        "naramro", "bekkar", "faltu", "kharab"
    }

    words        = set(text_lower.split())
    pos_count    = len(words & positive_words)
    neg_count    = len(words & negative_words)
    text_sentiment = "positive" if pos_count > neg_count \
                     else "negative" if neg_count > pos_count \
                     else "neutral"

    # Check rating-text consistency
    rating_sentiment = "positive" if rating >= 4 \
                       else "negative" if rating <= 2 \
                       else "neutral"

    inconsistent = (
        text_sentiment != "neutral" and
        text_sentiment != rating_sentiment
    )

    # Rating-based fake signals
    rating_flags = []
    if rating == 5:
        rating_flags.append("⚠️ Perfect 5-star rating")
    if rating == 1:
        rating_flags.append("⚠️ Extreme 1-star rating")
    if inconsistent:
        rating_flags.append(
            f"⚠️ Rating ({rating}★) inconsistent "
            f"with text sentiment ({text_sentiment})"
        )

    return rating_flags, inconsistent, text_sentiment

# ══════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════
def predict_proba(texts):
    """SHAP-compatible predict function"""
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]
    texts = [
        str(t) if not isinstance(t, str) else t
        for t in texts
    ]
    texts = [t if t.strip() else "[UNK]" for t in texts]

    all_probs  = []
    batch_size = 8

    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors = "pt",
            truncation     = True,
            padding        = True,
            max_length     = 128
        )
        inputs = {
            k: v.to(DEVICE) for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(
            outputs.logits, dim=1
        ).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)

# ══════════════════════════════════════════════════════════════
# SHAP SETUP
# ══════════════════════════════════════════════════════════════
print("Setting up SHAP explainer...")
masker    = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(
    predict_proba,
    masker,
    output_names=["genuine", "fake"]
)
print("✅ SHAP explainer ready!\n")

# ══════════════════════════════════════════════════════════════
# LANGUAGE DETECTOR
# ══════════════════════════════════════════════════════════════
NEPALI_WORDS = {
    "thiyo", "cha", "ramro", "mito", "thikai", "dherai",
    "paisa", "aaunchu", "gardinu", "lagyo", "bhayo",
    "garna", "paryo", "bro", "yaar", "sathi", "dami",
    "ekdum", "maja", "khana", "sewa", "राम्रो", "खाना",
    "सेवा", "धेरै", "छ", "थियो"
}

def detect_language(text):
    words      = text.lower().split()
    nepali_cnt = sum(1 for w in words if w in NEPALI_WORDS)
    ratio      = nepali_cnt / max(len(words), 1)
    if ratio > 0.3:
        return "Romanized Nepali"
    elif ratio > 0.1:
        return "Code-mixed (Nepali + English)"
    else:
        return "English"

# ══════════════════════════════════════════════════════════════
# SHAP PLOT
# ══════════════════════════════════════════════════════════════
SPECIAL_TOKENS = {
    "[CLS]","[SEP]","<s>","</s>",
    "","[PAD]","▁","[UNK]"
}

os.makedirs("./predictions", exist_ok=True)

def generate_shap_plot(text, rating, shap_vals,
                        is_fake, save_path):
    tokens = shap_vals[0].data
    values = shap_vals[0].values[:, 1]

    filtered = [
        (t.strip(), v)
        for t, v in zip(tokens, values)
        if t.strip() not in SPECIAL_TOKENS
    ]

    if not filtered:
        return

    sorted_tokens = sorted(
        filtered, key=lambda x: abs(x[1]), reverse=True
    )[:15]

    t_names = [t for t, _ in sorted_tokens]
    t_vals  = [v for _, v in sorted_tokens]
    colors  = [
        "#E74C3C" if v > 0 else "#2ECC71"
        for v in t_vals
    ]

    # ── Rating bar (separate) ─────────────────────────────────
    # Estimate rating influence based on extreme values
    rating_influence = 0.0
    if rating == 5:
        rating_influence = 0.05     # slight push to fake
    elif rating == 1:
        rating_influence = 0.04
    elif rating in [2, 3, 4]:
        rating_influence = -0.02    # slight push to genuine

    # Add rating as special bar
    t_names_full = [f"⭐ rating={rating}"] + t_names
    t_vals_full  = [rating_influence]      + t_vals
    colors_full  = [
        "#FFA500" if rating_influence >= 0 else "#27AE60"
    ] + colors

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(len(t_names_full)), t_vals_full,
        color=colors_full, alpha=0.85,
        edgecolor="white"
    )

    # Dashed separator after rating bar
    ax.axhline(y=0.5, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5)

    ax.set_yticks(range(len(t_names_full)))
    ax.set_yticklabels(t_names_full, fontsize=11)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel(
        "SHAP Value  "
        "(Red → FAKE | Green → GENUINE | Orange → Rating)",
        fontsize=10
    )

    verdict = "🚨 FAKE" if is_fake else "✅ GENUINE"
    ax.set_title(
        f"Token + Rating Importance — {verdict}\n"
        f"Rating: {'⭐' * rating} ({rating}/5)",
        fontsize=12, fontweight="bold"
    )

    # Value labels
    max_val = max(abs(v) for v in t_vals_full) if t_vals_full else 1
    for bar, val in zip(bars, t_vals_full):
        ax.text(
            bar.get_width() + max_val * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center",
            fontsize=8, color="gray"
        )

    # Review preview
    preview = text[:75] + "..." if len(text) > 75 else text
    ax.text(
        0.01, -0.12,
        f'Review: "{preview}"',
        transform=ax.transAxes,
        fontsize=8, style="italic", color="gray"
    )

    # Legend
    legend_items = [
        mpatches.Patch(color="#E74C3C", label="→ Fake"),
        mpatches.Patch(color="#2ECC71", label="→ Genuine"),
        mpatches.Patch(color="#FFA500", label="Rating influence"),
    ]
    ax.legend(
        handles=legend_items, loc="lower right", fontsize=9
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ SHAP plot saved → {save_path}")

# ══════════════════════════════════════════════════════════════
# MAIN PREDICT + EXPLAIN
# ══════════════════════════════════════════════════════════════
review_counter = 0

def predict_and_explain(text, rating, explain=True):
    global review_counter
    review_counter += 1

    # ── Text prediction ───────────────────────────────────────
    probs        = predict_proba([text])[0]
    genuine_prob = float(probs[0])
    fake_prob    = float(probs[1])

    # ── Rating analysis ───────────────────────────────────────
    rating_flags, inconsistent, sentiment = analyze_rating(
        rating, text
    )

    # ── Adjust probability based on rating signals ────────────
    rating_boost = 0.0
    if rating in [1, 5]:
        rating_boost = 0.03     # slight push toward fake
    if inconsistent:
        rating_boost += 0.05    # stronger push if inconsistent

    # Final adjusted probability
    fake_prob_adj    = min(fake_prob + rating_boost, 0.99)
    genuine_prob_adj = 1 - fake_prob_adj
    is_fake          = fake_prob_adj > 0.5
    confidence       = max(genuine_prob_adj, fake_prob_adj)
    lang             = detect_language(text)

    # ── Confidence level ──────────────────────────────────────
    if confidence >= 0.90:
        conf_level = "Very High"
    elif confidence >= 0.75:
        conf_level = "High"
    elif confidence >= 0.60:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    # ── Print results ─────────────────────────────────────────
    stars = "⭐" * rating

    print("\n" + "="*60)
    print(f"  RESULT #{review_counter}")
    print("="*60)
    print(f"  Review   : {text[:55]}...")
    print(f"  Rating   : {stars} ({rating}/5)")
    print(f"  Language : {lang}")
    print("-"*60)

    if is_fake:
        print(f"  🚨 VERDICT    : FAKE REVIEW")
    else:
        print(f"  ✅ VERDICT    : GENUINE REVIEW")

    print(f"  Fake prob : {fake_prob_adj:.1%} "
          f"(text={fake_prob:.1%} + "
          f"rating boost={rating_boost:.1%})")
    print(f"  Genuine   : {genuine_prob_adj:.1%}")
    print(f"  Confidence: {conf_level} ({confidence:.1%})")

    # ── Rating flags ──────────────────────────────────────────
    if rating_flags:
        print("\n  ⚠️  Rating Signals:")
        for flag in rating_flags:
            print(f"     {flag}")
    else:
        print(f"\n  ✅ Rating ({rating}★) consistent "
              f"with text sentiment ({sentiment})")

    print("="*60)

    # ── SHAP ──────────────────────────────────────────────────
    if explain:
        print("\n  Computing SHAP explanation...")
        print("  (Takes ~30 seconds...)")
        try:
            shap_vals = explainer(
                [text],
                max_evals  = 300,
                batch_size = 1,
            )

            # Print top tokens
            tokens = shap_vals[0].data
            values = shap_vals[0].values[:, 1]

            filtered = [
                (t.strip(), v)
                for t, v in zip(tokens, values)
                if t.strip() not in SPECIAL_TOKENS
            ]
            sorted_tokens = sorted(
                filtered,
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]

            print(f"\n  Top influential tokens:")
            print(f"  {'Token':<20} {'SHAP':>10}  Direction")
            print(f"  {'─'*45}")
            for t, v in sorted_tokens:
                direction = "→ FAKE" if v > 0 else "→ GENUINE"
                marker    = "🔴" if v > 0 else "🟢"
                print(f"  {marker} {t:<18} "
                      f"{v:>10.5f}  {direction}")

            # Rating in SHAP context
            print(f"\n  ⭐ Rating signal:")
            if rating in [1, 5]:
                print(f"     Rating {rating}★ → slight push "
                      f"toward FAKE (+{rating_boost:.1%})")
            elif inconsistent:
                print(f"     Rating {rating}★ inconsistent "
                      f"with text → stronger fake signal "
                      f"(+{rating_boost:.1%})")
            else:
                print(f"     Rating {rating}★ consistent "
                      f"with text → no additional signal")

            # Save plot
            save_path = (
                f"./predictions/"
                f"review_{review_counter:03d}_"
                f"rating{rating}_"
                f"{'FAKE' if is_fake else 'GENUINE'}_"
                f"shap.png"
            )
            generate_shap_plot(
                text, rating, shap_vals,
                is_fake, save_path
            )

        except Exception as e:
            print(f"  ⚠️ SHAP failed: {e}")

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  🔍 Bilingual Fake Review Detector")
print("  Supports: English | Nepali | Code-mixed")
print()
print("  Commands:")
print("    Enter review text when prompted")
print("    Enter rating (1-5) when prompted")
print("    Type 'noshap' to skip explanation")
print("    Type 'quit' to exit")
print("="*60)

while True:
    print()

    # ── Get review text ───────────────────────────────────────
    review = input("📝 Enter review text: ").strip()

    if review.lower() in ["quit", "exit", "q"]:
        print("\nGoodbye! 👋")
        break

    if not review:
        print("⚠️ Please enter a review.")
        continue

    # Check noshap flag
    if review.lower().startswith("noshap "):
        review  = review[7:].strip()
        explain = False
    else:
        explain = True

    if not review:
        print("⚠️ Please enter review text.")
        continue

    # ── Get rating ────────────────────────────────────────────
    while True:
        rating_input = input(
            "⭐ Enter rating (1-5): "
        ).strip()
        try:
            rating = int(rating_input)
            if 1 <= rating <= 5:
                break
            else:
                print("⚠️ Rating must be between 1 and 5.")
        except ValueError:
            print("⚠️ Please enter a number (1-5).")

    predict_and_explain(review, rating, explain)