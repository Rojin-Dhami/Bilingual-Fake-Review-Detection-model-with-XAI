# predict.py
# Run: python predict.py

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
matplotlib.use("Agg")    # non-interactive backend

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
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════
def predict_proba(texts):
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
# SHAP PLOT GENERATOR
# ══════════════════════════════════════════════════════════════
SPECIAL_TOKENS = {
    "[CLS]","[SEP]","<s>","</s>",
    "","[PAD]","▁","[UNK]"
}

os.makedirs("./predictions", exist_ok=True)

def generate_shap_plot(text, shap_vals, save_path):
    tokens = shap_vals[0].data
    values = shap_vals[0].values[:, 1]

    # Filter special tokens
    filtered = [
        (t.strip(), v)
        for t, v in zip(tokens, values)
        if t.strip() not in SPECIAL_TOKENS
    ]

    if not filtered:
        print("  ⚠️ No tokens to plot")
        return

    # Sort by absolute value top 15
    sorted_tokens = sorted(
        filtered,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:15]

    t_names = [t for t, _ in sorted_tokens]
    t_vals  = [v for _, v in sorted_tokens]
    colors  = [
        "#E74C3C" if v > 0 else "#2ECC71"
        for v in t_vals
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(len(t_names)), t_vals,
        color=colors, alpha=0.85,
        edgecolor="white"
    )
    ax.set_yticks(range(len(t_names)))
    ax.set_yticklabels(t_names, fontsize=11)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel(
        "SHAP Value  "
        "(Red → pushes FAKE | Green → pushes GENUINE)",
        fontsize=10
    )
    ax.set_title(
        "Token Importance — Which words drove prediction?",
        fontsize=12, fontweight="bold"
    )

    # Value labels on bars
    max_val = max(abs(v) for v in t_vals) if t_vals else 1
    for bar, val in zip(bars, t_vals):
        ax.text(
            bar.get_width() + max_val * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center",
            fontsize=8, color="gray"
        )

    # Review text preview at bottom
    preview = text[:80] + "..." if len(text) > 80 else text
    ax.text(
        0.01, -0.13,
        f'Review: "{preview}"',
        transform  = ax.transAxes,
        fontsize   = 8,
        style      = "italic",
        color      = "gray"
    )

    # Legend
    legend_items = [
        mpatches.Patch(
            color="#E74C3C", label="→ Fake"
        ),
        mpatches.Patch(
            color="#2ECC71", label="→ Genuine"
        ),
    ]
    ax.legend(
        handles  = legend_items,
        loc      = "lower right",
        fontsize = 9
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ SHAP plot saved → {save_path}")

# ══════════════════════════════════════════════════════════════
# FULL PREDICT + EXPLAIN FUNCTION
# ══════════════════════════════════════════════════════════════
review_counter = 0

def predict_and_explain(text, explain=True):
    global review_counter
    review_counter += 1

    # ── Predict ───────────────────────────────────────────────
    probs        = predict_proba([text])[0]
    genuine_prob = float(probs[0])
    fake_prob    = float(probs[1])
    is_fake      = fake_prob > 0.5
    confidence   = max(genuine_prob, fake_prob)
    lang         = detect_language(text)

    # ── Confidence level ──────────────────────────────────────
    if confidence >= 0.90:
        conf_level = "Very High"
    elif confidence >= 0.75:
        conf_level = "High"
    elif confidence >= 0.60:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    # ── Print result ──────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  RESULT #{review_counter}")
    print("="*55)
    print(f"  Review     : {text[:60]}...")
    print(f"  Language   : {lang}")
    print("-"*55)

    if is_fake:
        print(f"  🚨 VERDICT : FAKE REVIEW")
    else:
        print(f"  ✅ VERDICT : GENUINE REVIEW")

    print(f"  Fake prob  : {fake_prob:.1%}")
    print(f"  Genuine    : {genuine_prob:.1%}")
    print(f"  Confidence : {conf_level} ({confidence:.1%})")
    print("="*55)

    # ── SHAP explanation ──────────────────────────────────────
    if explain:
        print("\n  Computing SHAP explanation...")
        print("  (Takes ~30 seconds...)")
        try:
            shap_vals = explainer(
                [text],
                max_evals  = 300,
                batch_size = 1,
            )

            # Print top tokens in terminal
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

            print("\n  Top influential tokens:")
            print(f"  {'Token':<20} {'SHAP':>10} "
                  f"{'Direction'}")
            print(f"  {'-'*45}")
            for t, v in sorted_tokens:
                direction = (
                    "→ FAKE"    if v > 0
                    else "→ GENUINE"
                )
                marker = "🔴" if v > 0 else "🟢"
                print(f"  {marker} {t:<18} "
                      f"{v:>10.5f}  {direction}")

            # Save SHAP plot
            save_path = (
                f"./predictions/"
                f"review_{review_counter:03d}_"
                f"{'FAKE' if is_fake else 'GENUINE'}_"
                f"shap.png"
            )
            generate_shap_plot(text, shap_vals, save_path)

        except Exception as e:
            print(f"  ⚠️ SHAP failed: {e}")

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  🔍 Bilingual Fake Review Detector")
print("  Supports: English | Nepali | Code-mixed")
print("  Type 'quit' to exit")
print("  Type 'noshap' before review to skip SHAP")
print("="*55)

while True:
    print()
    user_input = input("Enter review: ").strip()

    # Exit
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nGoodbye! 👋")
        break

    # Empty input
    if not user_input:
        print("⚠️ Please enter a review.")
        continue

    # Check if user wants to skip SHAP
    if user_input.lower().startswith("noshap "):
        text    = user_input[7:].strip()
        explain = False
    else:
        text    = user_input
        explain = True

    if not text:
        print("⚠️ Please enter review text after 'noshap'.")
        continue

    predict_and_explain(text, explain=explain)