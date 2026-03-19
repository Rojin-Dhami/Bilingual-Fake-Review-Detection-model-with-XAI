# app.py
import os
import warnings
import numpy as np
import torch
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
matplotlib.use("Agg")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ── Load model ────────────────────────────────────────────────
MODEL_PATH = "./models/transformers/mBERT_finetuned"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH)
model      = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)
model.eval()
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(DEVICE)


def predict_proba(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) if not isinstance(t, str) else t
             for t in texts]
    texts = [t if t.strip() else "[UNK]" for t in texts]

    all_probs = []
    for i in range(0, len(texts), 8):
        batch  = texts[i:i+8]
        inputs = tokenizer(
            batch, return_tensors="pt",
            truncation=True, padding=True, max_length=128
        )
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(
            outputs.logits, dim=1
        ).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)

# ── SHAP setup ────────────────────────────────────────────────
masker    = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(
    lambda texts: predict_proba(texts),
    masker,
    output_names=["genuine", "fake"]
)

NEPALI_WORDS = {
    "thiyo", "cha", "ramro", "mito", "thikai", "dherai",
    "paisa", "aaunchu", "lagyo", "bhayo", "garna", "paryo",
    "bro", "yaar", "dami", "ekdum", "maja", "khana", "sewa"
}

SPECIAL_TOKENS = {
    "[CLS]","[SEP]","<s>","</s>","","[PAD]","▁","[UNK]"
}

def detect_language(text):
    words      = text.lower().split()
    nepali_cnt = sum(1 for w in words if w in NEPALI_WORDS)
    ratio      = nepali_cnt / max(len(words), 1)
    if ratio > 0.3:   return "🇳🇵 Romanized Nepali"
    elif ratio > 0.1: return "🔀 Code-mixed"
    else:             return "🇬🇧 English"



def analyze_rating(rating, text):
    text_lower = text.lower()
    pos_words  = {
        "good","great","amazing","excellent","best",
        "ramro","mito","dami","ekdum"
    }
    neg_words  = {
        "bad","worst","terrible","waste","horrible",
        "naramro","bekkar","kharab"
    }
    words     = set(text_lower.split())
    pos_count = len(words & pos_words)
    neg_count = len(words & neg_words)
    sentiment = (
        "positive" if pos_count > neg_count
        else "negative" if neg_count > pos_count
        else "neutral"
    )
    rating_sentiment = (
        "positive" if rating >= 4
        else "negative" if rating <= 2
        else "neutral"
    )
    inconsistent = (
        sentiment != "neutral" and
        sentiment != rating_sentiment
    )
    return inconsistent, sentiment

def make_confidence_plot(genuine_p, fake_p):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(
        ["Genuine", "Fake"],
        [genuine_p, fake_p],
        color=["#2ECC71", "#E74C3C"],
        alpha=0.85, edgecolor="white"
    )
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="gray",
               linestyle="--", alpha=0.6)
    ax.set_title(
        "Prediction Confidence", fontweight="bold"
    )
    ax.set_xlabel("Probability")
    for bar, val in zip(bars, [genuine_p, fake_p]):
        ax.text(
            min(val + 0.02, 0.92),
            bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=11
        )
    plt.tight_layout()
    return fig

def make_shap_plot(text, rating, shap_vals, is_fake):
    tokens = shap_vals[0].data
    values = shap_vals[0].values[:, 1]

    filtered = [
        (t.strip(), v) for t, v in zip(tokens, values)
        if t.strip() not in SPECIAL_TOKENS
    ]
    if not filtered:
        return None

    sorted_tokens = sorted(
        filtered, key=lambda x: abs(x[1]), reverse=True
    )[:15]

    # Add rating bar
    rating_influence = (
        0.05 if rating in [1,5]
        else -0.02
    )

    t_names = [f"⭐ rating={rating}"] + \
              [t for t,_ in sorted_tokens]
    t_vals  = [rating_influence] + \
              [v for _,v in sorted_tokens]
    colors  = [
        "#FFA500" if rating_influence >= 0 else "#27AE60"
    ] + [
        "#E74C3C" if v > 0 else "#2ECC71"
        for _, v in sorted_tokens
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(len(t_names)), t_vals,
        color=colors, alpha=0.85, edgecolor="white"
    )
    ax.axhline(y=0.5, color="gray",
               linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(t_names)))
    ax.set_yticklabels(t_names, fontsize=11)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_xlabel(
        "SHAP Value  "
        "(Red→FAKE | Green→GENUINE | Orange→Rating)",
        fontsize=10
    )
    verdict = "🚨 FAKE" if is_fake else "✅ GENUINE"
    ax.set_title(
        f"Token + Rating Importance — {verdict}\n"
        f"Rating: {'⭐'*rating} ({rating}/5)",
        fontsize=12, fontweight="bold"
    )

    max_val = max(abs(v) for v in t_vals) if t_vals else 1
    for bar, val in zip(bars, t_vals):
        ax.text(
            bar.get_width() + max_val * 0.03,
            bar.get_y() + bar.get_height()/2,
            f"{val:.5f}", va="center",
            fontsize=8, color="gray"
        )

    legend_items = [
        mpatches.Patch(color="#E74C3C", label="→ Fake"),
        mpatches.Patch(color="#2ECC71", label="→ Genuine"),
        mpatches.Patch(color="#FFA500", label="Rating"),
    ]
    ax.legend(handles=legend_items,
              loc="lower right", fontsize=9)

    preview = text[:70] + "..." if len(text) > 70 else text
    ax.text(
        0.01, -0.12, f'"{preview}"',
        transform=ax.transAxes,
        fontsize=8, style="italic", color="gray"
    )

    plt.tight_layout()
    return fig

# ── Main predict function ─────────────────────────────────────
def predict_and_explain(review_text, rating, explain):
    if not review_text or not review_text.strip():
        return "⚠️ Please enter a review.", None, None

    text   = review_text.strip()
    rating = int(rating)

    # Predict
    probs        = predict_proba([text])[0]
    genuine_prob = float(probs[0])
    fake_prob    = float(probs[1])

    # Rating analysis
    inconsistent, sentiment = analyze_rating(rating, text)
    rating_boost = 0.0
    if rating in [1, 5]:
        rating_boost += 0.03
    if inconsistent:
        rating_boost += 0.05

    fake_prob_adj    = min(fake_prob + rating_boost, 0.99)
    genuine_prob_adj = 1 - fake_prob_adj
    is_fake          = fake_prob_adj > 0.5
    confidence       = max(genuine_prob_adj, fake_prob_adj)
    lang             = detect_language(text)

    if confidence >= 0.90: conf_level = "Very High"
    elif confidence >= 0.75: conf_level = "High"
    elif confidence >= 0.60: conf_level = "Moderate"
    else: conf_level = "Low"

    # Rating flags
    rating_flags = []
    if rating == 5:
        rating_flags.append("⚠️ Perfect 5-star rating")
    if rating == 1:
        rating_flags.append("⚠️ Extreme 1-star rating")
    if inconsistent:
        rating_flags.append(
            f"⚠️ Rating ({rating}★) inconsistent "
            f"with text sentiment ({sentiment})"
        )

    flags_text = "\n".join(rating_flags) \
                 if rating_flags \
                 else f"✅ Rating consistent with text"

    # Result text
    verdict = "🚨 FAKE REVIEW DETECTED" \
              if is_fake else "✅ GENUINE REVIEW"

    result = f"""
## {verdict}

| Field | Value |
|-------|-------|
| 🎯 Prediction | {'FAKE' if is_fake else 'GENUINE'} |
| ⭐ Rating | {'⭐' * rating} ({rating}/5) |
| 📊 Fake Probability | {fake_prob_adj:.1%} |
| ✅ Genuine Probability | {genuine_prob_adj:.1%} |
| 💪 Confidence | {conf_level} ({confidence:.1%}) |
| 🌐 Language | {lang} |

**Rating Analysis:**
{flags_text}

**{'⚠️ This review shows patterns of fake reviews.' if is_fake else '✅ This review appears to be genuine.'}**
    """

    # Confidence plot
    conf_plot = make_confidence_plot(
        genuine_prob_adj, fake_prob_adj
    )

    # SHAP plot
    shap_plot = None
    if explain:
        try:
            shap_vals = explainer(
                [text], max_evals=300, batch_size=1
            )
            shap_plot = make_shap_plot(
                text, rating, shap_vals, is_fake
            )
        except Exception as e:
            print(f"SHAP error: {e}")

    return result, conf_plot, shap_plot

# ── Gradio interface ──────────────────────────────────────────
with gr.Blocks(
    title="Bilingual Fake Review Detector",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🔍 Bilingual Fake Review Detector
    ### English | Nepali | Code-mixed
    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            review_input  = gr.Textbox(
                label       = "📝 Review Text",
                placeholder = "Enter review here...",
                lines       = 5
            )
            rating_input  = gr.Slider(
                minimum = 1,
                maximum = 5,
                step    = 1,
                value   = 5,
                label   = "⭐ Rating (1-5)"
            )
            explain_check = gr.Checkbox(
                label = "🔍 Show SHAP Explanation",
                value = True
            )
            with gr.Row():
                submit_btn = gr.Button(
                    "🚀 Analyze",
                    variant="primary"
                )
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary"
                )

        with gr.Column(scale=3):
            result_output   = gr.Markdown()
            conf_plot_out   = gr.Plot(
                label="Confidence"
            )

    shap_plot_out = gr.Plot(
        label="🔍 SHAP Token + Rating Importance"
    )

    # Examples
    gr.Markdown("### 📌 Try These Examples")
    gr.Examples(
        examples=[
            [
                "ABSOLUTELY AMAZING!!! Best ever! "
                "Life changing! Buy now!!!",
                5, True
            ],
            [
                "khana thikai thiyo, dherai expensive "
                "lagyo tara atmosphere ramro cha",
                3, True
            ],
            [
                "ekdum best place, highly recommend "
                "everyone lai, ramro service",
                5, True
            ],
            [
                "service was okay, waited long but "
                "food came eventually",
                3, True
            ],
            [
                "TOTAL WASTE! Worst ever! "
                "Criminal fraud! Never again!!!",
                1, True
            ],
            [
                "driver late aayo tara ride comfortable "
                "thiyo, app sajilo cha",
                4, True
            ],
        ],
        inputs  = [review_input, rating_input, explain_check],
        outputs = [result_output, conf_plot_out, shap_plot_out],
        fn      = predict_and_explain,
        cache_examples = False,
    )

    submit_btn.click(
        fn      = predict_and_explain,
        inputs  = [review_input, rating_input, explain_check],
        outputs = [result_output, conf_plot_out, shap_plot_out]
    )

    clear_btn.click(
        fn      = lambda: ("", 5, True, None, None, None),
        inputs  = [],
        outputs = [
            review_input, rating_input, explain_check,
            result_output, conf_plot_out, shap_plot_out
        ]
    )

    gr.Markdown("""
    ---
    | Color | Meaning |
    |-------|---------|
    | 🔴 Red | Pushes toward FAKE |
    | 🟢 Green | Pushes toward GENUINE |
    | 🟠 Orange | Rating influence |
    """)

if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        inbrowser   = True,
    )
