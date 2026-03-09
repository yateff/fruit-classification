import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍊",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #f0ece3;
}

.stApp {
    background: #0d1117;
}

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 8vw, 5rem);
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #f5c842 0%, #f07c3a 50%, #e84545 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #6b7280;
    margin-top: 0.6rem;
}

/* ── Divider ── */
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #f5c84244, transparent);
    margin: 1.5rem 0 2rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1.5px dashed #f5c84255 !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #f5c842aa !important;
}
[data-testid="stFileUploader"] label {
    color: #9ca3af !important;
    font-size: 0.9rem !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #161b22 0%, #1a2030 100%);
    border: 1px solid #f5c84233;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 8px 40px #f5c84212;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.4rem;
}
.result-fruit {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f5c842, #f07c3a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.result-confidence {
    font-size: 1rem;
    color: #9ca3af;
    margin-top: 0.5rem;
}
.confidence-value {
    color: #f5c842;
    font-weight: 500;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #f5c842, #f07c3a) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: #1e2738 !important;
    border-radius: 99px !important;
    height: 8px !important;
}

/* ── Top-k table ── */
.topk-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid #1e2738;
    font-size: 0.9rem;
}
.topk-row:last-child { border-bottom: none; }
.topk-name {
    text-transform: capitalize;
    color: #d1d5db;
    font-weight: 400;
}
.topk-pct {
    color: #f5c842;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
.topk-bar-bg {
    flex: 1;
    height: 5px;
    background: #1e2738;
    border-radius: 99px;
    margin: 0 1rem;
    overflow: hidden;
}
.topk-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #f5c842, #f07c3a);
    transition: width 0.6s ease;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #1e2738;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 1rem;
}

/* ── Error / warning ── */
.stAlert {
    background: #1a1520 !important;
    border-color: #e84545 !important;
    border-radius: 12px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #9ca3af !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple", "Banana", "avocado", "cherry",
    "kiwi", "mango", "orange", "pinenapple",
    "strawberries", "watermelon"
]

IMG_SIZE = (455, 320)
MODEL_PATH = "fruit_classifier.keras"      # ← put your model file name here

FRUIT_EMOJI = {
    "Apple": "🍎", "Banana": "🍌", "avocado": "🥑", "cherry": "🍒",
    "kiwi": "🥝", "mango": "🥭", "orange": "🍊", "pinenapple": "🍍",
    "strawberries": "🍓", "watermelon": "🍉"
}

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path)

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))   # (W, H)
    arr = np.array(img, dtype=np.float32)[np.newaxis, ...]              # (1,H,W,3)
    probs = model.predict(arr, verbose=0)[0]                            # (num_classes,)
    return probs

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-sub">Deep Learning · Image Classification</p>
    <h1 class="hero-title">Fruit Classifier</h1>
</div>
<hr class="divider"/>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file **`{MODEL_PATH}`** not found.  \n"
        "Place your saved Keras model (`.h5` or SavedModel folder) "
        f"in the same directory as this script and rename it to `{MODEL_PATH}`."
    )
    st.stop()

with st.spinner("Loading model…"):
    model = load_model(MODEL_PATH)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Upload an image</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    pil_img = Image.open(uploaded)

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(pil_img, use_container_width=True)

    with col_res:
        with st.spinner("Analysing…"):
            probs = predict(model, pil_img)

        top_idx   = int(np.argmax(probs))
        top_name  = CLASS_NAMES[top_idx]
        top_prob  = float(probs[top_idx])
        emoji     = FRUIT_EMOJI.get(top_name, "🍑")

        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">Predicted fruit</p>
            <p class="result-fruit">{emoji} {top_name}</p>
            <p class="result-confidence">
                Confidence &nbsp;
                <span class="confidence-value">{top_prob*100:.1f}%</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Top-5 breakdown ───────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Top predictions</p>', unsafe_allow_html=True)

    top5_idx  = np.argsort(probs)[::-1][:5]
    max_prob  = float(probs[top5_idx[0]])

    rows_html = ""
    for rank, idx in enumerate(top5_idx):
        name  = CLASS_NAMES[idx]
        pct   = float(probs[idx]) * 100
        bar_w = int(pct / max(max_prob * 100, 1) * 100)
        bold  = "font-weight:600;color:#f0ece3;" if rank == 0 else ""
        rows_html += f"""
        <div class="topk-row">
            <span class="topk-name" style="{bold}">{FRUIT_EMOJI.get(name,'🍑')} {name}</span>
            <div class="topk-bar-bg">
                <div class="topk-bar-fill" style="width:{bar_w}%"></div>
            </div>
            <span class="topk-pct">{pct:.1f}%</span>
        </div>"""

    st.markdown(
        f'<div style="background:#161b22;border:1px solid #1e2738;'
        f'border-radius:16px;padding:1rem 1.5rem;">{rows_html}</div>',
        unsafe_allow_html=True
    )

else:
    st.markdown(
        '<p style="text-align:center;color:#4b5563;font-size:0.9rem;'
        'padding:2rem 0;">Drop a fruit photo above to get started.</p>',
        unsafe_allow_html=True
    )
