"""
app.py  –  SkinSense AI  (fixed: smooth LIME heatmap + working chatbot)
Run:  streamlit run app.py
"""

import os, io, requests
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

st.set_page_config(page_title="SkinSense AI", page_icon="🧴",
                   layout="centered", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg:    #070e14; --card: #0d1b25;
    --teal:  #00c6a2; --teal2: #007d68;
    --amber: #f5a623; --rose:  #e05a6b;
    --txt:   #e8f4f0; --muted: #6b8f8a;
    --bdr:   rgba(0,198,162,0.18);
}
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,section.main>div{
    background:var(--bg)!important;color:var(--txt)!important;
    font-family:'DM Sans',sans-serif!important}
[data-testid="stAppViewContainer"]::before{
    content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background:radial-gradient(ellipse 70% 50% at 15% 20%,rgba(0,198,162,.10) 0%,transparent 60%),
               radial-gradient(ellipse 60% 50% at 85% 75%,rgba(0,100,180,.10) 0%,transparent 60%);
    animation:mesh 12s ease-in-out infinite alternate}
@keyframes mesh{0%{opacity:.7;transform:scale(1)}100%{opacity:1;transform:scale(1.05)}}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"]{display:none!important}
.block-container{max-width:800px!important;padding:2rem 1.5rem!important}
.hero-wrap{text-align:center;padding:2.5rem 0 1.5rem}
.hero-eye{letter-spacing:.25em;text-transform:uppercase;font-size:.72rem;color:var(--teal);font-weight:500;margin-bottom:.5rem}
.hero-title{font-family:'Playfair Display',serif!important;font-size:3.2rem;font-weight:700;
    background:linear-gradient(135deg,#00c6a2,#7fffd4 45%,#f5a623);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 .6rem}
.hero-sub{font-size:1rem;color:var(--muted);font-weight:300;max-width:480px;margin:0 auto}
.hero-line{width:60px;height:2px;margin:1.4rem auto 0;
    background:linear-gradient(90deg,var(--teal),var(--amber));border-radius:2px}
hr{border-color:var(--bdr)!important;margin:1.8rem 0!important}
[data-testid="stFileUploader"]{background:var(--card)!important;
    border:1.5px dashed var(--teal2)!important;border-radius:16px!important}
[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span{color:var(--muted)!important}
.stButton>button{background:linear-gradient(135deg,var(--teal),#00a887)!important;
    color:#050e12!important;font-weight:600!important;border:none!important;
    border-radius:10px!important;box-shadow:0 4px 18px rgba(0,198,162,.25)!important;
    transition:transform .15s,box-shadow .15s!important}
.stButton>button:hover{transform:translateY(-2px)!important;
    box-shadow:0 8px 28px rgba(0,198,162,.4)!important}
[data-testid="stProgressBar"]>div>div{
    background:linear-gradient(90deg,var(--teal),var(--amber))!important;border-radius:4px!important}
[data-testid="stProgressBar"]>div{background:rgba(255,255,255,.06)!important;border-radius:4px!important}
.result-card{border-radius:16px;padding:1.3rem 1.8rem;margin:1rem 0;
    font-size:1.15rem;font-weight:600;text-align:center;backdrop-filter:blur(8px);
    animation:reveal .5s ease both}
@keyframes reveal{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.card-d{background:rgba(224,90,107,.12);border:1.5px solid rgba(224,90,107,.45);color:#f4a0ac}
.card-h{background:rgba(0,198,162,.10);border:1.5px solid rgba(0,198,162,.40);color:#7fffd4}
.chat-title{font-family:'Playfair Display',serif;text-align:center;font-size:1.5rem;
    font-weight:700;color:var(--txt);margin:1.5rem 0 .6rem}
.msg-bot{background:var(--card);border:1px solid var(--bdr);border-radius:14px 14px 14px 3px;
    padding:.75rem 1.1rem;margin:.5rem 0;color:var(--txt);max-width:82%;line-height:1.55;
    animation:pop .3s ease both}
.msg-usr{background:rgba(0,198,162,.12);border:1px solid rgba(0,198,162,.25);
    border-radius:14px 14px 3px 14px;padding:.75rem 1.1rem;
    margin:.5rem 0 .5rem auto;color:#b2fff0;max-width:82%;text-align:right;
    animation:pop .3s ease both}
@keyframes pop{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
[data-testid="stChatInput"] textarea{background:var(--card)!important;
    border:1.5px solid var(--bdr)!important;border-radius:12px!important;color:var(--txt)!important}
[data-testid="stChatInput"] textarea:focus{border-color:var(--teal)!important}
[data-testid="stCheckbox"] label{color:var(--txt)!important}
[data-testid="stCaptionContainer"],.stCaption,small{color:var(--muted)!important}
[data-testid="stAlert"]{background:rgba(245,166,35,.10)!important;
    border-color:rgba(245,166,35,.35)!important;border-radius:12px!important;color:#f5d79e!important}
[data-testid="stImage"] img{border-radius:14px!important;
    border:1px solid var(--bdr)!important;box-shadow:0 8px 32px rgba(0,0,0,.45)!important}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--teal2);border-radius:3px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE  — initialise before any widget
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {"prediction": None, "confidence": 0.0, "lime_img": None,
              "chat_history": [], "chat_started": False, "symptoms_sent": False,
              "groq_key_input": "", "mode": None, "scroll_to": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "skin_model.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _build_model(n):
    m = models.mobilenet_v2(weights=None)
    f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(f,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,n))
    return m

@st.cache_resource(show_spinner=False)
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    cls  = ckpt["class_names"]
    mdl  = _build_model(len(cls))
    mdl.load_state_dict(ckpt["model_state"])
    return mdl.eval().to(DEVICE), cls

# ══════════════════════════════════════════════════════════════════════════════
#  FACE DETECTION  (tight crop — face only, no background)
# ══════════════════════════════════════════════════════════════════════════════
def crop_face(pil_img, pad=0.20):
    import cv2
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    det  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        faces = det.detectMultiScale(gray, 1.05, 3, minSize=(40,40))
    if len(faces) == 0:
        return pil_img, None, arr          # fallback: full image
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    H,W     = arr.shape[:2]
    px,py   = int(w*pad), int(h*pad)
    x1,y1   = max(0,x-px), max(0,y-py)
    x2,y2   = min(W,x+w+px), min(H,y+h+py)
    # draw box on original for display
    vis = arr.copy()
    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,220,170),max(2,W//180))
    return Image.fromarray(arr[y1:y2,x1:x2]), (x1,y1,x2-x1,y2-y1), vis

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict(pil_img):
    mdl, cls = load_model()
    face, _, _ = crop_face(pil_img)
    t = preprocess(face.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(mdl(t), 1).cpu().numpy()[0]
    idx = int(p.argmax())
    return cls[idx], float(p[idx]), cls, p

def batch_predict(imgs):
    mdl, cls = load_model()
    t = torch.stack([preprocess(Image.fromarray(i.astype(np.uint8))) for i in imgs]).to(DEVICE)
    with torch.no_grad():
        return torch.softmax(mdl(t),1).cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTH LIME HEATMAP  — face-only, gaussian-smoothed
# ══════════════════════════════════════════════════════════════════════════════
def explain_lime(pil_img, n_samples=500, n_segments=80, confidence=0.5):
    from lime import lime_image as LI
    from skimage.segmentation import slic

    mdl, cls = load_model()
    face_pil, face_box, orig_vis = crop_face(pil_img)
    face_rgb = np.array(face_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    # ── Run LIME on face crop only ─────────────────────────────────────────────
    explainer   = LI.LimeImageExplainer()
    explanation = explainer.explain_instance(
        face_rgb, batch_predict,
        top_labels=1, hide_color=0,
        num_samples=n_samples,
        segmentation_fn=lambda x: slic(x, n_segments=n_segments,
                                        compactness=15, sigma=1,
                                        start_label=0)
    )

    pred_idx    = int(batch_predict(face_rgb[None])[0].argmax())
    label_name  = cls[pred_idx].upper()

    # ── Build smooth pixel-level heatmap ──────────────────────────────────────
    seg_map   = explanation.segments
    local_exp = dict(explanation.local_exp[pred_idx])
    heat_map  = np.zeros(seg_map.shape, dtype=np.float32)
    for seg_id, weight in local_exp.items():
        heat_map[seg_map == seg_id] = weight

    # Gaussian blur for smooth look
    heat_smooth = gaussian_filter(heat_map, sigma=12)

    # Normalise to [-1, 1]
    vmax = np.abs(heat_smooth).max() + 1e-8
    heat_norm = heat_smooth / vmax

    # ── Skin-only mask — wider range covering all Indian skin tones ───────────
    import cv2
    # Use HSV for robust skin detection across fair→wheatish→dusky→dark tones
    hsv  = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    H_ch = hsv[:,:,0].astype(float)
    S_ch = hsv[:,:,1].astype(float)
    V_ch = hsv[:,:,2].astype(float)
    # Hue 0–25 covers peach/tan/brown/dark-brown skin tones
    # Saturation 20–200 excludes white/grey backgrounds and very dark hair
    # Value > 40 excludes very dark hair and shadows
    skin_mask = (
        (H_ch >= 0)   & (H_ch <= 25)  &
        (S_ch >= 20)  & (S_ch <= 200) &
        (V_ch >= 40)
    ).astype(np.float32)

    # Also use YCrCb as second check (OR combination so darker tones aren't missed)
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    Cr, Cb = ycrcb[:,:,1].astype(float), ycrcb[:,:,2].astype(float)
    skin_ycrcb = (
        (Cr >= 125) & (Cr <= 180) &
        (Cb >= 70)  & (Cb <= 130)
    ).astype(np.float32)

    # Union of both masks — catches more Indian skin tones
    skin_mask = np.clip(skin_mask + skin_ycrcb, 0, 1)

    # Soft ellipse centred on face to further suppress background edges
    H, W = face_rgb.shape[:2]
    cy, cx = H * 0.48, W * 0.50          # slightly above centre (forehead)
    yy, xx = np.ogrid[:H, :W]
    ellipse = np.clip(1.2 - ((xx-cx)/(cx*0.95))**2 - ((yy-cy)/(cy*1.1))**2, 0, 1)

    combined_mask = skin_mask * ellipse
    combined_mask = gaussian_filter(combined_mask, sigma=9)
    combined_mask = combined_mask / (combined_mask.max() + 1e-8)

    # Apply mask
    heat_norm = heat_norm * combined_mask

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="#070e14")
    for ax in axes:
        ax.set_facecolor("#0d1b25")

    # Left: original with face box
    axes[0].imshow(orig_vis)
    note = "  (teal box = face region)" if face_box else "  (no face detected)"
    axes[0].set_title(f"Original{note}", fontsize=11, fontweight="bold",
                      color="#e8f4f0", pad=10)
    axes[0].axis("off")

    # Right: face + heatmap (skin pixels only)
    axes[1].imshow(face_rgb)

    pos = np.where(heat_norm > 0, heat_norm, 0)
    neg = np.where(heat_norm < 0, -heat_norm, 0)

    # ── Colour logic depends on what the model predicted ──────────────────────
    # LIME positive weights = regions that SUPPORT the predicted class
    # LIME negative weights = regions that OPPOSE the predicted class
    #
    # If DEHYDRATED predicted: positive → dehydrated zones (RED), negative → healthy (GREEN)
    # If HYDRATED predicted:   positive → healthy zones (GREEN), negative → dehydrated (RED)

    if label_name == "DEHYDRATED":
        dehydrated_heat = pos
        healthy_heat    = neg
    else:
        dehydrated_heat = neg
        healthy_heat    = pos

    # Boost contrast — raise to power <1 to amplify weak signals
    dehydrated_heat = np.power(dehydrated_heat, 0.55)
    healthy_heat    = np.power(healthy_heat,    0.55)

    # Re-apply skin mask after boosting
    dehydrated_heat = dehydrated_heat * combined_mask
    healthy_heat    = healthy_heat    * combined_mask

    # ── Confidence-aware scaling ───────────────────────────────────────────────
    # High confidence HYDRATED (e.g. 100%) → boost green, suppress red heavily
    # High confidence DEHYDRATED (e.g. 95%) → boost red, suppress green
    # Low confidence (50-70%) → show both equally
    #
    # dominant_alpha   = alpha for the "winning" colour (matches prediction)
    # opponent_alpha   = alpha for the "opposing" colour (suppressed by confidence)
    conf_strength  = (confidence - 0.5) * 2.0          # 0.0 at 50% → 1.0 at 100%
    conf_strength  = np.clip(conf_strength, 0.0, 1.0)

    dominant_alpha  = 0.45 + conf_strength * 0.45      # 0.45 → 0.90
    opponent_alpha  = 0.55 - conf_strength * 0.50      # 0.55 → 0.05  (nearly invisible at 100%)

    if label_name == "HYDRATED":
        green_alpha = dominant_alpha   # green is dominant
        red_alpha   = opponent_alpha   # red is suppressed
    else:
        red_alpha   = dominant_alpha   # red is dominant
        green_alpha = opponent_alpha   # green is suppressed

    # 🔴 RED = dehydrated / affected skin zones
    red_rgba = np.zeros((*face_rgb.shape[:2], 4), dtype=np.float32)
    red_rgba[..., 0] = np.clip(dehydrated_heat * 1.0,  0, 1)
    red_rgba[..., 1] = np.clip(dehydrated_heat * 0.05, 0, 1)
    red_rgba[..., 2] = 0.0
    red_rgba[..., 3] = np.clip(dehydrated_heat * red_alpha, 0, 1)

    # 🟢 GREEN = healthy / well-hydrated skin zones
    green_rgba = np.zeros((*face_rgb.shape[:2], 4), dtype=np.float32)
    green_rgba[..., 0] = 0.0
    green_rgba[..., 1] = np.clip(healthy_heat * 0.95, 0, 1)
    green_rgba[..., 2] = np.clip(healthy_heat * 0.20, 0, 1)
    green_rgba[..., 3] = np.clip(healthy_heat * green_alpha, 0, 1)

    axes[1].imshow(red_rgba)
    axes[1].imshow(green_rgba)

    title_col = "#f4a0ac" if label_name == "DEHYDRATED" else "#7fffd4"
    axes[1].set_title(f"Smooth LIME Heatmap — {label_name}", fontsize=11,
                      fontweight="bold", color=title_col, pad=10)
    axes[1].axis("off")

    # ── Colorbar legend — RED at TOP = dehydrated, GREEN at BOTTOM = healthy ──
    legend_ax = fig.add_axes([0.92, 0.15, 0.025, 0.70])
    # Build manual red→yellow→green gradient top-to-bottom
    grad_colors = np.zeros((256, 1, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0          # 0 = top (red/dehydrated) → 1 = bottom (green/healthy)
        if t < 0.5:
            # red → yellow
            grad_colors[i, 0] = [1.0, t * 2.0, 0.0]
        else:
            # yellow → green
            grad_colors[i, 0] = [1.0 - (t - 0.5) * 2.0, 1.0, 0.0]
    legend_ax.imshow(grad_colors, aspect='auto')
    legend_ax.set_xticks([])
    legend_ax.set_yticks([0, 128, 255])
    legend_ax.set_yticklabels(
        ['🔴 Dehydrated', 'Neutral', '🟢 Healthy'],
        fontsize=7, color='#aacfc8')
    legend_ax.tick_params(length=0)
    for sp in legend_ax.spines.values():
        sp.set_edgecolor('#1a3040')

    plt.subplots_adjust(right=0.90)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#070e14")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ══════════════════════════════════════════════════════════════════════════════
#  CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = """You are SkinBot, a friendly and professional AI skin-wellness assistant specialising in Indian skin types.

Your ONLY area of expertise is skin health — facial skin hydration, dehydration, moisturising, skin-care routines, ingredients, and skincare product recommendations.

Rules:
1. ONLY answer questions about skin health, hydration, dehydration, skincare, and beauty products.
2. For completely unrelated topics (cooking, coding, politics, etc.), say:
   "I'm only able to help with skin and skincare topics. Could we focus on your skin wellness?"

3. STRICT BRAND RULE — Read carefully:
   - NEVER mention any brand name unless the user's message contains words like:
     "brand", "product", "buy", "recommend a product", "which one", "what to buy", "suggest a product", "name a product".
   - If the user describes symptoms or asks for general advice → give ONLY ingredient-based or lifestyle tips.
     Example: Say "use a hyaluronic acid serum" NOT "try Minimalist Hyaluronic Acid Serum".
   - ONLY when the user explicitly asks for brands/products → recommend Indian brands:
     Minimalist, Dot & Key, Plum, mCaffeine, Mamaearth, WOW Skin Science, Forest Essentials,
     Kama Ayurveda, Biotique, Himalaya, Lakme, Lotus Herbals, Re'equil, The Derma Co,
     Aqualogica, Pilgrim, Fixderma, Cetaphil India.

4. When giving ingredient advice (no brands), structure it clearly:
   - What ingredient helps and why
   - Simple routine step (morning/night)
   - One lifestyle tip relevant to Indian climate (humidity, pollution, sun)

5. Never diagnose medical conditions — recommend a dermatologist for persistent or severe issues.
6. Be warm, supportive, empathetic, and non-judgmental.
7. Keep responses to 3–4 sentences unless asked for more detail.
8. Never be rude or condescending."""

def bot_opening(is_dehydrated):
    if is_dehydrated:
        return ("🔴 Your skin analysis shows signs of **dehydration**. "
                "I'm here to help! Would you like personalised recommendations "
                "to restore your skin's hydration?")
    return ("🟢 Great news — your skin looks **well-hydrated**! "
            "Would you like tips to maintain this healthy glow, "
            "or do you have any skin questions?")

def chat_with_bot(history, user_msg, api_key):
    if not api_key:
        return "⚠️ No API key — enter your Groq key in the sidebar.", history
    updated = history + [{"role":"user","content":user_msg}]
    try:
        r = requests.post(GROQ_URL,
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
            json={"model":GROQ_MODEL,
                  "messages":[{"role":"system","content":SYSTEM_PROMPT}]+updated,
                  "max_tokens":512,"temperature":0.7},
            timeout=30)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        reply = f"⚠️ Error: {e}"
    return reply, updated + [{"role":"assistant","content":reply}]

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — API key setup
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔑 Groq API Key")
    st.caption("Get a free key at [console.groq.com](https://console.groq.com/)")
    env_key = os.getenv("GROQ_API_KEY","")
    if env_key:
        st.success("✅ Key loaded from .env")
        GROQ_API_KEY = env_key
    else:
        key_input = st.text_input("Paste your key here", type="password",
                                   value=st.session_state.groq_key_input,
                                   placeholder="gsk_...")
        st.session_state.groq_key_input = key_input
        GROQ_API_KEY = key_input
        if key_input:
            st.success("✅ Key saved for this session")
        else:
            st.warning("No key — chatbot disabled")
    st.divider()
    st.caption("**SkinSense AI** v2.0\nFor demo/research only.")

# ══════════════════════════════════════════════════════════════════════════════
#  CHATBOT INPUT  — must be at TOP LEVEL (not inside any if block)
#  Streamlit requires st.chat_input at the page root
# ══════════════════════════════════════════════════════════════════════════════
chat_user_input = st.chat_input("Ask SkinBot anything about your skin…")

# Process chatbot input immediately (top-level)
if chat_user_input and st.session_state.chat_started:
    with st.spinner("SkinBot is thinking…"):
        reply, new_hist = chat_with_bot(
            st.session_state.chat_history, chat_user_input, GROQ_API_KEY)
    st.session_state.chat_history = new_hist
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-SCROLL JS helper
# ══════════════════════════════════════════════════════════════════════════════
def auto_scroll_to(anchor_id):
    st.markdown(f"""
    <script>
        setTimeout(function() {{
            var el = document.getElementById('{anchor_id}');
            if (el) {{ el.scrollIntoView({{behavior: 'smooth', block: 'start'}}); }}
        }}, 400);
    </script>
    """, unsafe_allow_html=True)

def anchor(anchor_id):
    st.markdown(f'<div id="{anchor_id}" style="scroll-margin-top:70px"></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eye">AI-Powered Skin Wellness</div>
  <h1 class="hero-title">SkinSense AI</h1>
  <p class="hero-sub">Instant facial hydration analysis powered by<br>
     deep learning &amp; explainable AI</p>
  <div class="hero-line"></div>
</div>""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  MODE SELECTOR — shown only if no mode chosen yet
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode is None:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <p style="color:#6b8f8a; font-size:1rem; margin-bottom:1.5rem;">
            What would you like to do today?
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#0d1b25;border:1.5px solid rgba(0,198,162,0.3);
             border-radius:16px;padding:1.8rem 1.2rem;text-align:center;margin-bottom:0.5rem">
            <div style="font-size:2.5rem">🔬</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e8f4f0;margin:.5rem 0 .3rem">
                Analyse My Skin</div>
            <div style="font-size:.85rem;color:#6b8f8a">Upload a selfie for hydration<br>
                analysis + XAI heatmap + chatbot</div>
        </div>""", unsafe_allow_html=True)
        if st.button("🔬 Start Analysis", use_container_width=True, type="primary", key="btn_analyse"):
            st.session_state.mode = "analyse"
            st.rerun()

    with c2:
        st.markdown("""
        <div style="background:#0d1b25;border:1.5px solid rgba(245,166,35,0.3);
             border-radius:16px;padding:1.8rem 1.2rem;text-align:center;margin-bottom:0.5rem">
            <div style="font-size:2.5rem">🤖</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e8f4f0;margin:.5rem 0 .3rem">
                Chat with SkinBot</div>
            <div style="font-size:.85rem;color:#6b8f8a">Ask any skin or hydration<br>
                question directly</div>
        </div>""", unsafe_allow_html=True)
        if st.button("🤖 Open SkinBot", use_container_width=True, key="btn_chat"):
            st.session_state.mode = "chat"
            if not st.session_state.chat_started:
                st.session_state.chat_history = [{"role":"assistant",
                    "content":"👋 Hi! I'm SkinBot, your personal skin wellness assistant. "
                              "Ask me anything about skin hydration, dehydration, routines, "
                              "or ingredients!"}]
                st.session_state.chat_started = True
            st.rerun()

    st.stop()

# ── Back button (always shown after mode is selected) ─────────────────────────
if st.button("← Back to Home", key="btn_back"):
    st.session_state.update({
        "mode": None, "prediction": None, "confidence": 0.0,
        "lime_img": None, "chat_history": [], "chat_started": False,
        "symptoms_sent": False
    })
    st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "analyse":

    # ── Upload ────────────────────────────────────────────────────────────────
    anchor("upload-section")
    st.subheader("📸 Upload Your Selfie")
    uploaded = st.file_uploader("Choose a clear, front-facing photo",
                                  type=["jpg","jpeg","png","webp"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns(2)
        with c1:
            st.image(pil_img, caption="Your photo", width=300)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔬 Analyse My Skin", type="primary", use_container_width=True):
                st.session_state.update({"chat_history":[],"chat_started":False,
                                          "symptoms_sent":False,"scroll_to":"xai-section"})
                with st.spinner("Detecting face & running analysis… (~20–40 s)"):
                    try:
                        lbl, conf, _, probs = predict(pil_img)
                        lime_img = explain_lime(pil_img, n_samples=500,
                                                n_segments=80, confidence=conf)
                        st.session_state.update({"prediction":lbl,"confidence":conf,
                                                  "lime_img":lime_img})
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.prediction:
        label = st.session_state.prediction
        conf  = st.session_state.confidence
        is_d  = label.lower() == "dehydrated"

        st.divider()
        anchor("result-section")
        st.subheader("📊 Analysis Results")
        cls = "card-d" if is_d else "card-h"
        em  = "💧🚫" if is_d else "💧✅"
        st.markdown(
            f'<div class="result-card {cls}">'
            f'{em} &nbsp; Skin Status: <b>{label.upper()}</b>'
            f' &nbsp;·&nbsp; {conf*100:.1f}% confidence</div>',
            unsafe_allow_html=True)
        st.progress(conf, text=f"Confidence: {conf*100:.1f}%")

        # Auto-scroll to XAI after analysis
        auto_scroll_to("xai-section")

        st.divider()
        anchor("xai-section")
        st.subheader("🔬 XAI — LIME Smooth Heatmap")
        st.caption("🔴 Red = dehydrated / affected zone  ·  🟢 Green = healthy / hydrated zone  ·  Face region only")
        if st.session_state.lime_img:
            st.image(st.session_state.lime_img, width=720)

        # Auto-scroll to chatbot after XAI loads
        auto_scroll_to("chatbot-section")

        # ── Chatbot ───────────────────────────────────────────────────────────
        st.divider()
        anchor("chatbot-section")
        st.markdown('<p class="chat-title">🤖 SkinBot</p>', unsafe_allow_html=True)
        st.caption("Your personal AI skin wellness assistant — only answers skin-related questions.")

        if not st.session_state.chat_started:
            st.session_state.chat_history = [{"role":"assistant",
                                               "content":bot_opening(is_d)}]
            st.session_state.chat_started = True

        for msg in st.session_state.chat_history:
            cls_m = "msg-bot" if msg["role"]=="assistant" else "msg-usr"
            icon  = "🤖 &nbsp;" if msg["role"]=="assistant" else ""
            end   = " &nbsp;👤" if msg["role"]=="user" else ""
            st.markdown(f'<div class="{cls_m}">{icon}{msg["content"]}{end}</div>',
                        unsafe_allow_html=True)

        if is_d and len(st.session_state.chat_history)==1 and not st.session_state.symptoms_sent:
            st.markdown("**🖐️ How do the highlighted areas feel? Select all that apply:**")
            opts = ["Skin feels tight","Skin feels dry / rough","Skin is flaky / peeling",
                    "Skin looks dull / lifeless","Skin feels oily / greasy",
                    "Fine lines visible","Redness / irritation","Pores look enlarged"]
            chosen = [o for i,o in enumerate(opts) if st.checkbox(o, key=f"s{i}")]
            if st.button("💬 Send symptoms to SkinBot", type="primary"):
                if chosen:
                    umsg = f"My skin feels: {', '.join(chosen)}. What ingredients and skincare routine steps would help?"
                    with st.spinner("SkinBot is thinking…"):
                        rep, nh = chat_with_bot(st.session_state.chat_history, umsg, GROQ_API_KEY)
                    st.session_state.chat_history  = nh
                    st.session_state.symptoms_sent = True
                    st.rerun()
                else:
                    st.warning("Please select at least one symptom.")
        elif not GROQ_API_KEY:
            st.info("💡 Enter your Groq API key in the **sidebar** to enable the chatbot.")

# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — CHAT ONLY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "chat":
    anchor("chatbot-section")
    st.markdown('<p class="chat-title">🤖 SkinBot</p>', unsafe_allow_html=True)
    st.caption("Your personal AI skin wellness assistant — only answers skin-related questions.")

    if not st.session_state.chat_started:
        st.session_state.chat_history = [{"role":"assistant",
            "content":"👋 Hi! I'm SkinBot, your personal skin wellness assistant. "
                      "Ask me anything about skin hydration, dehydration, routines, or ingredients!"}]
        st.session_state.chat_started = True

    for msg in st.session_state.chat_history:
        cls_m = "msg-bot" if msg["role"]=="assistant" else "msg-usr"
        icon  = "🤖 &nbsp;" if msg["role"]=="assistant" else ""
        end   = " &nbsp;👤" if msg["role"]=="user" else ""
        st.markdown(f'<div class="{cls_m}">{icon}{msg["content"]}{end}</div>',
                    unsafe_allow_html=True)

    if not GROQ_API_KEY:
        st.info("💡 Enter your Groq API key in the **sidebar** to enable the chatbot.")

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption("⚠️ SkinSense AI is for informational purposes only. "
           "Consult a dermatologist for persistent skin concerns.")