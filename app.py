from html import escape
from pathlib import Path
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.keras"
LABELS_PATH = ROOT / "class_names.json"

st.set_page_config(page_title="PlantCareNet | Leaf check", page_icon="🌿", layout="wide")
st.markdown("""
<style>
  :root { --forest:#183f2d; --leaf:#527e4b; --mint:#eaf2e6; --cream:#f7f8f2; --ink:#23342a; --muted:#66756a; --line:#dce5da; }
  .stApp { background:var(--cream); color:var(--ink); }
  header[data-testid="stHeader"] { display:none; }
  .block-container { max-width:1120px; padding:2rem 2rem 3rem; }
  h1,h2,h3 { color:var(--forest); letter-spacing:-.025em; }
  .brandbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:2.2rem; }
  .brand { display:flex; gap:12px; align-items:center; color:var(--forest); font-weight:750; font-size:1.15rem; }
  .brandmark { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; background:var(--mint); font-size:1.45rem; }
  .pill { border:1px solid var(--line); background:#fff; color:var(--leaf); padding:7px 12px; border-radius:99px; font-size:.8rem; font-weight:650; }
  .hero { position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; min-height:290px; border-radius:26px; padding:2.4rem 3rem; background:radial-gradient(circle at 85% 18%,#3b6845 0%,transparent 31%),linear-gradient(118deg,#173c2b 0%,#214d34 68%,#2c5b3c 100%); box-shadow:0 18px 42px rgba(24,63,45,.15); }
  .hero-copy { position:relative; z-index:1; max-width:650px; }
  .eyebrow { display:inline-flex; align-items:center; gap:8px; color:#cfe2c8; font-size:.76rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin-bottom:1rem; }
  .eyebrow:before { content:""; width:20px; height:1px; background:#a9ca9c; }
  .hero h1 { color:#fff; font-size:clamp(2.15rem,4.2vw,3.65rem); line-height:1.02; letter-spacing:-.045em; margin:0 0 1rem; max-width:610px; }
  .hero p { color:#d6e4d6; font-size:1.05rem; line-height:1.55; margin:0; max-width:540px; }
  .hero-tag { display:inline-block; margin-top:1.35rem; border:1px solid #ffffff42; color:#f0f7ec; border-radius:99px; padding:7px 12px; font-size:.8rem; }
  .hero-art { position:relative; width:245px; min-width:200px; height:220px; opacity:.98; }
  .hero-orbit { position:absolute; width:172px; height:172px; left:35px; top:22px; border:1px solid #d5e8ca44; border-radius:50%; background:#e5f1d80b; }
  .hero-stem { position:absolute; left:122px; top:52px; width:4px; height:145px; border-radius:5px; background:#d2e9c3; transform:rotate(-3deg); }
  .hero-leaf { position:absolute; width:88px; height:47px; border-radius:100% 0 100% 0; box-shadow:inset 0 0 0 1px #ffffff25,0 12px 20px #0b24142a; }
  .hero-leaf:after { content:""; position:absolute; height:1px; width:77%; left:12%; top:53%; background:#264e344f; transform:rotate(-28deg); }
  .leaf-a { left:47px; top:91px; background:#a8cf8e; transform:rotate(21deg); }
  .leaf-b { left:119px; top:54px; background:#d1e7a9; transform:rotate(-39deg) scaleX(-1); }
  .leaf-c { left:78px; top:143px; width:76px; height:40px; background:#78a96c; transform:rotate(-8deg); }
  .steps { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:1.3rem 0 2rem; }
  .step { display:flex; align-items:center; gap:11px; padding:12px 14px; background:#fff; border:1px solid var(--line); border-radius:14px; }
  .step-number { display:grid; place-items:center; width:31px; height:31px; flex:none; border-radius:10px; background:var(--mint); color:var(--leaf); font-size:.75rem; font-weight:800; }
  .step strong { display:block; color:var(--forest); font-size:.88rem; }
  .step small { display:block; color:var(--muted); font-size:.75rem; margin-top:2px; }
  .section-label { color:var(--forest); font-weight:700; margin:0 0 .5rem; }
  [data-testid="stFileUploader"] { border:1.5px dashed #9db59a; background:#fff; border-radius:18px; padding:1rem 1.1rem; }
  [data-testid="stFileUploader"] section { padding:0; }
  [data-testid="stFileUploaderDropzone"] { background:#f7faf5; border:0; }
  [data-testid="stFileUploaderDropzoneInstructions"] { color:#536657; }
  [data-testid="stFileUploaderDropzoneInstructions"] small { color:#718073; }
  [data-testid="stFileUploaderDropzone"] svg { color:#527e4b; }
  [data-testid="stFileUploader"] button { background:var(--forest); color:#fff; border:0; border-radius:10px; font-weight:650; }
  [data-testid="stFileUploader"] small { color:var(--muted); }
  .hint { color:var(--muted); font-size:.88rem; margin:.65rem 0 1.7rem; }
  .panel { background:#fff; border:1px solid var(--line); border-radius:20px; padding:1rem; box-shadow:0 8px 28px rgba(26,56,35,.055); }
  .photo-caption { color:var(--muted); font-size:.8rem; margin:.25rem 0 .6rem; }
  [data-testid="stImage"] img { border-radius:14px; }
  .result-card { height:100%; background:#fff; border:1px solid var(--line); border-radius:20px; padding:1.35rem; box-shadow:0 8px 28px rgba(26,56,35,.055); }
  .result-kicker { color:var(--leaf); font-size:.77rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; }
  .result-plant { color:var(--muted); font-size:1rem; margin-top:1.1rem; }
  .result-condition { color:var(--forest); font-size:1.8rem; font-weight:760; line-height:1.15; margin:.2rem 0 1rem; }
  .score-row { display:flex; justify-content:space-between; color:var(--muted); font-size:.85rem; margin-bottom:.45rem; }
  .score-track { height:9px; background:#edf1eb; border-radius:99px; overflow:hidden; }
  .score-fill { height:100%; background:linear-gradient(90deg,#7da66f,#315d3b); border-radius:99px; }
  .advice { background:#f3f7ef; border-radius:14px; padding:1rem; margin-top:1.25rem; color:#405547; font-size:.94rem; line-height:1.55; }
  .notice { background:#fff9e9; border:1px solid #f0dfac; border-radius:12px; padding:.85rem 1rem; margin-top:1rem; color:#6a5724; font-size:.9rem; }
  .footer { border-top:1px solid var(--line); margin-top:2.4rem; padding-top:1.1rem; color:var(--muted); font-size:.83rem; }
  div[data-testid="stMetric"] { background:#f3f7ef; border:1px solid var(--line); padding:14px; border-radius:14px; }
  @media (max-width:700px) {
    .block-container { padding:1rem .9rem 2rem; }
    .brandbar { margin-bottom:1.1rem; }
    .brand { font-size:1.03rem; gap:9px; }
    .brandmark { width:38px; height:38px; }
    .pill { padding:6px 9px; font-size:.68rem; }
    .hero { min-height:0; padding:1.5rem 1.2rem; border-radius:21px; }
    .hero-art { display:none; }
    .hero h1 { font-size:2.15rem; line-height:1.04; }
    .hero p { font-size:.96rem; }
    .hero-tag { font-size:.72rem; line-height:1.4; }
    .steps { gap:7px; margin:1rem 0 1.5rem; }
    .step { padding:9px 8px; gap:7px; border-radius:12px; }
    .step-number { width:28px; height:28px; border-radius:9px; }
    .step strong { font-size:.78rem; }
    .step small { display:none; }
    [data-testid="stFileUploader"] { padding:.7rem; }
    [data-testid="stFileUploader"] button { min-height:46px; padding:0 16px; }
    .hint { font-size:.8rem; line-height:1.45; margin:.55rem 0 1.2rem; }
    div[data-testid="stHorizontalBlock"] { flex-direction:column !important; gap:1rem !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:100% !important; }
    .result-card { padding:1.15rem; }
    .result-condition { font-size:1.55rem; overflow-wrap:anywhere; }
    .advice { font-size:.9rem; }
    .footer { margin-top:1.8rem; }
  }
  @media (max-width:380px) {
    .block-container { padding-left:.65rem; padding-right:.65rem; }
    .pill { font-size:.62rem; }
    .hero h1 { font-size:1.92rem; }
    .steps { gap:5px; }
    .step { padding:8px 6px; }
    .step strong { font-size:.72rem; }
  }
</style>
<div class="brandbar">
  <div class="brand"><span class="brandmark">🌱</span><span>PlantCareNet</span></div>
  <span class="pill">LEAF HEALTH CHECK</span>
</div>
<div class="hero">
  <div class="hero-copy">
    <div class="eyebrow">A clearer first look</div>
    <h1>Give every leaf a closer look.</h1>
    <p>Upload a clear photo and get a quick PlantCareNet prediction with practical next steps.</p>
    <span class="hero-tag">35 plant-health classes · Runs on your saved model</span>
  </div>
  <div class="hero-art" aria-hidden="true">
    <div class="hero-orbit"></div><div class="hero-stem"></div>
    <span class="hero-leaf leaf-a"></span><span class="hero-leaf leaf-b"></span><span class="hero-leaf leaf-c"></span>
  </div>
</div>
<div class="steps">
  <div class="step"><span class="step-number">01</span><span><strong>Upload</strong><small>Choose a clear leaf photo</small></span></div>
  <div class="step"><span class="step-number">02</span><span><strong>Analyze</strong><small>Model checks 35 classes</small></span></div>
  <div class="step"><span class="step-number">03</span><span><strong>Explore</strong><small>Review score and next steps</small></span></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the PlantCareNet model…")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_labels():
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))

def readable_label(label):
    parts = label.split("_", 1)
    plant = parts[0].replace("EggPlant", "Eggplant")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else ""
    return plant, condition

def prepare_image(upload):
    image = Image.open(upload).convert("RGB")
    rgb = np.asarray(image)
    # Match notebook preprocessing: cv2 BGR channels, 75×75, pixel values / 255.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (75, 75), interpolation=cv2.INTER_LINEAR)
    return image, np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

st.markdown('<div class="section-label">Start with a leaf photo</div>', unsafe_allow_html=True)
upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
st.markdown('<div class="hint">JPG, PNG, or WEBP · One leaf in focus · Natural light works best</div>', unsafe_allow_html=True)

if upload is not None:
    try:
        image, model_input = prepare_image(upload)
        with st.spinner("Checking the image…"):
            model = load_model()
            labels = load_labels()
            scores = np.asarray(model.predict(model_input, verbose=0))[0]
        if len(scores) != len(labels):
            st.error(f"Model has {len(scores)} outputs but the label file has {len(labels)} classes.")
        else:
            order = np.argsort(scores)[::-1]
            predicted_label = labels[int(order[0])]
            plant, condition = readable_label(predicted_label)
            confidence = float(scores[order[0]])
            left, right = st.columns([1.08, 1], gap="large", vertical_alignment="top")
            with left:
                with st.container(border=True):
                    st.markdown('<div class="photo-caption">YOUR PHOTO</div>', unsafe_allow_html=True)
                    st.image(image, width="stretch")
            with right:
                result_html = f'''<div class="result-card">
                  <div class="result-kicker">PlantCareNet result</div>
                  <div class="result-plant">{escape(plant)}</div>
                  <div class="result-condition">{escape(condition or predicted_label)}</div>
                  <div class="score-row"><span>Model score</span><strong>{confidence:.1%}</strong></div>
                  <div class="score-track"><div class="score-fill" style="width:{confidence*100:.1f}%"></div></div>
                </div>'''
                st.markdown(result_html, unsafe_allow_html=True)
                if confidence < 0.45:
                    st.markdown('<div class="notice">The model is uncertain. Try a sharper, closer photo, and compare with a local plant-health expert.</div>', unsafe_allow_html=True)
                elif "healthy" in predicted_label.lower():
                    st.markdown('<div class="advice">This image was classified as <strong>healthy</strong>. Keep monitoring the plant for changes.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="advice"><strong>Practical first steps</strong><br>Isolate the affected plant if possible, remove badly damaged leaves, avoid wetting foliage while watering, and improve airflow. Confirm the issue with a local agricultural extension service before applying any treatment.</div>', unsafe_allow_html=True)
            with st.expander("View other possible classes"):
                for idx in order[1:4]:
                    other_plant, other_condition = readable_label(labels[int(idx)])
                    st.write(f"{other_plant} — {other_condition}: {float(scores[idx]):.1%}")
    except Exception as exc:
        st.error(f"Could not analyze this image: {exc}")
else:
    st.markdown("""
    <div class="panel" style="display:flex;gap:18px;align-items:flex-start;margin-top:.3rem">
      <div class="brandmark">📷</div>
      <div><strong style="color:#183f2d">A good photo helps</strong><br><span style="color:#66756a">Fill the frame with one leaf, keep it steady, and avoid strong shadows. You’ll see the prediction here after upload.</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">PlantCareNet is an image-classification aid, not a confirmed diagnosis. Lighting, background, and photo quality can affect the result.</div>', unsafe_allow_html=True)
