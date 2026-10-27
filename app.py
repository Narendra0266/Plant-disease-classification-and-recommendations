from html import escape
from pathlib import Path
import difflib
import json
import os
import re

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.keras"
LABELS_PATH = ROOT / "class_names.json"
RECOMMENDATIONS_PATH = ROOT / "recommendations.json"

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
  [data-testid="stImage"] { display:flex; justify-content:center; }
  [data-testid="stImage"] img { border-radius:14px; width:auto !important; max-width:100% !important; height:auto !important; object-fit:contain; image-rendering:auto; }
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


@st.cache_data
def load_recommendations():
    return json.loads(RECOMMENDATIONS_PATH.read_text(encoding="utf-8"))

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


def reference_guidance(label):
    """Find the bundled reference record matching a model class."""
    plant, disease = label.split("_", 1)
    plant = plant.replace("EggPlant", "Eggplant")
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    candidates = [
        item for item in load_recommendations()
        if normalize(item.get("Plant", "")) == normalize(plant)
    ]
    if not candidates:
        return None
    match = max(
        candidates,
        key=lambda item: difflib.SequenceMatcher(
            None, normalize(disease), normalize(item.get("Disease", ""))
        ).ratio(),
    )
    return match


def create_gradcam_overlay(model, image, model_input, class_index):
    """Build an optional Grad-CAM overlay using the final convolution layer."""
    conv_layer = model.get_layer("conv2d_4")
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(model_input, training=False)
        class_score = predictions[:, class_index]
    gradients = tape.gradient(class_score, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(conv_output * weights, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap.numpy(), image.size, interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    return Image.blend(image, colored, 0.35)


def read_secret(name, default=None):
    value = os.environ.get(name, default)
    try:
        value = st.secrets.get(name, value)
    except Exception:
        # Local runs may not have a Streamlit secrets file configured.
        pass
    return value


def generate_llm_guidance(label, confidence, user_context, api_key, model_name):
    """Ask the OpenAI Responses API for cautious, contextual plant-care guidance."""
    is_healthy = "healthy" in label.lower()
    if is_healthy:
        response_sections = (
            "Use headings: Result explanation, Healthy crop prevention tips, and "
            "Related quantitative guidance. Explain that the image appears consistent "
            "with a healthy class, not that the whole crop is disease-free."
        )
    else:
        response_sections = (
            "Use headings: Detailed disease description, Symptoms to inspect, Prevention, "
            "Care and safe next steps, and Related quantitative guidance."
        )
    instructions = (
        "You are a cautious plant-care information assistant. The disease label is an "
        "image-classifier prediction, not a confirmed diagnosis; its score is not a "
        "calibrated probability. Do not claim certainty or invent symptoms. Give concise, "
        "low-risk next steps, what the user should inspect, and prevention suggestions. "
        f"{response_sections} "
        "For quantitative guidance, use only safe monitoring measures or values supported "
        "by the supplied context. Do not invent dosage, application rate, water amount, "
        "temperature threshold, or treatment schedule; if crop, location, or growing "
        "conditions are missing, say what should be measured and recommend local guidance. "
        "Do not recommend pesticide or fungicide brands, chemical doses, or treatment "
        "schedules. Encourage confirmation by a local agricultural extension service "
        "before treatment. State that advice can vary by crop and location."
    )
    prompt = (
        f"Predicted class: {label}\n"
        f"Model score: {confidence:.1%} (not calibrated)\n"
        f"User observations: {user_context or 'None provided'}\n\n"
        "Use the requested headings, make clear the label is not confirmed, and recommend "
        "professional confirmation before treatment."
    )
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model_name, "instructions": instructions, "input": prompt, "max_output_tokens": 500},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    text_parts = [
        part.get("text", "")
        for item in payload.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]
    answer = "\n".join(part for part in text_parts if part).strip()
    if not answer:
        raise ValueError("The AI service returned no text. Please try again.")
    return answer

st.markdown('<div class="section-label">Start with a leaf photo</div>', unsafe_allow_html=True)
upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
st.markdown('<div class="hint">JPG, PNG, or WEBP · One leaf in focus · Natural light works best</div>', unsafe_allow_html=True)

if upload is not None:
    try:
        labels = load_labels()
        crop_names = sorted({label.split("_", 1)[0].replace("EggPlant", "Eggplant") for label in labels})
        selected_crop = st.selectbox("Select crop type", crop_names, help="Choose the crop shown in the uploaded photo. Predictions will be ranked among this crop’s model classes.")
        recommendation_mode = st.radio(
            "Choose recommendation mode",
            ["Standard Advice", "LLM Mode"],
            horizontal=True,
            help="Standard Advice uses the bundled reference guide. LLM Mode creates tailored guidance and requires an API key.",
        )
        image, model_input = prepare_image(upload)
        with st.spinner("Checking the image…"):
            model = load_model()
            scores = np.asarray(model.predict(model_input, verbose=0))[0]
        if len(scores) != len(labels):
            st.error(f"Model has {len(scores)} outputs but the label file has {len(labels)} classes.")
        else:
            crop_indices = [
                index for index, label in enumerate(labels)
                if label.split("_", 1)[0].replace("EggPlant", "Eggplant") == selected_crop
            ]
            if not crop_indices:
                st.error(f"No model classes are available for {selected_crop}.")
                st.stop()
            crop_order = np.argsort(scores[crop_indices])[::-1]
            order = np.asarray([crop_indices[position] for position in crop_order])
            predicted_label = labels[int(order[0])]
            plant, condition = readable_label(predicted_label)
            confidence = float(scores[order[0]])
            left, right = st.columns([1.08, 1], gap="large", vertical_alignment="top")
            with left:
                with st.container(border=True):
                    st.markdown('<div class="photo-caption">YOUR PHOTO</div>', unsafe_allow_html=True)
                    st.image(image, width="stretch")
                    st.caption("The classifier analyzes a 75 × 75 copy of this photo. A sharp, well-lit original gives it the best detail available to the model.")
                    if min(image.size) < 500:
                        st.caption(
                            f"Photo size: {image.width} × {image.height} pixels. This source is small, "
                            "so it may look blocky when enlarged. For a clearer preview, upload the "
                            "original photo rather than a thumbnail or screenshot."
                        )
                    with st.expander("See what the model focused on (Grad-CAM)"):
                        if st.button("Generate focus map", key="generate_gradcam"):
                            try:
                                overlay = create_gradcam_overlay(model, image, model_input, int(order[0]))
                                st.image(overlay, caption="Heatmap overlay for the predicted class", width="stretch")
                                st.caption("Grad-CAM highlights image regions that influenced this prediction. It is an explanation aid, not proof that the model found the true disease cause.")
                            except Exception as exc:
                                st.info(f"Grad-CAM is unavailable for this model run: {exc}")
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
                st.subheader("Recommendations")
                if recommendation_mode == "Standard Advice":
                    guide = reference_guidance(predicted_label)
                    if guide:
                        st.caption("Reference guide · bundled with the Android app")
                        if "healthy" in predicted_label.lower():
                            sections = [
                                ("Healthy crop prevention tips", guide.get("Recommendations(for prevention)", "")),
                                ("Routine care", guide.get("Recommendation(For cure and take care)", "")),
                            ]
                        else:
                            sections = [
                                ("Symptoms", guide.get("Symptoms", "")),
                                ("Prevention", guide.get("Recommendations(for prevention)", "")),
                                ("Care and treatment", guide.get("Recommendation(For cure and take care)", "")),
                            ]
                        for title, detail in sections:
                            if detail.strip():
                                st.markdown(
                                    f'<div class="advice"><strong>{escape(title)}</strong><br>{escape(detail).replace(chr(10), "<br>")}</div>',
                                    unsafe_allow_html=True,
                                )
                        st.caption("These are reference suggestions, not a confirmed diagnosis. Treatment choices depend on crop, location, and local rules. Confirm with an agricultural expert and follow product labels before applying treatments.")
                    else:
                        st.info("No matching reference entry was found for this class. Ask a local agricultural expert to confirm the cause.")
                else:
                    st.caption("AI-generated guidance · review with a local agricultural expert before treatment")
                    user_context = st.text_input(
                        "Anything else about the plant? (optional)",
                        placeholder="For example: crop age, recent rain, or where spots first appeared",
                        key="llm_plant_context",
                    )
                    st.caption("When you generate guidance, the predicted class, model score, and your optional note are sent to the OpenAI API. Do not enter personal or sensitive information.")
                    api_key = read_secret("OPENAI_API_KEY")
                    model_name = read_secret("OPENAI_MODEL", "gpt-4.1-mini")
                    if not api_key:
                        st.info(
                            "To enable LLM Mode, add OPENAI_API_KEY to your app's Streamlit Cloud "
                            "Secrets. Reference Mode works without an API key."
                        )
                    elif st.button("Generate tailored guidance", type="primary"):
                        try:
                            with st.spinner("Preparing cautious plant-care guidance…"):
                                advice = generate_llm_guidance(
                                    predicted_label, confidence, user_context, api_key, model_name
                                )
                            st.markdown(advice)
                        except requests.RequestException as exc:
                            st.error(f"Could not contact the AI service: {exc}")
                        except (ValueError, KeyError) as exc:
                            st.error(f"Could not generate guidance: {exc}")
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

with st.expander("How the PlantCareNet workflow works"):
    steps = st.columns(4)
    workflow = [
        ("01 · Upload image", "Choose a clear photo of one leaf."),
        ("02 · Select crop", "Choose the crop; the app ranks its matching classes."),
        ("03 · Choose mode", "Use Standard Advice or request AI-generated guidance."),
        ("04 · View results", "See the prediction, care guidance, and optional focus map."),
    ]
    for column, (title, detail) in zip(steps, workflow):
        with column:
            st.markdown(f"**{title}**")
            st.caption(detail)
    st.info("The current notebook uses an 85/15 train-test split. It does not implement cross-validation or automated hyperparameter tuning, so the site does not report those as completed experiments.")
    evaluation = st.columns(2)
    confusion_path = ROOT / "outputs" / "confusion_matrix_last.png"
    roc_path = ROOT / "outputs" / "roc_curve_multiclass_final.png"
    if confusion_path.exists():
        with evaluation[0]:
            st.image(str(confusion_path), caption="Confusion matrix saved from the notebook", width="stretch")
    if roc_path.exists():
        with evaluation[1]:
            st.image(str(roc_path), caption="Multiclass ROC chart saved from the notebook", width="stretch")
    st.caption("These saved charts describe the notebook evaluation run; they are not recalculated for each uploaded photo.")

st.markdown('<div class="footer">PlantCareNet is an image-classification aid, not a confirmed diagnosis. Scores are not calibrated probabilities. Lighting, background, and photo quality can affect results. Confirm a suspected disease with a local agricultural expert before treatment.</div>', unsafe_allow_html=True)
