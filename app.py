import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import pickle
import joblib
import time
import os
from collections import deque
from gtts import gTTS
import tempfile
from dotenv import load_dotenv
from groq import Groq
from tensorflow.keras.models import load_model

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SignBridge",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    color: white;
}
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b2fff, #00d4ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite;
    margin-bottom: 0;
}
@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}
.sign-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.sign-display {
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d4ff;
    text-align: center;
    letter-spacing: 2px;
}
.conf-display {
    font-size: 1rem;
    color: #7b2fff;
    text-align: center;
    margin-top: -0.3rem;
}
.sentence-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    min-height: 60px;
    font-size: 1rem;
    color: white;
    word-spacing: 4px;
}
.llm-box {
    background: rgba(123,47,255,0.1);
    border: 1px solid rgba(123,47,255,0.4);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-size: 1rem;
    color: #c9a7ff;
    font-style: italic;
    min-height: 60px;
}
.status-pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.pill-motion {
    background: rgba(0,212,255,0.15);
    border: 1px solid #00d4ff;
    color: #00d4ff;
}
.pill-static {
    background: rgba(255,200,0,0.15);
    border: 1px solid #ffc800;
    color: #ffc800;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    padding: 0.6rem;
    transition: all 0.2s;
}
div[data-testid="column"]:nth-child(1) .stButton > button {
    background: linear-gradient(135deg, #7b2fff, #5500cc);
    color: white;
}
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: rgba(255,255,255,0.08);
    color: #ccc;
    border: 1px solid rgba(255,255,255,0.15);
}
div[data-testid="column"]:nth-child(3) .stButton > button {
    background: rgba(255,50,50,0.15);
    color: #ff6b6b;
    border: 1px solid rgba(255,50,50,0.3);
}
[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(0,212,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────
for key, val in {
    "sentence"           : [],
    "llm_output"         : "",
    "last_sign"          : None,
    "last_sign_time"     : 0,
    "spell_mode"         : False,
    # ── Persistent buffers — survive Streamlit reruns ──
    "frame_buffer"       : deque(maxlen=25),
    "motion_history"     : deque(maxlen=10),
    "static_hold"        : deque(maxlen=15),
    "prev_landmarks"     : None,
    "hand_visible_frames": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── CONFIG ───────────────────────────────────────────────────────
STATIC_CONF       = 0.60
MOTION_CONF       = 0.92
COOLDOWN          = 3.0
BUFFER_SIZE       = 25
STABILITY         = 10
HOLD_FRAMES       = 15
WARMUP            = 20
MAX_SIGNS         = 8
MOVE_THRESHOLD    = 0.005
MODEL_PATH        = "hand_landmarker.task"

# ── LOAD MODELS (cached) ─────────────────────────────────────────
@st.cache_resource
def load_models():
    static_model  = load_model("models/static_model.keras", compile=False)
    static_scaler = joblib.load("models/static_scaler.pkl")
    static_labels = joblib.load("models/static_config.pkl")['classes']
    try:
        motion_model = load_model("models/motion_model.keras", compile=False)
    except:
        motion_model = load_model("models/motion_model.h5",   compile=False)
    with open("models/motion_labels.pkl","rb") as f:
        motion_labels = pickle.load(f)
    return static_model, static_scaler, static_labels, motion_model, motion_labels

@st.cache_resource
def load_detector():
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return HandLandmarker.create_from_options(options)

@st.cache_resource
def load_groq():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

static_model, static_scaler, static_labels, motion_model, motion_labels = load_models()
detector    = load_detector()
groq_client = load_groq()

# ── HELPERS ──────────────────────────────────────────────────────
def normalize_landmarks(pts_63):
    pts   = np.array(pts_63, dtype=float).reshape(21, 3)
    wrist = pts[0]
    rel   = pts - wrist
    scale = np.linalg.norm(rel[9]) + 1e-6
    return (rel / scale).flatten().tolist()

def extract_landmarks(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    h, w   = frame.shape[:2]
    hand1, hand2 = [0.0]*63, [0.0]*63
    n_hands = 0
    if result.hand_landmarks:
        n_hands = len(result.hand_landmarks)
        for idx, lm in enumerate(result.hand_landmarks[:2]):
            pts = [v for pt in lm for v in (pt.x, pt.y, pt.z)]
            if idx == 0: hand1 = pts
            else:        hand2 = pts
            col = (0,255,0) if idx==0 else (255,150,0)
            for pt in lm:
                cv2.circle(frame,(int(pt.x*w),int(pt.y*h)),5,col,-1)
            for i,j in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                        (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                cv2.line(frame,
                         (int(lm[i].x*w),int(lm[i].y*h)),
                         (int(lm[j].x*w),int(lm[j].y*h)),
                         (0,200,0) if idx==0 else (200,120,0), 2)
    return hand1[:63], hand1+hand2, n_hands, frame

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            os.system(f"afplay {tmp.name} &")
    except Exception as e:
        st.error(f"TTS error: {e}")

def llm_format(words):
    if not words: return ""
    try:
        signs_str = ", ".join(words)
        r = groq_client.chat.completions.create(
            model=os.environ.get("MODEL_NAME","llama-3.3-70b-versatile"),
            messages=[
                {"role":"system","content":
                 f"You are an ASL healthcare assistant. Patient signed: {signs_str}. "
                 f"Form ONE or TWO clear natural sentences. Return ONLY the sentence."},
                {"role":"user","content":f"Signs: {signs_str}"}
            ],
            max_tokens=100, temperature=0.3
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return " ".join(words)

def llm_spell(letters):
    if not letters: return ""
    try:
        ls = "".join(letters).upper()
        r  = groq_client.chat.completions.create(
            model=os.environ.get("MODEL_NAME","llama-3.3-70b-versatile"),
            messages=[
                {"role":"system","content":
                 "Spelling correction: user finger-spelled a word, some letters may be wrong. "
                 "Return ONLY the most likely intended English word."},
                {"role":"user","content":f"Letters: {ls}"}
            ],
            max_tokens=20, temperature=0.1
        )
        return r.choices[0].message.content.strip()
    except:
        return "".join(letters)

# ── UI LAYOUT ────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🤟 SignBridge</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Real-time ASL → Natural Language • Healthcare Communication</p>',
    unsafe_allow_html=True)

col_cam, col_panel = st.columns([3,2], gap="large")

with col_cam:
    cam_ph    = st.empty()
    status_ph = st.empty()

with col_panel:
    st.markdown("#### Detected Sign")
    sign_ph = st.empty()
    sign_ph.markdown(
        '<div class="sign-card">'
        '<div class="sign-display">—</div>'
        '<div class="conf-display">Waiting...</div></div>',
        unsafe_allow_html=True)

    st.markdown("#### Sentence")
    sent_ph = st.empty()
    sent_ph.markdown(
        '<div class="sentence-box">Start signing...</div>',
        unsafe_allow_html=True)

    st.markdown("#### 🤖 AI Output")
    llm_ph = st.empty()
    llm_ph.markdown(
        '<div class="llm-box">Press L to generate...</div>',
        unsafe_allow_html=True)

    st.markdown("####")
    b1, b2, b3, b4 = st.columns(4)
    llm_btn   = b1.button("🤖 LLM",      key="llm")
    back_btn  = b2.button("⌫ Back",      key="back")
    mode_btn  = b3.button("🔄 Mode",      key="mode")
    clear_btn = b4.button("🗑 Clear",     key="clear")

    st.markdown(
        '<p style="text-align:center;color:#444;font-size:0.72rem">'
        'L=LLM • B=back • M=mode • C=clear</p>',
        unsafe_allow_html=True)

# ── BUTTON ACTIONS ───────────────────────────────────────────────
if llm_btn and st.session_state.sentence:
    if st.session_state.spell_mode:
        raw       = "".join(st.session_state.sentence).upper()
        corrected = llm_spell(st.session_state.sentence)
        result    = f"{raw} → {corrected}"
        speak(corrected)
    else:
        result = llm_format(st.session_state.sentence)
        speak(result)
    st.session_state.llm_output = result
    st.session_state.sentence   = []
    st.session_state.last_sign  = None
    llm_ph.markdown(f'<div class="llm-box">{result}</div>',
                    unsafe_allow_html=True)

if back_btn and st.session_state.sentence:
    st.session_state.sentence.pop()
    st.session_state.last_sign = None

if mode_btn:
    st.session_state.spell_mode = not st.session_state.spell_mode
    st.session_state.sentence   = []
    st.session_state.last_sign  = None

if clear_btn:
    st.session_state.sentence   = []
    st.session_state.last_sign  = None
    st.session_state.llm_output = ""
    llm_ph.markdown(
        '<div class="llm-box">Cleared...</div>',
        unsafe_allow_html=True)

# ── CAMERA LOOP ──────────────────────────────────────────────────
# Use session state buffers — persist across reruns
frame_buffer        = st.session_state.frame_buffer
motion_history      = st.session_state.motion_history
static_hold         = st.session_state.static_hold

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    static_lm, motion_lm, n_hands, frame = extract_landmarks(frame)

    chosen_sign = None
    chosen_conf = 0.0
    chosen_type = "—"
    hand_moving = False
    status_text = "🔴 No hands"

    if n_hands > 0:
        st.session_state.hand_visible_frames += 1
        warming_up = st.session_state.hand_visible_frames < WARMUP

        if st.session_state.prev_landmarks is not None:
            diff        = np.abs(np.array(static_lm) -
                                 np.array(st.session_state.prev_landmarks))
            hand_moving = float(np.mean(diff)) > MOVE_THRESHOLD
        st.session_state.prev_landmarks = list(static_lm)

        if warming_up:
            pct = int((st.session_state.hand_visible_frames / WARMUP) * frame.shape[1])
            cv2.rectangle(frame,(0,frame.shape[0]-8),
                          (frame.shape[1],frame.shape[0]),(30,30,30),-1)
            cv2.rectangle(frame,(0,frame.shape[0]-8),
                          (pct,frame.shape[0]),(0,140,255),-1)
            status_text = f"⏳ Stabilizing {WARMUP - st.session_state.hand_visible_frames}..."
        else:
            # Static
            norm_lm     = normalize_landmarks(static_lm)
            scaled_lm   = static_scaler.transform([norm_lm])
            s_probs     = static_model.predict(scaled_lm, verbose=0)[0]
            static_conf = float(np.max(s_probs))
            static_sign = static_labels[int(np.argmax(s_probs))]
            static_hold.append(static_sign)
            if len(static_hold) == HOLD_FRAMES:
                mc = max(set(static_hold), key=list(static_hold).count)
                if list(static_hold).count(mc)/HOLD_FRAMES < 0.80:
                    static_conf = 0.0

            # Motion
            motion_conf, motion_sign = 0.0, None
            frame_buffer.append(motion_lm)
            if hand_moving and len(frame_buffer) == BUFFER_SIZE:
                seq         = np.array(list(frame_buffer),
                                       dtype=np.float32)[np.newaxis,...]
                m_probs     = motion_model.predict(seq, verbose=0)[0]
                motion_conf = float(np.max(m_probs))
                motion_sign = motion_labels[int(np.argmax(m_probs))]
                motion_history.append(motion_sign)
                if len(motion_history) == STABILITY:
                    mc = max(set(motion_history),
                             key=list(motion_history).count)
                    if list(motion_history).count(mc)/STABILITY < 0.75:
                        motion_conf, motion_sign = 0.0, None
            elif not hand_moving:
                motion_conf, motion_sign = 0.0, None
                motion_history.clear()

            # Winner
            if hand_moving and motion_conf >= MOTION_CONF and motion_sign:
                chosen_sign = motion_sign
                chosen_conf = motion_conf
                chosen_type = "MOTION"
            elif static_conf >= STATIC_CONF:
                chosen_sign = static_sign
                chosen_conf = static_conf
                chosen_type = "STATIC"

            # Capture
            now = time.time()
            if (chosen_sign and
                    chosen_sign != st.session_state.last_sign and
                    now - st.session_state.last_sign_time > COOLDOWN and
                    len(st.session_state.sentence) < MAX_SIGNS):
                st.session_state.sentence.append(chosen_sign)
                st.session_state.last_sign      = chosen_sign
                st.session_state.last_sign_time = now

                # Auto LLM
                if len(st.session_state.sentence) >= MAX_SIGNS:
                    result = llm_format(st.session_state.sentence)
                    st.session_state.llm_output = result
                    speak(result)
                    llm_ph.markdown(
                        f'<div class="llm-box">{result}</div>',
                        unsafe_allow_html=True)
                    st.session_state.sentence  = []
                    st.session_state.last_sign = None

            move_str    = "MOVING" if hand_moving else "STILL"
            status_text = f"{'🟢' if hand_moving else '🔵'} {move_str} | {n_hands} hand{'s' if n_hands>1 else ''}"

    else:
        # Hand gone — full reset all persistent buffers
        st.session_state.hand_visible_frames = 0
        st.session_state.prev_landmarks      = None
        st.session_state.motion_history.clear()
        st.session_state.frame_buffer.clear()
        st.session_state.static_hold.clear()
        motion_history.clear()
        frame_buffer.clear()
        static_hold.clear()

    # Update camera
    cam_ph.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                 channels="RGB", use_container_width=True)
    mode_str = "✏️ SPELL" if st.session_state.spell_mode else "🤟 SIGN"
    status_ph.markdown(f"**{status_text}** &nbsp;&nbsp; {mode_str} MODE")

    # Sign card
    if chosen_sign:
        pill = "pill-motion" if chosen_type=="MOTION" else "pill-static"
        sign_ph.markdown(f"""
        <div class="sign-card">
          <div class="sign-display">
            {chosen_sign.upper().replace('_',' ')}
          </div>
          <div class="conf-display">{chosen_conf:.0%} confidence</div>
          <div style="text-align:center;margin-top:0.5rem">
            <span class="status-pill {pill}">{chosen_type}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Sentence chips
    words = st.session_state.sentence
    if words:
        chips = " ".join([
            f'<span style="background:rgba(0,212,255,0.15);'
            f'border:1px solid rgba(0,212,255,0.3);border-radius:8px;'
            f'padding:2px 10px;margin:2px;display:inline-block">'
            f'{w.upper().replace("_"," ")}</span>'
            for w in words
        ])
        sent_ph.markdown(
            f'<div class="sentence-box">{chips}<br>'
            f'<small style="color:#555">'
            f'{len(words)}/{MAX_SIGNS} signs</small></div>',
            unsafe_allow_html=True)
    else:
        sent_ph.markdown(
            '<div class="sentence-box" style="color:#555">'
            'Start signing...</div>',
            unsafe_allow_html=True)

    # LLM output
    if st.session_state.llm_output:
        llm_ph.markdown(
            f'<div class="llm-box">'
            f'{st.session_state.llm_output}</div>',
            unsafe_allow_html=True)

cap.release()