import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import pickle
import time
import urllib.request
import os
import joblib
from collections import deque
from gtts import gTTS
import tempfile
from dotenv import load_dotenv
from groq import Groq
from tensorflow.keras.models import load_model

# ── Environment ──────────────────────────────────────────────────
load_dotenv()

# ── Groq LLM ─────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an ASL communication assistant in a healthcare setting.
The user is a deaf or hard-of-hearing patient communicating via sign language.
They have signed these words/concepts: {signs}
Form ONE or TWO clear natural sentences. Return ONLY the sentence. No explanation."""

def llm_format(words):
    """Send signs to Groq LLM → natural sentence."""
    if not words:
        return ""
    try:
        signs_str = ", ".join(words)
        response  = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system",
                 "content": SYSTEM_PROMPT.format(signs=signs_str)},
                {"role": "user",
                 "content": f"Signs: {signs_str}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return " ".join(words)

def llm_spell(letters):
    """Correct misspelled ASL fingerspelling → correct word."""
    if not letters:
        return ""
    try:
        letters_str = "".join(letters).upper()
        response    = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": """You are a spelling correction assistant.
The user spelled a word using ASL finger spelling.
Some letters may be wrong or missing due to recognition errors.
Return ONLY the single most likely intended English word. Nothing else."""},
                {"role": "user",
                 "content": f"Letters signed: {letters_str}"}
            ],
            max_tokens=20,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "".join(letters)

# ── CONFIG ───────────────────────────────────────────────────────
STATIC_CONF_THRESHOLD = 0.60   # min confidence for static sign
MOTION_CONF_THRESHOLD = 0.92   # min confidence for motion sign
SIGN_COOLDOWN         = 3.0    # seconds between captures
FRAMES_FOR_MOTION     = 25     # rolling buffer size for LSTM
MOTION_STABILITY      = 10     # motion must be same for N frames
HOLD_FRAMES           = 15     # static must be held for N frames
MAX_SENTENCE          = 8      # auto-LLM after N signs
WARMUP_FRAMES         = 20     # ignore first N frames on hand appear
MOVEMENT_THRESHOLD    = 0.005  # sensitivity for hand movement
MODEL_PATH            = "hand_landmarker.task"
# ────────────────────────────────────────────────────────────────

# ── Download MediaPipe model if needed ───────────────────────────
if not os.path.exists(MODEL_PATH):
    print("⏬ Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("✅ Downloaded!")

# ── Load ML models ───────────────────────────────────────────────
print("Loading models...")
static_model  = load_model("models/static_model.keras",  compile=False)
static_scaler = joblib.load("models/static_scaler.pkl")
static_labels = joblib.load("models/static_config.pkl")['classes']

try:
    motion_model = load_model("models/motion_model.keras", compile=False)
except:
    motion_model = load_model("models/motion_model.h5",   compile=False)
with open("models/motion_labels.pkl","rb") as f:
    motion_labels = pickle.load(f)

print(f"✅ Static model: {len(static_labels)} signs")
print(f"✅ Motion model: {len(motion_labels)} signs")

# ── MediaPipe ────────────────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

# ── Normalize landmarks ──────────────────────────────────────────
def normalize_landmarks(pts_63):
    """Center on wrist, scale by palm — must match train.py!"""
    pts   = np.array(pts_63, dtype=float).reshape(21, 3)
    wrist = pts[0]
    rel   = pts - wrist
    scale = np.linalg.norm(rel[9]) + 1e-6
    return (rel / scale).flatten().tolist()

# ── Extract landmarks from frame ─────────────────────────────────
def extract_landmarks(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    h, w    = frame.shape[:2]
    hand1   = [0.0] * 63
    hand2   = [0.0] * 63
    n_hands = 0

    if result.hand_landmarks:
        n_hands = len(result.hand_landmarks)
        for idx, lm in enumerate(result.hand_landmarks[:2]):
            pts = [v for pt in lm for v in (pt.x, pt.y, pt.z)]
            if idx == 0: hand1 = pts
            else:        hand2 = pts
            col_dot  = (0,255,0)  if idx==0 else (255,150,0)
            col_line = (0,200,0)  if idx==0 else (200,120,0)
            for pt in lm:
                cv2.circle(frame,(int(pt.x*w),int(pt.y*h)),5,col_dot,-1)
            for i,j in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                        (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                cv2.line(frame,
                         (int(lm[i].x*w),int(lm[i].y*h)),
                         (int(lm[j].x*w),int(lm[j].y*h)),
                         col_line, 2)

    return hand1[:63], hand1+hand2, n_hands

# ── TTS ──────────────────────────────────────────────────────────
def speak(text):
    if not text.strip():
        return
    print(f"🔊 Speaking: {text}")
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            os.system(f"afplay {tmp.name} &")
    except Exception as e:
        print(f"TTS error: {e}")

# ── State ────────────────────────────────────────────────────────
frame_buffer        = deque(maxlen=FRAMES_FOR_MOTION)
motion_history      = deque(maxlen=MOTION_STABILITY)
static_hold         = deque(maxlen=HOLD_FRAMES)
sentence            = []
last_sign           = None
last_sign_time      = 0
prev_landmarks      = None
hand_visible_frames = 0
llm_display         = ""
llm_display_time    = 0
spell_mode          = False

cap = cv2.VideoCapture(0)
print("\n🤙 ASL Real-Time Inference")
print("S=speak | B=backspace | L=LLM | M=toggle spell | C=clear | Q=quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    static_lm, motion_lm, n_hands = extract_landmarks(frame)

    static_sign = None
    static_conf = 0.0
    motion_sign = None
    motion_conf = 0.0
    chosen_sign = None
    chosen_conf = 0.0
    chosen_type = "—"
    hand_moving = False

    if n_hands > 0:
        hand_visible_frames += 1
        warming_up = hand_visible_frames < WARMUP_FRAMES

        # ── Movement detection ─────────────────────────────────
        if prev_landmarks is not None:
            diff        = np.abs(np.array(static_lm) -
                                 np.array(prev_landmarks))
            hand_moving = float(np.mean(diff)) > MOVEMENT_THRESHOLD
        prev_landmarks = list(static_lm)

        if warming_up:
            # Orange warmup progress bar
            pct = int((hand_visible_frames / WARMUP_FRAMES) * 300)
            cv2.rectangle(frame, (0, frame.shape[0]-8),
                          (frame.shape[1], frame.shape[0]), (30,30,30), -1)
            cv2.rectangle(frame, (0, frame.shape[0]-8),
                          (pct, frame.shape[0]), (0,140,255), -1)
            cv2.putText(frame,
                        f"Stabilizing... {WARMUP_FRAMES-hand_visible_frames}",
                        (10,45), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0,140,255), 2)
        else:
            # ── Static prediction ──────────────────────────────
            norm_lm     = normalize_landmarks(static_lm)
            scaled_lm   = static_scaler.transform([norm_lm])
            s_probs     = static_model.predict(scaled_lm, verbose=0)[0]
            static_conf = float(np.max(s_probs))
            static_sign = static_labels[int(np.argmax(s_probs))]

            # Hold buffer — must be consistent for HOLD_FRAMES
            static_hold.append(static_sign)
            if len(static_hold) == HOLD_FRAMES:
                mc = max(set(static_hold), key=list(static_hold).count)
                consistency = list(static_hold).count(mc) / HOLD_FRAMES
                if consistency < 0.80:
                    static_conf = 0.0  # not stable enough

            # ── Motion prediction ──────────────────────────────
            frame_buffer.append(motion_lm)

            if hand_moving and len(frame_buffer) == FRAMES_FOR_MOTION:
                seq         = np.array(list(frame_buffer),
                                       dtype=np.float32)[np.newaxis,...]
                m_probs     = motion_model.predict(seq, verbose=0)[0]
                motion_conf = float(np.max(m_probs))
                motion_sign = motion_labels[int(np.argmax(m_probs))]

                # Stability buffer
                motion_history.append(motion_sign)
                if len(motion_history) == MOTION_STABILITY:
                    mc = max(set(motion_history),
                             key=list(motion_history).count)
                    consistency = list(motion_history).count(mc) / MOTION_STABILITY
                    if consistency < 0.75:
                        motion_conf = 0.0
                        motion_sign = None
            elif not hand_moving:
                motion_conf = 0.0
                motion_sign = None
                motion_history.clear()

            # ── Pick winner ────────────────────────────────────
            if (hand_moving and
                    motion_conf >= MOTION_CONF_THRESHOLD and
                    motion_sign):
                chosen_sign = motion_sign
                chosen_conf = motion_conf
                chosen_type = "MOTION"
            elif static_conf >= STATIC_CONF_THRESHOLD:
                chosen_sign = static_sign
                chosen_conf = static_conf
                chosen_type = "STATIC"

            # ── Auto-capture with cooldown ─────────────────────
            now = time.time()
            if (chosen_sign and
                    chosen_sign != last_sign and
                    now - last_sign_time > SIGN_COOLDOWN and
                    len(sentence) < MAX_SENTENCE):
                sentence.append(chosen_sign)
                last_sign      = chosen_sign
                last_sign_time = now
                print(f"  ✅ [{chosen_type}] {chosen_sign} ({chosen_conf:.0%})")

                # Auto-trigger LLM at max signs
                if len(sentence) >= MAX_SENTENCE:
                    print("  🤖 Auto LLM...")
                    formatted        = llm_format(sentence)
                    llm_display      = formatted
                    llm_display_time = time.time()
                    print(f"  🤖 {formatted}")
                    speak(formatted)
                    sentence  = []
                    last_sign = None

            # ── Sign HUD ───────────────────────────────────────
            sign_txt = chosen_sign.upper().replace("_"," ") \
                       if chosen_sign else "..."
            conf_txt = f"{chosen_conf:.0%}" if chosen_conf > 0 else ""
            cv2.putText(frame, f"{sign_txt}  {conf_txt}",
                        (10,45), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0,255,0) if chosen_sign else (100,100,100), 2)
            cv2.putText(frame, chosen_type,
                        (10,78), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0,255,255) if chosen_type=="MOTION" else (255,255,0), 2)

            # Confidence bars
            if static_conf > 0:
                bw = int(static_conf * 180)
                cv2.rectangle(frame,(10,88),(190,103),(40,40,40),-1)
                cv2.rectangle(frame,(10,88),(10+bw,103),(255,255,0),-1)
                cv2.putText(frame,f"S {static_conf:.0%}",
                            (195,101),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,0),1)
            if motion_conf > 0:
                bw = int(motion_conf * 180)
                cv2.rectangle(frame,(10,108),(190,123),(40,40,40),-1)
                cv2.rectangle(frame,(10,108),(10+bw,123),(0,255,255),-1)
                cv2.putText(frame,f"M {motion_conf:.0%}",
                            (195,121),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,255),1)

    else:
        # Hand gone — full reset all buffers
        hand_visible_frames = 0
        prev_landmarks      = None
        hand_moving         = False
        motion_history.clear()
        frame_buffer.clear()
        static_hold.clear()

    # ── Status row ───────────────────────────────────────────────
    hand_txt = ["No hands","1 hand","Both hands"][min(n_hands,2)]
    hand_col = [(0,60,255),(0,200,255),(0,255,0)][min(n_hands,2)]
    move_txt = " | MOVING" if hand_moving else " | STILL"
    move_col = (0,255,255) if hand_moving else (150,150,150)
    cv2.putText(frame, hand_txt,(10,145),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,hand_col,1)
    cv2.putText(frame, move_txt,(115,145),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,move_col,1)

    # Mode indicator top right
    mode_txt = "SPELL MODE" if spell_mode else "SIGN MODE"
    mode_col = (0,255,255) if spell_mode else (0,255,0)
    cv2.putText(frame, mode_txt,
                (frame.shape[1]-170,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,mode_col,2)

    # Controls
    cv2.putText(frame,"S=speak B=back L=LLM M=mode C=clear Q=quit",
                (10,165),cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,160,160),1)

    # ── LLM output (green, shown 6 seconds) ──────────────────────
    bar_h = frame.shape[0]
    if llm_display and time.time() - llm_display_time < 6:
        cv2.rectangle(frame,(0,bar_h-125),
                      (frame.shape[1],bar_h-58),(0,40,0),-1)
        words = llm_display.split()
        line1 = " ".join(words[:9])
        line2 = " ".join(words[9:])
        cv2.putText(frame, line1,(10,bar_h-98),
                    cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,255,100),2)
        if line2:
            cv2.putText(frame, line2,(10,bar_h-70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,255,100),2)

    # ── Sentence bar ─────────────────────────────────────────────
    cv2.rectangle(frame,(0,bar_h-55),
                  (frame.shape[1],bar_h),(20,20,20),-1)
    disp = " ".join([s.upper().replace("_"," ")
                     for s in sentence[-10:]]) or "..."
    cv2.putText(frame, disp,(10,bar_h-20),
                cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
    cv2.putText(frame,
                f"{'SPELL' if spell_mode else 'SIGN'} "
                f"({len(sentence)}/{MAX_SENTENCE})",
                (10,bar_h-38),cv2.FONT_HERSHEY_SIMPLEX,0.42,(130,130,130),1)

    cv2.imshow("SignBridge — ASL Healthcare", frame)
    key = cv2.waitKey(1) & 0xFF

    # ── Key controls ─────────────────────────────────────────────
    if key == ord('s') and sentence:
        speak(" ".join(sentence))

    elif key == ord('m'):
        spell_mode = not spell_mode
        sentence   = []
        last_sign  = None
        print(f"  🔄 {'SPELL' if spell_mode else 'SIGN'} MODE")

    elif key == ord('b') and sentence:
        removed   = sentence.pop()
        last_sign = None
        print(f"  ⌫ Removed [{removed}] | Now: {sentence}")

    elif key == ord('l') and sentence:
        if spell_mode:
            raw      = "".join(sentence).upper()
            corrected= llm_spell(sentence)
            print(f"📝 Letters: {raw} → {corrected}")
            speak(corrected)
            llm_display      = f"{raw} → {corrected}"
            llm_display_time = time.time()
        else:
            formatted        = llm_format(sentence)
            llm_display      = formatted
            llm_display_time = time.time()
            print(f"🤖 {formatted}")
            speak(formatted)
        sentence  = []
        last_sign = None

    elif key == ord('c'):
        sentence  = []
        last_sign = None
        print("🗑 Cleared")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("👋 Goodbye!")