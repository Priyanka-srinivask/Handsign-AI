import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import csv
import os
import urllib.request
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────
SIGNS = [
    # Already captured — auto-skip
    'j', 'z',
    'yes', 'no', 'please', 'thank_you', 'sorry', 'help',
    'more', 'all_done', 'find', 'want', 'open', 'sit',
    'sleep', 'play', 'eat', 'drink', 'hurt', 'restroom',
    'stop', 'go', 'what', 'where', 'when', 'who', 'why',
    'how', 'my_name_is',
    'hello', 'goodbye', 'good', 'bad',
    'happy', 'sad', 'angry', 'tired', 'hungry', 'water',
    'medicine', 'doctor', 'emergency',
    'understand', 'repeat', 'slow_down',
    # New signs to capture
    'nice_to_meet_you',
    'i_dont_understand',
    'i_love_you_motion',
    'how_are_you',
    'no_problem',
    'feeling',
]

SAMPLES_PER_SIGN  = 60
FRAMES_PER_SAMPLE = 30
OUTPUT_CSV        = "data/motion_landmarks.csv"
MODEL_PATH        = "hand_landmarker.task"
# ────────────────────────────────────────────────────────────────

# ── ASL reference hints shown on screen ─────────────────────────
HINTS = {
    # Already captured
    'j'               : "Trace letter J in air with pinky",
    'z'               : "Trace letter Z in air with index finger",
    'yes'             : "Closed fist — nod up and down",
    'no'              : "Index+middle fingers tap thumb side to side",
    'please'          : "Flat hand circles on chest",
    'thank_you'       : "Flat hand moves from chin outward",
    'sorry'           : "Fist circles on chest",
    'help'            : "Fist on flat palm, lift upward",
    'more'            : "Fingertips tap together repeatedly",
    'all_done'        : "Both hands flip outward twice",
    'find'            : "Fingers pinch downward",
    'want'            : "Both hands pull toward body",
    'open'            : "Both hands spread apart",
    'sit'             : "Two fingers tap down on other hand",
    'sleep'           : "Hand closes over face downward",
    'play'            : "Y-hands shake side to side",
    'eat'             : "Hand taps mouth repeatedly",
    'drink'           : "Thumb tips to mouth, tilts back",
    'hurt'            : "Index fingers circle toward each other",
    'restroom'        : "R-hand shakes side to side",
    'stop'            : "Flat hand chops down on other palm",
    'go'              : "Both index fingers curl and point forward",
    'what'            : "Fingers wiggle side to side",
    'where'           : "Index finger wags side to side",
    'when'            : "Index fingers circle then touch",
    'who'             : "Index finger circles lips",
    'why'             : "Hand at forehead, fingers spread then Y",
    'how'             : "Both fists roll forward and open",
    'my_name_is'      : "Touch chest, then H-hand taps twice",
    # New signs
    'hello'           : "Open hand waves side to side from forehead",
    'goodbye'         : "Open hand waves outward repeatedly",
    'good'            : "Flat hand moves from chin to other palm",
    'bad'             : "Hand at mouth then flips down sharply",
    'happy'           : "Both hands brush upward on chest twice",
    'sad'             : "Both hands slide down face slowly",
    'angry'           : "Claw hand at face, pull outward with tension",
    'tired'           : "Both bent hands drop down from shoulders",
    'hungry'          : "C-hand circles down chest toward stomach",
    'water'           : "W-hand taps chin twice",
    'bathroom'        : "T-hand shakes side to side",
    'pain'            : "Both index fingers circle toward each other",
    'medicine'        : "Middle finger taps palm in circular motion",
    'doctor'          : "D-hand taps wrist pulse point",
    'emergency'       : "Both E-hands shake urgently side to side",
    'understand'      : "Index finger at temple then flicks open",
    'repeat'          : "Dominant hand circles over other palm",
    'slow_down'       : "Both hands press downward slowly",
    'nice_to_meet_you': "Flat hand from chest moves to other palm",
    'i_dont_understand': "Index finger twists at temple",
    'i_love_you_motion': "ILY handshape — pinky, index, thumb out — shake gently",
    'how_are_you'     : "Both bent hands roll forward",
    'no_problem'      : "Flat hands wave downward twice — relaxed",
    'feeling'         : "Middle finger brushes up chest slowly",
    'me'              : "Index finger points to your own chest",
    'you'             : "Index finger points outward toward other person",
    'sorry'           : "Fist circles on chest — already in HINTS above",
}

os.makedirs("data", exist_ok=True)

# ── Download model if not present ───────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("⏬ Downloading hand landmarker model (~8MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("✅ Model downloaded!")

# ── Setup HandLandmarker — 2 hands ───────────────────────────────
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,                          # ← detect both hands
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

# ── CSV header: hand1 (63) + hand2 (63) + meta ──────────────────
# 126 landmark values total (2 hands × 21 points × 3 coords)
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        h1 = [f"h1_{ax}{i}" for i in range(21) for ax in ['x','y','z']]
        h2 = [f"h2_{ax}{i}" for i in range(21) for ax in ['x','y','z']]
        csv.writer(f).writerow(h1 + h2 + ['label', 'sequence_id', 'frame_num'])

# ── Helpers ──────────────────────────────────────────────────────
def count_existing(sign):
    if not os.path.exists(OUTPUT_CSV):
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    if 'label' not in df.columns:
        return 0
    seqs = df[df['label'] == sign]['sequence_id'].nunique()
    return seqs

def delete_last_sequence(sign):
    if not os.path.exists(OUTPUT_CSV):
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    sign_df = df[df['label'] == sign]
    if sign_df.empty:
        print(f"  ⚠️  No sequences for [{sign.upper()}]")
        return 0
    last_seq = sign_df['sequence_id'].max()
    df = df[~((df['label'] == sign) & (df['sequence_id'] == last_seq))]
    df.to_csv(OUTPUT_CSV, index=False)
    remaining = df[df['label'] == sign]['sequence_id'].nunique()
    print(f"  🗑  Deleted last sequence of [{sign.upper()}] — {remaining} remaining")
    return remaining

def delete_all_sequences(sign):
    if not os.path.exists(OUTPUT_CSV):
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    before = df[df['label'] == sign]['sequence_id'].nunique()
    df = df[df['label'] != sign]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  🗑  Deleted all {before} sequences of [{sign.upper()}]")
    return 0

def get_next_seq_id():
    if not os.path.exists(OUTPUT_CSV):
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    if df.empty or 'sequence_id' not in df.columns:
        return 0
    return int(df['sequence_id'].max()) + 1

# ── Extract landmarks from both hands ───────────────────────────
def extract_landmarks(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    h, w = frame.shape[:2]
    hand1 = [0.0] * 63   # zeros if hand not detected
    hand2 = [0.0] * 63

    if result.hand_landmarks:
        for idx, lm in enumerate(result.hand_landmarks[:2]):  # max 2 hands
            pts = [v for pt in lm for v in (pt.x, pt.y, pt.z)]
            if idx == 0:
                hand1 = pts
            else:
                hand2 = pts
            # Draw
            for pt in lm:
                color = (0,255,0) if idx == 0 else (255,150,0)
                cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 4, color, -1)
            for i,j in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                        (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                x1,y1 = int(lm[i].x*w), int(lm[i].y*h)
                x2,y2 = int(lm[j].x*w), int(lm[j].y*h)
                color = (0,200,0) if idx == 0 else (200,120,0)
                cv2.line(frame, (x1,y1), (x2,y2), color, 1)

    hands_detected = len(result.hand_landmarks) if result.hand_landmarks else 0
    return hand1 + hand2, hands_detected   # 126 values total

# ── Main loop ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
seq_id = get_next_seq_id()

print("\n🤙 ASL Motion Sign Collector (2 hands)")
print("Controls:")
print("  SPACE = start recording sequence")
print("  D     = delete last sequence")
print("  X     = delete ALL sequences for current sign")
print("  N     = skip to next sign")
print("  Q     = quit\n")

for sign in SIGNS:

    already = count_existing(sign)
    if already >= SAMPLES_PER_SIGN:
        print(f"  ✅ [{sign.upper()}] complete ({already} sequences) — skipping")
        continue

    count = already
    hint  = HINTS.get(sign, "Perform the sign naturally")

    if already > 0:
        print(f"👉 Resuming [{sign.upper()}] — {already} done, need {SAMPLES_PER_SIGN - already} more")
    else:
        print(f"👉 Sign: [{sign.upper()}]  —  need {SAMPLES_PER_SIGN} sequences")
    print(f"   💡 How: {hint}")

    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        landmarks, hands_detected = extract_landmarks(frame)

        # ── HUD ─────────────────────────────────────────────────
        cv2.rectangle(frame, (0,0), (680, 185), (0,0,0), -1)

        cv2.putText(frame, f"Sign: {sign.upper()}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)
        cv2.putText(frame, f"Seq: {count}/{SAMPLES_PER_SIGN}", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,0), 2)

        # Progress bar
        prog = int((count / SAMPLES_PER_SIGN) * 400)
        cv2.rectangle(frame, (10,78), (410,95), (50,50,50), -1)
        cv2.rectangle(frame, (10,78), (10+prog,95), (0,200,0), -1)

        # Hint text (truncate if too long)
        hint_display = hint if len(hint) < 55 else hint[:52] + "..."
        cv2.putText(frame, f"Hint: {hint_display}", (10, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,255,255), 1)

        cv2.putText(frame, "SPACE=record  D=del last  X=del all  N=next  Q=quit",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (200,200,200), 1)

        # Hand detection status
        if hands_detected == 2:
            cv2.putText(frame, "Both hands detected", (10, 163),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        elif hands_detected == 1:
            cv2.putText(frame, "1 hand detected", (10, 163),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
        else:
            cv2.putText(frame, "No hands detected!", (10, 163),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,60,255), 1)

        cv2.imshow("ASL Collector - Motion", frame)
        key = cv2.waitKey(30) & 0xFF

        # ── SPACE: record sequence ───────────────────────────────
        if key == 32:
            # Countdown
            for c in [3, 2, 1]:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                extract_landmarks(frame)
                cv2.rectangle(frame, (0,0), (680,185), (0,0,0), -1)
                cv2.putText(frame, f"Get ready: {c}", (180, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 8)
                cv2.putText(frame, f"Sign: {sign.upper()}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)
                cv2.imshow("ASL Collector - Motion", frame)
                cv2.waitKey(1000)

            # Record 30 frames
            frames_data = []
            for f_idx in range(FRAMES_PER_SAMPLE):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                lm, hd = extract_landmarks(frame)
                frames_data.append(lm)

                # Progress bar while recording
                prog_r = int((f_idx / FRAMES_PER_SAMPLE) * 500)
                cv2.rectangle(frame, (0,0), (680,185), (0,0,0), -1)
                cv2.putText(frame, f"RECORDING... {f_idx+1}/{FRAMES_PER_SAMPLE}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)
                cv2.putText(frame, f"Sign: {sign.upper()}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame, (10,110), (510,135), (50,50,50), -1)
                cv2.rectangle(frame, (10,110), (10+prog_r,135), (0,0,255), -1)
                hand_txt = f"{hd} hand(s) detected"
                cv2.putText(frame, hand_txt, (10, 163),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0,255,0) if hd > 0 else (0,60,255), 1)
                cv2.imshow("ASL Collector - Motion", frame)
                cv2.waitKey(1)

            # Save to CSV
            with open(OUTPUT_CSV, 'a', newline='') as f:
                w = csv.writer(f)
                for f_idx, lm in enumerate(frames_data):
                    w.writerow(lm + [sign, seq_id, f_idx])
            seq_id += 1
            count  += 1
            print(f"  ✅ [{sign.upper()}] — sequence {count}/{SAMPLES_PER_SIGN} saved")

        # ── D: delete last sequence ──────────────────────────────
        elif key == ord('d'):
            count = delete_last_sequence(sign)

        # ── X: delete all sequences for this sign ───────────────
        elif key == ord('x'):
            count = delete_all_sequences(sign)

        # ── N: skip ─────────────────────────────────────────────
        elif key == ord('n'):
            print(f"  ⏭  Skipped [{sign.upper()}]")
            break

        # ── Q: quit ─────────────────────────────────────────────
        elif key == ord('q'):
            print("\n👋 Quitting. Progress saved!")
            cap.release()
            cv2.destroyAllWindows()
            detector.close()
            exit()

    if count >= SAMPLES_PER_SIGN:
        print(f"  ✔  [{sign.upper()}] complete!\n")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\n🎉 Motion data collection complete! Saved to:", OUTPUT_CSV)