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
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    'yes','no','stop','i_love_you'
]
SAMPLES_PER_SIGN = 50
OUTPUT_CSV       = "data/static_landmarks.csv"
MODEL_PATH       = "hand_landmarker.task"
# ────────────────────────────────────────────────────────────────

os.makedirs("data", exist_ok=True)

# ── Download model if not present ───────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("⏬ Downloading hand landmarker model (~8MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("✅ Model downloaded!")

# ── Setup HandLandmarker ─────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

# ── CSV header ───────────────────────────────────────────────────
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        cols = [f"{ax}{i}" for i in range(21) for ax in ['x','y','z']]
        csv.writer(f).writerow(cols + ['label'])

# ── Helper: count existing samples for a sign ───────────────────
def count_existing(sign):
    if not os.path.exists(OUTPUT_CSV):
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    return len(df[df['label'] == sign])

# ── Helper: delete ALL samples for a sign ───────────────────────
def delete_sign(sign):
    if not os.path.exists(OUTPUT_CSV):
        print(f"  ⚠️  No CSV found.")
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    before = len(df[df['label'] == sign])
    df = df[df['label'] != sign]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  🗑  Deleted all {before} samples of [{sign.upper()}]")
    return 0  # reset count to 0

# ── Helper: delete LAST sample for a sign ───────────────────────
def delete_last_sample(sign):
    if not os.path.exists(OUTPUT_CSV):
        return 0
    df = pd.read_csv(OUTPUT_CSV)
    sign_rows = df[df['label'] == sign]
    if len(sign_rows) == 0:
        print(f"  ⚠️  No samples of [{sign.upper()}] to delete.")
        return 0
    # Drop the last row of this sign
    last_idx = sign_rows.index[-1]
    df = df.drop(index=last_idx)
    df.to_csv(OUTPUT_CSV, index=False)
    remaining = len(df[df['label'] == sign])
    print(f"  🗑  Deleted last sample of [{sign.upper()}] — {remaining} remaining")
    return remaining

# ── Extract landmarks ────────────────────────────────────────────
def extract_landmarks(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        for pt in lm:
            cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 5, (0,255,0), -1)
        for i,j in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                    (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
            x1,y1 = int(lm[i].x*w), int(lm[i].y*h)
            x2,y2 = int(lm[j].x*w), int(lm[j].y*h)
            cv2.line(frame, (x1,y1), (x2,y2), (0,200,0), 1)
        return [v for pt in lm for v in (pt.x, pt.y, pt.z)]
    return None

cap = cv2.VideoCapture(0)

print("\n🤙 ASL Static Sign Collector")
print("Controls:")
print("  SPACE = capture sample")
print("  D     = delete last sample")
print("  X     = delete ALL samples for current sign")
print("  N     = skip to next sign")
print("  Q     = quit\n")

for sign in SIGNS:

    # ── Resume: skip if already complete ────────────────────────
    already = count_existing(sign)
    if already >= SAMPLES_PER_SIGN:
        print(f"  ✅ [{sign.upper()}] complete ({already} samples) — skipping")
        continue

    count = already
    if already > 0:
        print(f"👉 Resuming [{sign.upper()}] — {already} done, need {SAMPLES_PER_SIGN - already} more")
    else:
        print(f"👉 Get ready for: [{sign.upper()}]  —  need {SAMPLES_PER_SIGN} samples")

    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        landmarks = extract_landmarks(frame)

        # ── HUD ─────────────────────────────────────────────────
        # Background bar for readability
        cv2.rectangle(frame, (0,0), (640, 170), (0,0,0), -1)

        cv2.putText(frame, f"Sign: {sign.upper()}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.putText(frame, f"Captured: {count}/{SAMPLES_PER_SIGN}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        # Progress bar
        prog = int((count / SAMPLES_PER_SIGN) * 400)
        cv2.rectangle(frame, (10,85), (410,105), (50,50,50), -1)
        cv2.rectangle(frame, (10,85), (10+prog,105), (0,200,0), -1)

        cv2.putText(frame, "SPACE=capture  D=del last  X=del all  N=next  Q=quit",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if landmarks:
            cv2.putText(frame, "Hand detected", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        else:
            cv2.putText(frame, "No hand — show your hand!", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,80,255), 1)

        cv2.imshow("ASL Collector", frame)
        key = cv2.waitKey(30) & 0xFF

        # ── SPACE: capture ───────────────────────────────────────
        if key == 32:
            if landmarks:
                with open(OUTPUT_CSV, 'a', newline='') as f:
                    csv.writer(f).writerow(landmarks + [sign])
                count += 1
                print(f"  ✅ [{sign.upper()}] — {count}/{SAMPLES_PER_SIGN}")
            else:
                print("  ⚠️  No hand detected!")

        # ── D: delete last sample ────────────────────────────────
        elif key == ord('d'):
            count = delete_last_sample(sign)

        # ── X: delete ALL samples for this sign ─────────────────
        elif key == ord('x'):
            count = delete_sign(sign)

        # ── N: skip to next sign ─────────────────────────────────
        elif key == ord('n'):
            print(f"  ⏭  Skipped [{sign.upper()}]")
            break

        # ── Q: quit ──────────────────────────────────────────────
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
print("\n🎉 All signs collected! Saved to:", OUTPUT_CSV)