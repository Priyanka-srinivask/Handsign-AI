# SignBridge
 
> **Bridging the communication gap for ASL users and healthcare providers — one sign at a time.**
 
Built in 20 hours at **SharkHack 2026** · Simmons University · April 4–5, 2026
 
---
 
## What it does
 
SignBridge uses a standard webcam to recognise American Sign Language (ASL) in real time and convert it into spoken English — with a specific focus on healthcare communication.
 
1. **MediaPipe** detects 21 hand landmarks per frame via webcam
2. **Static signs** (ASL alphabet A–Z + YES, NO, I LOVE YOU) → MLP Neural Network classifier
3. **Dynamic signs** (HELP, HURT, PLEASE, THANK YOU, EAT, DRINK, WHERE + 20 more) → LSTM sequence model
4. Recognised signs → **LLaMA 3 via Groq** → natural medical sentence
5. Sentence spoken aloud via **gTTS** text-to-speech
 
A patient signs. A doctor hears them speak. No interpreter needed.
 
---
 
## Healthcare Impact
 
> 1 in 5 ASL users report miscommunication with healthcare providers.
 
SignBridge lets patients sign key concepts like `hurt`, `doctor`, `emergency` — and our LLM forms complete, contextually appropriate sentences instantly.
 
---
 
## Key Innovations
 
- First system combining **static + dynamic ASL recognition** in a single real-time pipeline
- **Movement detection layer** routes to the correct model — no interference between static and motion
- **LLaMA 3 via Groq** for ultra-low latency sentence formation from sign keywords
- **Spell correction mode** for fingerspelling names and medical terms
- Custom dataset recorded and trained entirely within 20 hours
 
---
 
## Model Performance
 
| Model | Dataset | Architecture | Test Accuracy |
|---|---|---|---|
| Static classifier | 1,450 samples, 29 classes | MLP: Dense(256)→Dense(128)→Dense(64) | 100% |
| Motion classifier | 33,600 frames, 28 classes | LSTM(128)→LSTM(64)→Dense | 95.54% |
 
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Hand tracking | MediaPipe 0.10.x (HandLandmarker API) |
| Static model | TensorFlow / Keras MLP |
| Motion model | TensorFlow / Keras LSTM |
| Data processing | NumPy, Pandas, Scikit-learn |
| LLM | LLaMA 3 via Groq API |
| Text-to-speech | gTTS |
| UI | Streamlit |
| Computer vision | OpenCV |
 
---
 
## Repository Structure
 
```
Handsign-AI/
├── collect_static.py       # Record static landmark dataset via webcam
├── collect_motion.py       # Record motion gesture dataset via webcam
├── train.py                # Train both static (MLP) and motion (LSTM) models
├── inference.py            # Real-time inference pipeline (webcam → signs → speech)
├── hand_landmarker.task    # MediaPipe hand landmarker model file
├── requirements.txt        # Python dependencies
├── data/                   # Dataset folder (see note below)
└── models/                 # Trained model files
```
 
---
 
## Dataset Note
 
> **The full training dataset is not included in this repository due to file size constraints.**
>
> The dataset consists of:
> - **1,450 static landmark samples** across 29 classes
> - **33,600 motion frames** across 28 gesture classes
>
> To reproduce the dataset, use the provided collection scripts:
> ```bash
> python collect_static.py   # Records static signs (letters)
> python collect_motion.py   # Records dynamic gestures (words/phrases)
> ```
> Each script guides you through recording samples per sign class via your webcam.
 
---
 
## Installation
 
**Requirements:** Python 3.11 (TensorFlow does not support Python 3.12+)
 
```bash
# Clone the repo
git clone https://github.com/Priyanka-srinivask/Handsign-AI.git
cd Handsign-AI
 
# Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
 
# Install dependencies
pip install -r requirements.txt
```
 
---
 
## Usage
 
### Step 1 — Collect your dataset
```bash
python collect_static.py    # Follow prompts to record each letter
python collect_motion.py    # Follow prompts to record each gesture
```
 
### Step 2 — Train the models
```bash
python train.py
```
Trained models are saved to the `/models` folder.
 
### Step 3 — Run real-time inference
```bash
python inference.py
```
 
**Controls while running:**
- `Q` — quit
- `C` — clear the current gloss buffer
 
---
 
## Environment Variables
 
Create a `.env` file in the root directory:
 
```
GROQ_API_KEY=your_groq_api_key_here
```
 
Get a free Groq API key at [console.groq.com](https://console.groq.com)
 
---
 
## Team
 
Built at SharkHack 2026 — Simmons University, Boston MA
 
