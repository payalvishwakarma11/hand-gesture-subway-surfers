# 🎮 Hand Gesture Subway Surfers Controller
### Control the game with your hand — no keyboard needed!
A real-time AI-based system that lets you control Subway Surfers using hand gestures — no keyboard needed!

Highlights
Gesture-based game control
Real-time hand tracking using MediaPipe
Smooth and responsive gameplay
Combines Computer Vision + Automation
---

## 📁 Folder Structure

```
hand_gesture_subway_surfers/
│
├── hand_detector.py          ← STEP 1: MediaPipe hand detection
├── collect_data.py           ← STEP 2: Record gesture data
├── train_model.py            ← STEP 3: Train neural network
├── real_time_predictor.py    ← STEP 4: Run the game controller
├── gui_launcher.py           ← BONUS: Tkinter GUI launcher
│
├── utils/
│   └── gesture_smoother.py  ← Smooths jumpy predictions
│
├── data/
│   └── gesture_dataset.csv  ← Created by collect_data.py
│
├── models/
│   ├── gesture_model.h5     ← Trained Keras model
│   ├── scaler.pkl           ← StandardScaler
│   ├── label_encoder.pkl    ← LabelEncoder
│   ├── confusion_matrix.png ← Evaluation results
│   └── training_history.png ← Loss/accuracy curves
│
├── logs/                    ← Optional logging directory
└── requirements.txt         ← Python dependencies
```

---

## 🚀 Setup — Step by Step

### 1. Install Python 3.10
Download from: https://www.python.org/downloads/

### 2. Create a Virtual Environment (recommended)
```bash
python -m venv gesture_env

# Activate it:
# Windows:
gesture_env\Scripts\activate
# Mac/Linux:
source gesture_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install opencv-python mediapipe tensorflow numpy pandas scikit-learn pyautogui matplotlib seaborn
```

### 4. Test Your Webcam
Make sure your webcam is connected and working.

---

## 📋 How to Run (4 Steps)

### Step 1 — Test Hand Detection
```bash
python hand_detector.py
```
✅ You should see your webcam feed with a hand skeleton drawn on it.
Press Q to quit.

---

### Step 2 — Collect Training Data
```bash
python collect_data.py
```

**In the window:**
- Hold your left hand pointing LEFT → press and hold **L**
- Hold your right hand pointing RIGHT → press and hold **R**
- Raise your hand UP (like jumping) → press and hold **U**
- Push your hand DOWN → press and hold **D**

**Tips:**
- Aim for 300+ samples per gesture
- Vary your hand position and angle slightly
- Good lighting helps a lot
- Press Q when done

---

### Step 3 — Train the Model
```bash
python train_model.py
```

This will:
- Load your CSV dataset
- Train a Neural Network for up to 100 epochs
- Show accuracy and confusion matrix
- Save the model to models/

**Expected accuracy: 90-99%** (depends on data quality)

---

### Step 4 — Play the Game!
1. Open Subway Surfers in your browser (poki.com/en/g/subway-surfers or an Android emulator)
2. Click on the game window to give it keyboard focus
3. Run the controller:
```bash
python real_time_predictor.py
```
4. Use your hand gestures to play!

---

## 🎯 Gesture Reference

| Gesture | Hand Position | Game Action | Key |
|---------|---------------|-------------|-----|
| **Left** | Point/lean hand left | Move Left | ← |
| **Right** | Point/lean hand right | Move Right | → |
| **Jump** | Raise hand up | Jump | ↑ |
| **Down** | Push hand down | Slide/Roll | ↓ |

---

## 🖥️ GUI Launcher (Bonus)
```bash
python gui_launcher.py
```
A dark-themed control panel with one-click buttons for each step.

---

## 🔧 Troubleshooting

### "Cannot open webcam"
- Check if another app is using your webcam
- Try changing `VideoCapture(0)` to `VideoCapture(1)` or `VideoCapture(2)`

### "ModuleNotFoundError"
```bash
pip install <missing-module>
```

### Low accuracy (< 85%)
1. Collect more samples (500+ per gesture)
2. Ensure gestures are visually distinct
3. Use consistent lighting
4. Don't collect data when tired (shaky hands)

### Game not responding to gestures
- Make sure the game window has focus (click on it)
- Check the webcam window shows "→ GESTURE [key]" labels
- Lower the CONFIDENCE_THRESHOLD in real_time_predictor.py

### PyAutoGUI safety stop
- Moving mouse to any screen corner stops the script (safety feature)
- This is intentional — to stop the script quickly

---

## 🎓 How the ML Pipeline Works

```
Webcam Frame
    ↓
MediaPipe → 21 hand landmarks (x, y coordinates)
    ↓
Normalize relative to wrist (position-independent)
    ↓
StandardScaler (same scale as training)
    ↓
Neural Network → [0.02, 0.91, 0.05, 0.02]
    ↓
Argmax → "Jump" (91% confident)
    ↓
GestureSmoother → Confirm after 8 consistent frames
    ↓
PyAutoGUI → Press ↑ arrow key
    ↓
Subway Surfers → Character jumps!
```

---

## 💡 Improving Accuracy

1. **More data**: 500+ samples per gesture > 300
2. **Clean data**: Delete mislabeled rows from CSV
3. **Augmentation**: Flip left/right hand data
4. **Better gestures**: Make gestures more visually distinct
5. **Better lighting**: Avoid backlighting
6. **Tune smoother**: Adjust `SMOOTHER_WINDOW` and `COOLDOWN_SECONDS`
7. **Model tuning**: Add more layers, increase epochs

---

Made with using MediaPipe + TensorFlow + OpenCV
---

## 👩‍💻 About Me
Hi, I'm **Payal Vishwakarma**, an AIML student passionate about building real-world AI projects.
⭐ If you like this project
Give it a ⭐ on GitHub!

