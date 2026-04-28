"""
=============================================================
 STEP 2: collect_data.py
 Purpose: Collect training data by recording your gestures
=============================================================

HOW TO USE:
  1. Run this script: python collect_data.py
  2. Press the key shown on screen for each gesture:
     [L] = Left   [R] = Right   [U] = Jump (Up)   [D] = Down
  3. Hold your gesture STEADY and press/hold the key
  4. Collect ~200 samples per gesture
  5. Press [Q] to quit and save

WHAT IT SAVES:
  Each sample = 42 numbers (21 landmarks × 2 coordinates)
  + 1 label (gesture name)
  Saved to: data/gesture_dataset.csv
"""

import cv2
import csv
import os
import time
import numpy as np
from hand_detector import HandDetector

# ─── CONFIGURATION ───────────────────────────
DATA_DIR    = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "gesture_dataset.csv")
SAMPLES_PER_GESTURE = 300   # Target samples for each gesture
COLLECTION_DELAY    = 0.05  # Seconds between samples (avoid duplicates)

# Gesture mapping: key pressed → class label
GESTURE_MAP = {
    'l': 'Left',
    'r': 'Right',
    'u': 'Jump',
    'd': 'Down',
}
# ─────────────────────────────────────────────


def count_existing_samples():
    """Count how many samples already exist in the dataset."""
    counts = {gesture: 0 for gesture in GESTURE_MAP.values()}
    
    if not os.path.exists(OUTPUT_FILE):
        return counts
    
    with open(OUTPUT_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row:
                label = row[-1]  # Last column is label
                if label in counts:
                    counts[label] += 1
    
    return counts


def draw_ui(frame, active_gesture, sample_counts, collecting):
    """
    Draw the data collection UI overlay on the frame.
    
    Shows:
    - Instructions
    - Current gesture being recorded
    - Sample counts per gesture
    - Recording indicator
    """
    h, w = frame.shape[:2]
    
    # Semi-transparent black background for text area
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    
    # Title
    cv2.putText(frame, "DATA COLLECTOR", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)
    
    # Keyboard instructions
    instructions = [
        "Keys:",
        " [L] = Left (hand pointing left)",
        " [R] = Right (hand pointing right)",
        " [U] = Jump (hand pointing up)",
        " [D] = Down (hand pointing down)",
        " [Q] = Quit & Save",
    ]
    
    y = 65
    for line in instructions:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
    
    # Sample counts for each gesture
    y += 10
    cv2.putText(frame, "Samples collected:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 25
    
    gesture_colors = {
        'Left':  (255, 100, 50),
        'Right': (50, 150, 255),
        'Jump':  (50, 255, 150),
        'Down':  (50, 50, 255),
    }
    
    for gesture in ['Left', 'Right', 'Jump', 'Down']:
        count = sample_counts.get(gesture, 0)
        color = gesture_colors.get(gesture, (200, 200, 200))
        
        # Progress bar
        bar_len = int((count / SAMPLES_PER_GESTURE) * 150)
        bar_len = min(bar_len, 150)  # Cap at full
        cv2.rectangle(frame, (10, y - 12), (10 + bar_len, y - 2), color, -1)
        cv2.rectangle(frame, (10, y - 12), (160, y - 2), (100, 100, 100), 1)
        
        status = "✓ DONE" if count >= SAMPLES_PER_GESTURE else f"{count}/{SAMPLES_PER_GESTURE}"
        cv2.putText(frame, f"{gesture}: {status}", (170, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 30
    
    # Active gesture indicator
    if active_gesture:
        color = gesture_colors.get(active_gesture, (255, 255, 255))
        indicator = "● RECORDING" if collecting else "○ Hold key to record"
        cv2.putText(frame, f"Gesture: {active_gesture}", (10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, indicator, (10, y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if collecting else (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Press a key to start recording", (10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    
    return frame


def collect_data():
    """
    Main data collection loop.
    Opens webcam, detects hands, and saves gesture data to CSV.
    """
    print("=" * 60)
    print("  GESTURE DATA COLLECTOR")
    print("=" * 60)
    print(f"Saving data to: {OUTPUT_FILE}")
    print(f"Target: {SAMPLES_PER_GESTURE} samples per gesture")
    print()
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize hand detector
    detector = HandDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Load existing sample counts
    sample_counts = count_existing_samples()
    print("Existing samples:", sample_counts)
    
    # Open CSV file for appending
    file_exists = os.path.exists(OUTPUT_FILE)
    csv_file = open(OUTPUT_FILE, 'a', newline='')
    writer = csv.writer(csv_file)
    
    # Write header if file is new
    if not file_exists:
        header = [f"x{i}" if i % 2 == 0 else f"y{i//2}" for i in range(42)]
        header[-1] = "x20"
        # Actually create proper header
        header = []
        for i in range(21):
            header.append(f"lm{i}_x")
            header.append(f"lm{i}_y")
        header.append("label")
        writer.writerow(header)
    
    # State tracking
    active_gesture  = None
    last_save_time  = 0
    total_collected = 0
    
    print("\nWebcam opened! Press gesture keys to collect data.")
    print("Press Q to quit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Lost webcam connection!")
            break
        
        # Mirror the frame for intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Detect hand landmarks
        landmarks, annotated = detector.get_landmarks(frame)
        
        # Check keyboard input (non-blocking, 1ms wait)
        key = cv2.waitKey(1) & 0xFF
        
        # Map key to gesture
        if key == ord('q') or key == ord('Q'):
            print("\nQ pressed — saving and quitting...")
            break
        elif chr(key).lower() in GESTURE_MAP:
            active_gesture = GESTURE_MAP[chr(key).lower()]
        
        # Collect sample if hand is detected and key is held
        current_time = time.time()
        collecting = False
        
        if (active_gesture and landmarks and 
            current_time - last_save_time >= COLLECTION_DELAY):
            
            # Save this sample
            row = landmarks + [active_gesture]
            writer.writerow(row)
            
            sample_counts[active_gesture] = sample_counts.get(active_gesture, 0) + 1
            last_save_time = current_time
            total_collected += 1
            collecting = True
            
            # Console progress
            if total_collected % 50 == 0:
                print(f"Collected {total_collected} samples | Counts: {sample_counts}")
        
        # Draw UI overlay
        annotated = draw_ui(annotated, active_gesture, sample_counts, collecting)
        
        # Show the frame
        cv2.imshow("Gesture Data Collector", annotated)
    
    # Cleanup
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("  DATA COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"Total new samples: {total_collected}")
    print(f"Final counts: {sample_counts}")
    print(f"Data saved to: {OUTPUT_FILE}")
    print("\nNext step: Run train_model.py")


if __name__ == "__main__":
    collect_data()
