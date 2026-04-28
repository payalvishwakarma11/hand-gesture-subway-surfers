import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui

# Load files
model = pickle.load(open("model/model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting... Press Q to quit")

last_gesture = ""
count = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            row = []
            for lm in handLms.landmark:
                row.extend([lm.x, lm.y])

            # Apply scaling
            row = scaler.transform([row])

            prediction = model.predict(row)
            gesture = encoder.inverse_transform(prediction)[0]

            # Stability logic
            if gesture == last_gesture:
                count += 1
            else:
                count = 0

            if count > 5:
                if gesture == "Jump":
                    pyautogui.press("up")
                elif gesture == "Left":
                    pyautogui.press("left")
                elif gesture == "Right":
                    pyautogui.press("right")
                elif gesture == "Down":
                    pyautogui.press("down")

            last_gesture = gesture

            cv2.putText(img, gesture, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Prediction", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break