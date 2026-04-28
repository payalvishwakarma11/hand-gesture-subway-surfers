import cv2
import mediapipe as mp
import pyautogui
import time

# Cooldown setup
last_action_time = 0
cooldown = 0.35   # थोड़ा बढ़ाया smooth के लिए

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Start controlling game... Press Q to quit")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    h, w, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Finger tip position
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

            current_time = time.time()

            # 👉 COOL DOWN LOGIC (IMPORTANT)
            if current_time - last_action_time > cooldown:

                if x < w // 3:
                    pyautogui.press("left")
                    cv2.putText(img, "LEFT", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    last_action_time = current_time

                elif x > 2 * w // 3:
                    pyautogui.press("right")
                    cv2.putText(img, "RIGHT", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_action_time = current_time

                elif y < h // 3:
                    pyautogui.press("up")
                    cv2.putText(img, "JUMP", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    last_action_time = current_time

                elif y > 2 * h // 3:
                    pyautogui.press("down")
                    cv2.putText(img, "DOWN", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    last_action_time = current_time

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Game Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()