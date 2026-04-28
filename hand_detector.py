"""
=============================================================
 STEP 1: hand_detector.py
 Purpose: Detect hand landmarks using MediaPipe
 This is the FOUNDATION of the entire project.
=============================================================

HOW IT WORKS:
  MediaPipe detects 21 keypoints (landmarks) on your hand.
  Each keypoint has an x, y, z coordinate.
  We normalize these coordinates relative to the wrist
  so our model works regardless of where the hand is in the frame.

  Hand Landmark Map:
  0 = Wrist
  4 = Thumb tip     8 = Index tip
  12 = Middle tip   16 = Ring tip    20 = Pinky tip
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Detects hand landmarks using Google's MediaPipe library.
    
    Usage:
        detector = HandDetector()
        landmarks = detector.get_landmarks(frame)
    """
    
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initialize the hand detector.
        
        Args:
            max_hands (int): Maximum number of hands to detect (we use 1 for simplicity)
            detection_confidence (float): Minimum confidence for hand detection (0.0 to 1.0)
            tracking_confidence (float): Minimum confidence for tracking (0.0 to 1.0)
        """
        # Initialize MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create the hand detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        # False = video stream (faster tracking)
            max_num_hands=max_hands,        # Detect only 1 hand
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        print(f"[HandDetector] Initialized with max_hands={max_hands}")
    
    def get_landmarks(self, frame):
        """
        Extract 21 hand landmarks from a webcam frame.
        
        Args:
            frame: BGR image from OpenCV (numpy array)
        
        Returns:
            landmarks (list): 42 values [x0,y0, x1,y1, ... x20,y20]
                              normalized relative to wrist position
                              Returns None if no hand detected.
            annotated_frame: Frame with hand skeleton drawn on it
        """
        # MediaPipe needs RGB images, but OpenCV gives us BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = self.hands.process(frame_rgb)
        
        annotated_frame = frame.copy()
        
        # Check if any hand was detected
        if results.multi_hand_landmarks:
            # Take the FIRST detected hand only
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw the hand skeleton on the frame (visual feedback)
            self.mp_draw.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract raw landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                # landmark.x and landmark.y are already normalized (0.0 to 1.0)
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
            
            # Normalize relative to the wrist (landmark 0)
            # This makes the gesture recognition position-independent
            normalized = self._normalize_landmarks(landmarks)
            
            return normalized, annotated_frame
        
        # No hand detected
        return None, annotated_frame
    
    def _normalize_landmarks(self, landmarks):
        """
        Normalize landmark positions relative to the wrist.
        
        WHY? If your hand is at the left side vs right side of screen,
        raw coordinates will be very different. But relative positions
        (finger distances from wrist) stay the same for the same gesture.
        
        Args:
            landmarks: Raw [x0,y0, x1,y1, ... x20,y20] coordinates
        
        Returns:
            Normalized landmarks as numpy array
        """
        # Reshape to (21, 2) for easier processing
        pts = np.array(landmarks).reshape(21, 2)
        
        # Wrist is landmark 0 — use it as the origin
        wrist = pts[0]
        
        # Subtract wrist position from all points
        pts_normalized = pts - wrist
        
        # Scale by the max distance to make it scale-independent
        # (works whether hand is close or far from camera)
        max_dist = np.max(np.abs(pts_normalized))
        if max_dist > 0:
            pts_normalized = pts_normalized / max_dist
        
        # Flatten back to 42 values
        return pts_normalized.flatten().tolist()
    
    def draw_fps(self, frame, fps):
        """
        Draw FPS counter on the frame.
        
        Args:
            frame: Frame to draw on
            fps: Frames per second value
        
        Returns:
            Frame with FPS overlay
        """
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
        )
        return frame
    
    def draw_gesture_label(self, frame, gesture, confidence=None):
        """
        Draw the predicted gesture label on the frame.
        
        Args:
            frame: Frame to draw on
            gesture: Gesture name string (e.g., "Jump")
            confidence: Optional confidence score (0.0 to 1.0)
        
        Returns:
            Frame with gesture label overlay
        """
        # Gesture-to-color mapping for visual feedback
        color_map = {
            "Left":  (255, 100, 0),    # Blue
            "Right": (0, 150, 255),    # Orange
            "Jump":  (0, 255, 100),    # Green
            "Down":  (0, 0, 255),      # Red
        }
        color = color_map.get(gesture, (255, 255, 255))
        
        # Draw gesture name
        text = gesture
        if confidence is not None:
            text += f" ({confidence*100:.0f}%)"
        
        cv2.putText(
            frame, text,
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, color, 3, cv2.LINE_AA
        )
        return frame
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Hand Detector")
    print("Press Q to quit")
    print("=" * 50)
    
    detector = HandDetector()
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        exit()
    
    import time
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from webcam!")
            break
        
        # Flip the frame (mirror effect — more natural)
        frame = cv2.flip(frame, 1)
        
        # Get landmarks
        landmarks, annotated = detector.get_landmarks(frame)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        
        # Draw FPS
        annotated = detector.draw_fps(annotated, fps)
        
        if landmarks:
            print(f"Detected! First 4 values: {[f'{v:.3f}' for v in landmarks[:4]]}")
            cv2.putText(annotated, "Hand Detected!", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "No Hand", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Hand Detector Test", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")
