"""
=============================================================
 utils/gesture_smoother.py
 Purpose: Smooth out noisy gesture predictions
=============================================================

THE PROBLEM:
  Raw ML predictions jump around a lot frame-to-frame.
  For example, frame sequence might be:
    Jump, Jump, Left, Jump, Jump, Right, Jump
  But you actually meant: Jump

  This makes the game unplayable because it sends
  random keyboard presses.

THE SOLUTION — Prediction Buffer:
  Keep a rolling window of the last N predictions.
  Only output a gesture if it appears consistently.
  
  Window: [Jump, Jump, Jump, Jump, Jump]
  → Stable! Output: Jump ✓

  Window: [Jump, Left, Jump, Right, Jump]
  → Unstable! Output: Nothing (wait for stability)
"""

from collections import deque
import time


class GestureSmoother:
    """
    Smooths gesture predictions using a rolling majority vote.
    
    How it works:
    1. Keep a deque (sliding window) of last N predictions
    2. Count votes for each gesture
    3. Only confirm a gesture if it has >= threshold% of votes
    4. Rate-limit output to prevent rapid key-pressing
    
    Usage:
        smoother = GestureSmoother(window_size=10, threshold=0.7)
        
        # In prediction loop:
        stable_gesture = smoother.update("Jump", 0.95)
        if stable_gesture:
            # Send keypress
    """
    
    def __init__(self, window_size=10, threshold=0.7, cooldown=0.3):
        """
        Initialize the smoother.
        
        Args:
            window_size (int): Number of recent predictions to consider.
                               Larger = smoother but more lag.
                               Smaller = more responsive but noisier.
            threshold (float): Fraction of window that must agree (0.0-1.0).
                               0.7 = 70% of recent predictions must match.
            cooldown (float): Minimum seconds between confirmed gestures.
                              Prevents holding a gesture from spamming keys.
        """
        self.window_size = window_size
        self.threshold   = threshold
        self.cooldown    = cooldown
        
        # Rolling window of recent predictions
        self.prediction_buffer = deque(maxlen=window_size)
        
        # Confidence buffer for weighted voting
        self.confidence_buffer = deque(maxlen=window_size)
        
        # Tracking last output
        self.last_gesture  = None
        self.last_sent_time = 0
        
        print(f"[GestureSmoother] window={window_size}, threshold={threshold}, cooldown={cooldown}s")
    
    def update(self, gesture, confidence=1.0):
        """
        Add a new prediction and get the smoothed output.
        
        Args:
            gesture: Predicted gesture string ("Left", "Right", "Jump", "Down")
                     or None if no hand detected
            confidence: Model confidence (0.0 to 1.0)
        
        Returns:
            str or None: Confirmed stable gesture, or None if still unstable
        """
        # If no hand detected, clear the buffer
        if gesture is None:
            self.prediction_buffer.clear()
            self.confidence_buffer.clear()
            return None
        
        # Add to rolling window
        self.prediction_buffer.append(gesture)
        self.confidence_buffer.append(confidence)
        
        # Need a full window before making decisions
        if len(self.prediction_buffer) < self.window_size:
            return None
        
        # Find the most common prediction (majority vote)
        counts = {}
        for pred in self.prediction_buffer:
            counts[pred] = counts.get(pred, 0) + 1
        
        best_gesture = max(counts, key=counts.get)
        vote_fraction = counts[best_gesture] / self.window_size
        
        # Only output if above threshold
        if vote_fraction < self.threshold:
            return None  # Not stable enough
        
        # Rate limiting — don't spam the same key
        current_time = time.time()
        
        # Allow if: gesture changed OR enough time has passed
        if (best_gesture != self.last_gesture or 
            current_time - self.last_sent_time >= self.cooldown):
            
            self.last_gesture   = best_gesture
            self.last_sent_time = current_time
            return best_gesture
        
        return None  # Cooldown not expired
    
    def get_current_prediction(self):
        """
        Get the current most common prediction WITHOUT outputting it.
        Useful for displaying what the model sees right now.
        
        Returns:
            tuple: (gesture_name, vote_fraction) or (None, 0)
        """
        if not self.prediction_buffer:
            return None, 0.0
        
        counts = {}
        for pred in self.prediction_buffer:
            counts[pred] = counts.get(pred, 0) + 1
        
        if not counts:
            return None, 0.0
        
        best = max(counts, key=counts.get)
        fraction = counts[best] / len(self.prediction_buffer)
        return best, fraction
    
    def reset(self):
        """Clear all buffers — useful when starting a new game session."""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.last_gesture   = None
        self.last_sent_time = 0
        print("[GestureSmoother] Reset!")
    
    def get_stats(self):
        """Return debugging stats about current buffer state."""
        if not self.prediction_buffer:
            return {"status": "empty", "size": 0}
        
        counts = {}
        for pred in self.prediction_buffer:
            counts[pred] = counts.get(pred, 0) + 1
        
        return {
            "buffer_fill": len(self.prediction_buffer),
            "window_size": self.window_size,
            "counts": counts,
            "last_gesture": self.last_gesture
        }
