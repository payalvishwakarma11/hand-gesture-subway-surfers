"""
=============================================================
 BONUS: gui_launcher.py
 Purpose: Tkinter GUI to manage the entire project
=============================================================

FEATURES:
  - Launch each step with one click
  - View live status and logs
  - Monitor dataset statistics
  - Start/stop the game controller
  - Beautiful dark-themed UI
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import csv
import time
from pathlib import Path


# ─── CONFIGURATION ───────────────────────────
DATA_FILE   = "data/gesture_dataset.csv"
MODEL_DIR   = "models"
GESTURES    = ["Left", "Right", "Jump", "Down"]
# ─────────────────────────────────────────────


class GestureProjectGUI:
    """
    Main GUI application for the Hand Gesture Subway Surfers project.
    Provides buttons to run each step of the pipeline.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Hand Gesture Subway Surfers — Control Panel")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)
        
        # Track running processes
        self.running_process = None
        
        self._setup_styles()
        self._build_ui()
        self._update_stats()  # Load initial stats
    
    def _setup_styles(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Progress bar style
        style.configure(
            "Gesture.Horizontal.TProgressbar",
            background="#e94560",
            troughcolor="#16213e",
            bordercolor="#0f3460",
            lightcolor="#e94560",
            darkcolor="#e94560"
        )
    
    def _build_ui(self):
        """Build all UI components."""
        
        # ── HEADER ──────────────────────────────
        header = tk.Frame(self.root, bg="#16213e", pady=15)
        header.pack(fill="x")
        
        tk.Label(
            header,
            text="🖐  Hand Gesture Controller",
            font=("Courier New", 22, "bold"),
            fg="#e94560", bg="#16213e"
        ).pack()
        
        tk.Label(
            header,
            text="Subway Surfers • MediaPipe + TensorFlow",
            font=("Courier New", 11),
            fg="#a8a8b3", bg="#16213e"
        ).pack()
        
        # ── MAIN CONTENT (2 columns) ─────────────
        content = tk.Frame(self.root, bg="#1a1a2e", padx=10, pady=10)
        content.pack(fill="both", expand=True)
        
        # Left column: Steps + Stats
        left = tk.Frame(content, bg="#1a1a2e")
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Right column: Log
        right = tk.Frame(content, bg="#1a1a2e")
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self._build_steps_panel(left)
        self._build_stats_panel(left)
        self._build_log_panel(right)
    
    def _build_steps_panel(self, parent):
        """Build the pipeline steps control panel."""
        
        # Panel header
        frame = tk.Frame(parent, bg="#16213e", bd=1, relief="solid")
        frame.pack(fill="x", pady=(0, 8))
        
        tk.Label(
            frame, text="📋  Pipeline Steps",
            font=("Courier New", 13, "bold"),
            fg="#e94560", bg="#16213e", pady=8
        ).pack()
        
        # Step buttons
        steps = [
            {
                "num": "01",
                "title": "Test Hand Detection",
                "desc": "Opens webcam, shows landmarks",
                "script": "hand_detector.py",
                "color": "#4cc9f0"
            },
            {
                "num": "02",
                "title": "Collect Training Data",
                "desc": "Record gesture samples to CSV",
                "script": "collect_data.py",
                "color": "#7bed9f"
            },
            {
                "num": "03",
                "title": "Train Neural Network",
                "desc": "Train & evaluate the model",
                "script": "train_model.py",
                "color": "#ffa502"
            },
            {
                "num": "04",
                "title": "🎮  Start Game Controller",
                "desc": "Real-time prediction + game control",
                "script": "real_time_predictor.py",
                "color": "#e94560"
            },
        ]
        
        for step in steps:
            self._create_step_button(frame, step)
        
        # Stop button
        stop_btn = tk.Button(
            frame,
            text="⏹  Stop Running Script",
            font=("Courier New", 10),
            bg="#333", fg="#ff4757",
            bd=0, padx=10, pady=6,
            cursor="hand2",
            command=self.stop_script
        )
        stop_btn.pack(fill="x", padx=20, pady=(0, 12))
    
    def _create_step_button(self, parent, step):
        """Create a single step button."""
        btn_frame = tk.Frame(parent, bg="#16213e", pady=2)
        btn_frame.pack(fill="x", padx=20, pady=3)
        
        # Left: step number
        tk.Label(
            btn_frame,
            text=step["num"],
            font=("Courier New", 14, "bold"),
            fg=step["color"], bg="#16213e",
            width=3
        ).pack(side="left")
        
        # Center: title + description
        info = tk.Frame(btn_frame, bg="#16213e")
        info.pack(side="left", fill="x", expand=True, padx=8)
        
        tk.Label(
            info, text=step["title"],
            font=("Courier New", 11, "bold"),
            fg="#ffffff", bg="#16213e",
            anchor="w"
        ).pack(fill="x")
        
        tk.Label(
            info, text=step["desc"],
            font=("Courier New", 9),
            fg="#a8a8b3", bg="#16213e",
            anchor="w"
        ).pack(fill="x")
        
        # Right: Run button
        btn = tk.Button(
            btn_frame,
            text="▶ Run",
            font=("Courier New", 9, "bold"),
            bg=step["color"], fg="#1a1a2e",
            bd=0, padx=12, pady=4,
            cursor="hand2",
            command=lambda s=step["script"]: self.run_script(s)
        )
        btn.pack(side="right")
    
    def _build_stats_panel(self, parent):
        """Build the dataset statistics panel."""
        frame = tk.Frame(parent, bg="#16213e", bd=1, relief="solid")
        frame.pack(fill="x", pady=(8, 0))
        
        header_frame = tk.Frame(frame, bg="#16213e")
        header_frame.pack(fill="x", pady=(8, 4))
        
        tk.Label(
            header_frame, text="📊  Dataset Statistics",
            font=("Courier New", 13, "bold"),
            fg="#e94560", bg="#16213e"
        ).pack(side="left", padx=15)
        
        refresh_btn = tk.Button(
            header_frame,
            text="↻ Refresh",
            font=("Courier New", 8),
            bg="#0f3460", fg="#4cc9f0",
            bd=0, padx=8, pady=2,
            cursor="hand2",
            command=self._update_stats
        )
        refresh_btn.pack(side="right", padx=15)
        
        # Stats content
        self.stats_frame = tk.Frame(frame, bg="#16213e", pady=5)
        self.stats_frame.pack(fill="x", padx=15, pady=(0, 12))
        
        self.stat_bars  = {}
        self.stat_labels = {}
        
        colors = {
            "Left":  "#4cc9f0",
            "Right": "#7bed9f",
            "Jump":  "#ffa502",
            "Down":  "#e94560"
        }
        
        for gesture in GESTURES:
            row = tk.Frame(self.stats_frame, bg="#16213e")
            row.pack(fill="x", pady=2)
            
            tk.Label(
                row, text=f"{gesture:6s}",
                font=("Courier New", 10),
                fg=colors[gesture], bg="#16213e",
                width=7, anchor="w"
            ).pack(side="left")
            
            # Progress bar
            bar_bg = tk.Frame(row, bg="#0f3460", height=14, width=200)
            bar_bg.pack(side="left", padx=5)
            bar_bg.pack_propagate(False)
            
            bar = tk.Frame(bar_bg, bg=colors[gesture], height=14, width=0)
            bar.place(x=0, y=0, height=14, width=0)
            
            lbl = tk.Label(
                row, text="0 samples",
                font=("Courier New", 9),
                fg="#a8a8b3", bg="#16213e",
                width=12
            )
            lbl.pack(side="left")
            
            self.stat_bars[gesture]  = (bar, bar_bg)
            self.stat_labels[gesture] = lbl
    
    def _build_log_panel(self, parent):
        """Build the console log panel."""
        frame = tk.Frame(parent, bg="#16213e", bd=1, relief="solid")
        frame.pack(fill="both", expand=True)
        
        tk.Label(
            frame, text="🖥  Console Output",
            font=("Courier New", 13, "bold"),
            fg="#e94560", bg="#16213e", pady=8
        ).pack()
        
        self.log_text = scrolledtext.ScrolledText(
            frame,
            font=("Courier New", 9),
            bg="#0d0d0d", fg="#00ff88",
            insertbackground="#00ff88",
            wrap="word", bd=0,
            padx=10, pady=10
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Clear log button
        clear_btn = tk.Button(
            frame,
            text="Clear Log",
            font=("Courier New", 8),
            bg="#333", fg="#a8a8b3",
            bd=0, padx=10, pady=3,
            cursor="hand2",
            command=self.clear_log
        )
        clear_btn.pack(pady=(0, 8))
        
        self.log("═" * 50, color="#e94560")
        self.log("  Hand Gesture Project — Control Panel", color="#ffa502")
        self.log("═" * 50, color="#e94560")
        self.log("Click a step button to run it.")
        self.log("Make sure your webcam is connected!")
    
    def log(self, message, color="#00ff88"):
        """Append a message to the log panel."""
        self.log_text.configure(state="normal")
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
    
    def clear_log(self):
        """Clear the log panel."""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
    
    def _update_stats(self):
        """Update gesture sample counts in the stats panel."""
        counts = {g: 0 for g in GESTURES}
        max_count = 300  # Target samples per gesture
        
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if row and row[-1] in counts:
                            counts[row[-1]] += 1
            except Exception:
                pass
        
        # Update UI
        for gesture in GESTURES:
            count = counts[gesture]
            bar, bar_bg = self.stat_bars[gesture]
            label = self.stat_labels[gesture]
            
            # Update progress bar width
            bar_width = bar_bg.winfo_width()
            if bar_width < 10:
                bar_width = 200
            
            fill = min(count / max_count, 1.0)
            filled_px = int(fill * bar_width)
            bar.place(x=0, y=0, height=14, width=filled_px)
            
            # Update label
            status = "✓" if count >= max_count else ""
            label.config(text=f"{count} {status}")
        
        # Model status
        model_exists = os.path.exists(os.path.join(MODEL_DIR, "gesture_model.h5"))
        
        # Schedule next update
        self.root.after(3000, self._update_stats)
    
    def run_script(self, script_name):
        """Run a Python script in a background thread."""
        if self.running_process and self.running_process.poll() is None:
            result = messagebox.askyesno(
                "Script Running",
                "A script is already running. Stop it and start the new one?"
            )
            if result:
                self.stop_script()
            else:
                return
        
        self.log(f"\n{'─'*45}")
        self.log(f"Running: {script_name}", color="#ffa502")
        self.log(f"{'─'*45}")
        
        def run():
            try:
                process = subprocess.Popen(
                    [sys.executable, script_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                self.running_process = process
                
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        self.root.after(0, lambda l=line: self.log(l))
                
                process.wait()
                code = process.returncode
                
                msg = f"\nScript finished (exit code: {code})"
                color = "#7bed9f" if code == 0 else "#ff4757"
                self.root.after(0, lambda: self.log(msg, color))
                self.root.after(0, self._update_stats)
                
            except Exception as e:
                self.root.after(0, lambda: self.log(f"ERROR: {e}", "#ff4757"))
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    
    def stop_script(self):
        """Stop the currently running script."""
        if self.running_process and self.running_process.poll() is None:
            self.running_process.terminate()
            self.log("\n⏹  Script stopped.", color="#ff4757")
        else:
            self.log("No script is currently running.")


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    
    # Try to set window icon
    try:
        root.iconbitmap(default='')
    except Exception:
        pass
    
    app = GestureProjectGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth()  - root.winfo_width())  // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
