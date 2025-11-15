"""
Kalman Filter untuk Ball Tracking - KRSBI-B
Smooth tracking dan prediksi posisi bola
"""

import numpy as np
import cv2

class BallKalmanTracker:
    """
    Kalman Filter untuk tracking posisi bola
    State: [x, y, vx, vy] - posisi dan velocity
    """
    
    def __init__(self, process_noise=1.0, measurement_noise=10.0):
        """
        Initialize Kalman Filter
        
        Args:
            process_noise: Process noise covariance (Q) - default 1.0
            measurement_noise: Measurement noise covariance (R) - default 10.0
        """
        # Create Kalman Filter (4 state variables, 2 measurements)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # State transition matrix (A)
        # x' = x + vx*dt
        # y' = y + vy*dt
        # vx' = vx
        # vy' = vy
        dt = 1.0  # Time step
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (H)
        # We only measure x and y
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (Q)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance (R)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance (P)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000
        
        # Tracking state
        self.initialized = False
        self.last_measurement = None
        self.frames_without_detection = 0
        self.max_frames_without_detection = 10
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'total_predictions': 0,
            'lost_tracks': 0
        }
    
    def initialize(self, x, y):
        """
        Initialize Kalman filter dengan deteksi pertama
        
        Args:
            x, y: Initial position
        """
        self.kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32)
        self.initialized = True
        self.last_measurement = (x, y)
        self.frames_without_detection = 0
        print(f"[Kalman] Initialized at ({x}, {y})")
    
    def update(self, x, y):
        """
        Update Kalman filter dengan measurement baru
        
        Args:
            x, y: Measured position
        
        Returns:
            tuple: (corrected_x, corrected_y, vx, vy)
        """
        if not self.initialized:
            self.initialize(x, y)
            return x, y, 0, 0
        
        # Measurement
        measurement = np.array([[x], [y]], dtype=np.float32)
        
        # Predict
        prediction = self.kalman.predict()
        
        # Correct (update dengan measurement)
        corrected = self.kalman.correct(measurement)
        
        # Extract state
        corrected_x = int(corrected[0])
        corrected_y = int(corrected[1])
        vx = corrected[2]
        vy = corrected[3]
        
        self.last_measurement = (x, y)
        self.frames_without_detection = 0
        self.stats['total_updates'] += 1
        
        return corrected_x, corrected_y, vx, vy
    
    def predict(self):
        """
        Predict posisi berikutnya (tanpa measurement)
        Digunakan saat bola tidak terdeteksi
        
        Returns:
            tuple: (predicted_x, predicted_y, vx, vy) or None if lost
        """
        if not self.initialized:
            return None
        
        self.frames_without_detection += 1
        
        # Check if track lost
        if self.frames_without_detection > self.max_frames_without_detection:
            print(f"[Kalman] Track lost after {self.frames_without_detection} frames")
            self.reset()
            self.stats['lost_tracks'] += 1
            return None
        
        # Predict without measurement
        prediction = self.kalman.predict()
        
        predicted_x = int(prediction[0])
        predicted_y = int(prediction[1])
        vx = prediction[2]
        vy = prediction[3]
        
        self.stats['total_predictions'] += 1
        
        return predicted_x, predicted_y, vx, vy
    
    def reset(self):
        """Reset tracker"""
        self.initialized = False
        self.last_measurement = None
        self.frames_without_detection = 0
    
    def is_initialized(self):
        """Check if tracker is initialized"""
        return self.initialized
    
    def get_velocity(self):
        """Get current velocity estimate"""
        if not self.initialized:
            return 0, 0
        
        state = self.kalman.statePost
        return state[2], state[3]
    
    def get_speed(self):
        """Get current speed (magnitude of velocity)"""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)
    
    def print_stats(self):
        """Print tracking statistics"""
        print("\n" + "="*50)
        print("KALMAN FILTER STATISTICS")
        print("="*50)
        print(f"Total updates: {self.stats['total_updates']}")
        print(f"Total predictions: {self.stats['total_predictions']}")
        print(f"Lost tracks: {self.stats['lost_tracks']}")
        print(f"Current state: {'Tracking' if self.initialized else 'Not initialized'}")
        if self.initialized:
            vx, vy = self.get_velocity()
            speed = self.get_speed()
            print(f"Current velocity: ({vx:.2f}, {vy:.2f})")
            print(f"Current speed: {speed:.2f} px/frame")
        print("="*50 + "\n")


class MultiObjectKalmanTracker:
    """
    Kalman tracker untuk multiple objects (ball, goals, etc)
    """
    
    def __init__(self):
        self.trackers = {}
    
    def get_tracker(self, object_id, process_noise=1.0, measurement_noise=10.0):
        """Get or create tracker untuk object_id"""
        if object_id not in self.trackers:
            self.trackers[object_id] = BallKalmanTracker(process_noise, measurement_noise)
        return self.trackers[object_id]
    
    def reset_tracker(self, object_id):
        """Reset specific tracker"""
        if object_id in self.trackers:
            self.trackers[object_id].reset()
    
    def reset_all(self):
        """Reset all trackers"""
        for tracker in self.trackers.values():
            tracker.reset()


# ============================================
# HELPER FUNCTIONS
# ============================================

def draw_kalman_info(frame, x, y, vx, vy, color=(0, 255, 0), label="Ball"):
    """
    Draw Kalman filter info on frame
    
    Args:
        frame: Image frame
        x, y: Current position
        vx, vy: Velocity
        color: Color for visualization
        label: Text label
    """
    # Draw position
    cv2.circle(frame, (x, y), 5, color, -1)
    cv2.circle(frame, (x, y), 10, color, 2)
    
    # Draw velocity vector
    if abs(vx) > 0.1 or abs(vy) > 0.1:
        # Scale velocity for visualization
        scale = 10
        end_x = int(x + vx * scale)
        end_y = int(y + vy * scale)
        cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2, tipLength=0.3)
    
    # Draw label
    speed = np.sqrt(vx**2 + vy**2)
    text = f"{label} v={speed:.1f}"
    cv2.putText(frame, text, (x + 15, y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_prediction_trail(frame, predictions, color=(255, 255, 0)):
    """
    Draw predicted trajectory
    
    Args:
        frame: Image frame
        predictions: List of (x, y) predicted positions
        color: Trail color
    """
    if len(predictions) < 2:
        return
    
    # Draw trail
    for i in range(len(predictions) - 1):
        pt1 = predictions[i]
        pt2 = predictions[i + 1]
        cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw prediction end point
    if predictions:
        cv2.circle(frame, predictions[-1], 8, color, 2)


if __name__ == "__main__":
    # Test Kalman Filter
    print("Testing Kalman Filter...")
    
    tracker = BallKalmanTracker(process_noise=1.0, measurement_noise=10.0)
    
    # Simulate ball movement
    print("\nSimulating ball trajectory:")
    for i in range(20):
        # Simulate measurement with noise
        true_x = 100 + i * 10
        true_y = 100 + i * 5
        noise_x = np.random.randn() * 5
        noise_y = np.random.randn() * 5
        
        measured_x = true_x + noise_x
        measured_y = true_y + noise_y
        
        # Update Kalman
        kf_x, kf_y, vx, vy = tracker.update(measured_x, measured_y)
        
        print(f"Frame {i}: Measured=({measured_x:.1f}, {measured_y:.1f}) "
              f"Kalman=({kf_x}, {kf_y}) Velocity=({vx:.2f}, {vy:.2f})")
    
    tracker.print_stats()