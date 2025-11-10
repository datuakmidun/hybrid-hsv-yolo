"""
HSV Tuner Tool untuk KRSRI-B
=============================

Tool interaktif untuk mencari nilai HSV threshold yang optimal.
Gunakan tool ini sebelum menjalankan detection untuk tuning warna bola, field, dan goal.

Usage:
    python hsv_tuner.py

Controls:
    - Adjust sliders untuk tuning HSV values
    - Press 's' untuk save current values ke file
    - Press 'q' untuk quit
    - Press 'f' untuk freeze frame

Author: KRSBI-B Team
Date: 2024
"""

import cv2
import numpy as np
import json
from datetime import datetime

class HSVTuner:
    def __init__(self, camera_index=0, target_name="ball"):
        """
        Initialize HSV Tuner
        
        Args:
            camera_index: Camera index
            target_name: Name of target object (ball, field, goal_cyan, etc)
        """
        self.camera_index = camera_index
        self.target_name = target_name
        self.frozen_frame = None
        
        # Default HSV values
        self.h_min = 0
        self.h_max = 180
        self.s_min = 0
        self.s_max = 255
        self.v_min = 0
        self.v_max = 255
        
        # Morphology parameters
        self.erode_size = 0
        self.dilate_size = 0
        self.blur_size = 1  # Must be odd
        
        # Window names
        self.window_controls = f"HSV Tuner - {target_name}"
        self.window_original = "Original"
        self.window_hsv = "HSV Mask"
        self.window_result = "Result"
        
    def nothing(self, x):
        """Dummy callback for trackbar"""
        pass
    
    def create_trackbars(self):
        """Create control window with trackbars"""
        cv2.namedWindow(self.window_controls)
        
        # HSV trackbars
        cv2.createTrackbar("H Min", self.window_controls, self.h_min, 180, self.nothing)
        cv2.createTrackbar("H Max", self.window_controls, self.h_max, 180, self.nothing)
        cv2.createTrackbar("S Min", self.window_controls, self.s_min, 255, self.nothing)
        cv2.createTrackbar("S Max", self.window_controls, self.s_max, 255, self.nothing)
        cv2.createTrackbar("V Min", self.window_controls, self.v_min, 255, self.nothing)
        cv2.createTrackbar("V Max", self.window_controls, self.v_max, 255, self.nothing)
        
        # Morphology trackbars
        cv2.createTrackbar("Erode", self.window_controls, self.erode_size, 20, self.nothing)
        cv2.createTrackbar("Dilate", self.window_controls, self.dilate_size, 20, self.nothing)
        cv2.createTrackbar("Blur", self.window_controls, self.blur_size, 21, self.nothing)
        
    def get_trackbar_values(self):
        """Get current trackbar values"""
        self.h_min = cv2.getTrackbarPos("H Min", self.window_controls)
        self.h_max = cv2.getTrackbarPos("H Max", self.window_controls)
        self.s_min = cv2.getTrackbarPos("S Min", self.window_controls)
        self.s_max = cv2.getTrackbarPos("S Max", self.window_controls)
        self.v_min = cv2.getTrackbarPos("V Min", self.window_controls)
        self.v_max = cv2.getTrackbarPos("V Max", self.window_controls)
        
        self.erode_size = cv2.getTrackbarPos("Erode", self.window_controls)
        self.dilate_size = cv2.getTrackbarPos("Dilate", self.window_controls)
        blur_value = cv2.getTrackbarPos("Blur", self.window_controls)
        
        # Ensure blur is odd
        if blur_value % 2 == 0:
            blur_value += 1
        self.blur_size = max(1, blur_value)
    
    def process_frame(self, frame):
        """
        Process frame with current HSV values
        
        Returns:
            mask, result, info
        """
        # Apply blur
        if self.blur_size > 1:
            frame_blur = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)
        else:
            frame_blur = frame.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        
        # Create mask
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphology
        if self.erode_size > 0:
            kernel = np.ones((self.erode_size, self.erode_size), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        
        if self.dilate_size > 0:
            kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply mask to original frame
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Find contours untuk info
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        # Draw contours and info
        result_annotated = result.copy()
        info = {
            'num_contours': len(contours),
            'areas': []
        }
        
        if len(contours) > 0:
            # Sort by area
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for i, c in enumerate(contours_sorted):
                area = cv2.contourArea(c)
                info['areas'].append(area)
                
                # Draw contour
                cv2.drawContours(result_annotated, [c], 0, (0, 255, 0), 2)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(c)
                
                # Draw bounding box
                cv2.rectangle(result_annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Label
                label = f"#{i+1} A:{int(area)}"
                cv2.putText(result_annotated, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return mask, result_annotated, info
    
    def save_config(self):
        """Save current HSV values to JSON file"""
        config = {
            'target_name': self.target_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'hsv_values': {
                'lower': [self.h_min, self.s_min, self.v_min],
                'upper': [self.h_max, self.s_max, self.v_max],
            },
            'morphology': {
                'erode': self.erode_size,
                'dilate': self.dilate_size,
                'blur': self.blur_size,
            }
        }
        
        filename = f"hsv_config_{self.target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f"Configuration saved to: {filename}")
        print(f"{'='*60}")
        print(f"Target: {self.target_name}")
        print(f"HSV Lower: {config['hsv_values']['lower']}")
        print(f"HSV Upper: {config['hsv_values']['upper']}")
        print(f"Erode: {self.erode_size}, Dilate: {self.dilate_size}, Blur: {self.blur_size}")
        print(f"{'='*60}\n")
        
        return filename
    
    def run(self):
        """Run the tuner"""
        print("="*60)
        print(f"HSV TUNER - {self.target_name.upper()}")
        print("="*60)
        print("Controls:")
        print("  - Adjust sliders to tune HSV values")
        print("  - Press 's' to save configuration")
        print("  - Press 'f' to freeze/unfreeze current frame")
        print("  - Press 'r' to reset to defaults")
        print("  - Press 'q' to quit")
        print("="*60 + "\n")
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create trackbars
        self.create_trackbars()
        
        print("Camera opened successfully!")
        print("Adjust the trackbars to isolate your target object.\n")
        
        while True:
            # Get frame
            if self.frozen_frame is None:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to grab frame")
                    break
            else:
                frame = self.frozen_frame.copy()
            
            # Get current trackbar values
            self.get_trackbar_values()
            
            # Process frame
            mask, result, info = self.process_frame(frame)
            
            # Add info overlay on original frame
            frame_display = frame.copy()
            
            # Info text
            y_offset = 30
            cv2.putText(frame_display, f"Target: {self.target_name}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            cv2.putText(frame_display, f"HSV: [{self.h_min}-{self.h_max}] [{self.s_min}-{self.s_max}] [{self.v_min}-{self.v_max}]",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame_display, f"Morphology: E:{self.erode_size} D:{self.dilate_size} B:{self.blur_size}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame_display, f"Contours: {info['num_contours']}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if self.frozen_frame is not None:
                cv2.putText(frame_display, "FROZEN", (frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display windows
            cv2.imshow(self.window_original, frame_display)
            cv2.imshow(self.window_hsv, mask)
            cv2.imshow(self.window_result, result)
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord('s'):
                self.save_config()
            
            elif key == ord('f'):
                if self.frozen_frame is None:
                    self.frozen_frame = frame.copy()
                    print("\n[INFO] Frame FROZEN - Press 'f' again to unfreeze")
                else:
                    self.frozen_frame = None
                    print("\n[INFO] Frame UNFROZEN")
            
            elif key == ord('r'):
                print("\n[INFO] Reset to default values")
                cv2.setTrackbarPos("H Min", self.window_controls, 0)
                cv2.setTrackbarPos("H Max", self.window_controls, 180)
                cv2.setTrackbarPos("S Min", self.window_controls, 0)
                cv2.setTrackbarPos("S Max", self.window_controls, 255)
                cv2.setTrackbarPos("V Min", self.window_controls, 0)
                cv2.setTrackbarPos("V Max", self.window_controls, 255)
                cv2.setTrackbarPos("Erode", self.window_controls, 0)
                cv2.setTrackbarPos("Dilate", self.window_controls, 0)
                cv2.setTrackbarPos("Blur", self.window_controls, 1)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    target_name = "ball"
    camera_index = 0
    
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            camera_index = int(sys.argv[2])
        except:
            print("Invalid camera index, using default 0")
    
    # Run tuner
    tuner = HSVTuner(camera_index=camera_index, target_name=target_name)
    tuner.run()