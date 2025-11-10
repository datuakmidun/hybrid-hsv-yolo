"""
YOLO Ball Verifier Module
=========================

Module ini digunakan untuk verify apakah region yang di-detect HSV
benar-benar bola atau false positive (jeruk, cone, dll)

Usage:
    from yolo_verifier import YOLOBallVerifier
    
    verifier = YOLOBallVerifier(model_path="models/yolov8n.pt")
    is_ball, confidence = verifier.verify_region(frame, x, y, w, h)

Author: KRSBI-B Team
Date: 2024
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time


class YOLOBallVerifier:
    """
    Class untuk verify ball detection menggunakan YOLOv8
    """
    
    def __init__(self, 
                 model_path="models/yolov8n.pt",
                 confidence_threshold=0.45,
                 sports_ball_class_id=32,
                 roi_margin=0.3,
                 max_roi_size=320,
                 enable_debug=False):
        """
        Initialize YOLO verifier
        
        Args:
            model_path (str): Path ke YOLOv8 model file
            confidence_threshold (float): Minimum confidence untuk accept detection (0.0-1.0)
            sports_ball_class_id (int): Class ID untuk 'sports ball' di COCO dataset (default: 32)
            roi_margin (float): Margin untuk expand bounding box (0.0-1.0, default: 0.3 = 30%)
            enable_debug (bool): Enable debug output dan visualization
        """
        print(f"[YOLO Verifier] Initializing...")
        print(f"[YOLO Verifier] Model: {model_path}")
        print(f"[YOLO Verifier] Confidence threshold: {confidence_threshold}")
        
        # Load model
        try:
            self.model = YOLO(model_path, verbose=False)
            print(f"[YOLO Verifier] Model loaded successfully!")
        except Exception as e:
            print(f"[YOLO Verifier] ERROR: Failed to load model - {e}")
            raise
        
        self.confidence_threshold = confidence_threshold
        self.sports_ball_class_id = sports_ball_class_id
        self.roi_margin = roi_margin
        self.max_roi_size = max_roi_size
        self.enable_debug = enable_debug
        
        # Statistics
        self.stats = {
            'total_verifications': 0,
            'balls_confirmed': 0,
            'balls_rejected': 0,
            'avg_inference_time': 0,
            'total_inference_time': 0
        }
    
    def expand_bbox(self, x, y, w, h, frame_width, frame_height):
        """
        Expand bounding box dengan margin untuk safety
        
        Args:
            x, y, w, h: Original bounding box
            frame_width, frame_height: Frame dimensions untuk clipping
            
        Returns:
            x_new, y_new, w_new, h_new: Expanded bounding box
        """
        margin_w = int(w * self.roi_margin)
        margin_h = int(h * self.roi_margin)
        
        x_new = max(0, x - margin_w)
        y_new = max(0, y - margin_h)
        w_new = min(frame_width - x_new, w + 2 * margin_w)
        h_new = min(frame_height - y_new, h + 2 * margin_h)
        
        return x_new, y_new, w_new, h_new
    
    def verify_region(self, frame, x, y, w, h):
        """
        Verify apakah region mengandung bola
        
        Args:
            frame: Full image frame (BGR)
            x, y, w, h: Bounding box dari HSV detection
            
        Returns:
            tuple: (is_ball: bool, confidence: float, bbox: tuple or None)
                - is_ball: True jika region mengandung bola
                - confidence: YOLO confidence score (0.0-1.0)
                - bbox: Refined bounding box dari YOLO (x, y, w, h) atau None
        """
        start_time = time.time()
        
        # Expand bbox untuk safety margin
        frame_h, frame_w = frame.shape[:2]
        x_exp, y_exp, w_exp, h_exp = self.expand_bbox(x, y, w, h, frame_w, frame_h)
        
        # Crop region of interest
        roi = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
        
        scale = 1.0
        if max(roi.shape[0], roi.shape[1]) > self.max_roi_size:
            scale = self.max_roi_size / max(roi.shape[0], roi.shape[1])
            new_h = int(roi.shape[0] * scale)
            new_w = int(roi.shape[1] * scale)
            roi = cv2.resize(roi, (new_w, new_h))
        
        # Check if ROI is valid
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            if self.enable_debug:
                print(f"[YOLO Verifier] Invalid ROI size: {roi.shape}")
            return False, 0.0, None
        
        # YOLO inference
        try:
            results = self.model(roi, conf=self.confidence_threshold, verbose=False)
        except Exception as e:
            if self.enable_debug:
                print(f"[YOLO Verifier] Inference error: {e}")
            return False, 0.0, None
        
        # Update statistics
        inference_time = (time.time() - start_time) * 1000  # ms
        self.stats['total_verifications'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['total_verifications']
        )
        
        # Check detections
        detections = results[0].boxes
        
        for detection in detections:
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            
            # Check if it's a sports ball
            if class_id == self.sports_ball_class_id:
                # Get bounding box in ROI coordinates
                bbox_roi = detection.xyxy[0].cpu().numpy()
                x1_roi, y1_roi, x2_roi, y2_roi = bbox_roi
                
                # Convert to original frame coordinates
                x1_frame = int(x_exp + x1_roi)
                y1_frame = int(y_exp + y1_roi)
                x2_frame = int(x_exp + x2_roi)
                y2_frame = int(y_exp + y2_roi)
                
                # Calculate center and dimensions
                w_frame = x2_frame - x1_frame
                h_frame = y2_frame - y1_frame
                
                bbox_frame = (x1_frame, y1_frame, w_frame, h_frame)
                
                if self.enable_debug:
                    print(f"[YOLO Verifier] ✅ Ball confirmed! Confidence: {confidence:.2f}")
                
                self.stats['balls_confirmed'] += 1
                return True, confidence, bbox_frame
        
        # No ball detected
        if self.enable_debug:
            print(f"[YOLO Verifier] ❌ Not a ball (or confidence too low)")
        
        self.stats['balls_rejected'] += 1
        return False, 0.0, None
    
    def verify_contour(self, frame, contour):
        """
        Verify contour dari HSV detection
        
        Args:
            frame: Full image frame
            contour: OpenCV contour dari HSV detection
            
        Returns:
            tuple: (is_ball: bool, confidence: float, bbox: tuple or None)
        """
        # Get bounding box dari contour
        x, y, w, h = cv2.boundingRect(contour)
        
        return self.verify_region(frame, x, y, w, h)
    
    def get_stats(self):
        """
        Get verification statistics
        
        Returns:
            dict: Statistics dictionary
        """
        return self.stats.copy()
    
    def print_stats(self):
        """
        Print verification statistics
        """
        print("\n" + "="*50)
        print("YOLO VERIFIER STATISTICS")
        print("="*50)
        print(f"Total verifications: {self.stats['total_verifications']}")
        print(f"Balls confirmed: {self.stats['balls_confirmed']}")
        print(f"Balls rejected: {self.stats['balls_rejected']}")
        
        if self.stats['total_verifications'] > 0:
            confirm_rate = (self.stats['balls_confirmed'] / 
                          self.stats['total_verifications'] * 100)
            print(f"Confirmation rate: {confirm_rate:.1f}%")
        
        print(f"Average inference time: {self.stats['avg_inference_time']:.1f}ms")
        print("="*50 + "\n")
    
    def reset_stats(self):
        """
        Reset statistics
        """
        self.stats = {
            'total_verifications': 0,
            'balls_confirmed': 0,
            'balls_rejected': 0,
            'avg_inference_time': 0,
            'total_inference_time': 0
        }


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage untuk testing module
    """
    import time
    
    print("Testing YOLO Verifier Module...")
    
    # Initialize verifier
    verifier = YOLOBallVerifier(
        model_path="models/yolov8n.pt",
        confidence_threshold=0.45,
        enable_debug=True
    )
    
    # Test dengan dummy frame dan bbox
    print("\nCreating test frame...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[100:200, 100:200] = [0, 100, 255]  # Orange region
    
    # Test verification
    print("\nTesting verification...")
    is_ball, confidence, bbox = verifier.verify_region(test_frame, 100, 100, 100, 100)
    
    print(f"\nResult:")
    print(f"  Is ball: {is_ball}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Bbox: {bbox}")
    
    # Print stats
    verifier.print_stats()
    
    print("Test complete!")