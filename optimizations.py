"""
Performance Optimizations untuk KRSBI-B
Tier 1: Quick wins untuk boost FPS
"""

import cv2
import numpy as np

class FrameSkipper:
    """Skip YOLO verification setiap N frames untuk boost FPS"""
    
    def __init__(self, skip_frames=2):
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_result = (False, 0.0, None)
    
    def should_verify(self):
        """Check apakah frame ini perlu di-verify YOLO"""
        self.frame_count += 1
        return (self.frame_count % (self.skip_frames + 1)) == 0
    
    def get_last_result(self):
        """Get hasil verification terakhir"""
        return self.last_result
    
    def update_result(self, result):
        """Update hasil verification"""
        self.last_result = result


class ROIOptimizer:
    """Optimize ROI size untuk reduce YOLO inference time"""
    
    @staticmethod
    def optimize_roi(frame, x, y, w, h, max_size=300):
        """
        Resize ROI jika terlalu besar
        
        Args:
            frame: Full frame
            x, y, w, h: ROI coordinates
            max_size: Maximum dimension (width or height)
        
        Returns:
            resized_roi, scale_factor
        """
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        
        # Check if resize needed
        if max(w, h) > max_size:
            # Calculate scale
            scale = max_size / max(w, h)
            
            # Resize
            new_w = int(w * scale)
            new_h = int(h * scale)
            roi_resized = cv2.resize(roi, (new_w, new_h))
            
            return roi_resized, scale
        else:
            return roi, 1.0