"""
Configuration file untuk KRSBI-B Hybrid Detection System
=========================================================

File ini berisi semua konfigurasi HSV thresholds dan parameter lainnya.
Anda bisa tuning nilai-nilai ini tanpa mengubah main code.

Author: KRSBI-B Team
Date: 2024
"""

# ============================================
# CAMERA CONFIGURATION
# ============================================

CAMERA_CONFIG = {
    'camera_index': 0,          # 0 untuk /dev/video0, 1 untuk /dev/video1, dst
    'frame_width': 640,         # Lebar frame (resolusi lebih kecil = FPS lebih tinggi)
    'frame_height': 480,        # Tinggi frame
    'fps': 30,                  # Target FPS (jika kamera support)
}

# ============================================
# YOLO CONFIGURATION
# ============================================

YOLO_CONFIG = {
    'model_path': 'models/yolov8n.pt',     # Path ke model YOLO
    'confidence_threshold': 0.45,           # Confidence threshold (0.0-1.0)
    'roi_margin': 0.3,                      # Margin expansion untuk ROI (30%)
    'enable_debug': False,                  # Debug output
    'enabled_by_default': True,             # Start dengan YOLO enabled
}

# ============================================
# HSV THRESHOLDS - BALL (ORANGE)
# ============================================
# Untuk bola oranye standar KRSBI
# Hue: 0-15 (orange/red)
# Saturation: 100-255 (warna jenuh)
# Value: 100-255 (brightness)

BALL_HSV = {
    'lower': (0, 100, 100),      # Lower bound (H, S, V)
    'upper': (15, 255, 255),     # Upper bound (H, S, V)
    'erode': 2,                  # Erosion kernel size
    'dilate': 2,                 # Dilation kernel size
    'blur': 5,                   # Gaussian blur kernel size (odd number)
    'min_area': 100,             # Minimum contour area (pixels)
    'max_candidates': 2,         # Maximum HSV candidates to verify with YOLO
}

# Alternative: Jika bola Anda lebih ke merah
BALL_HSV_RED_ALT = {
    'lower': (170, 100, 100),
    'upper': (180, 255, 255),
    'erode': 2,
    'dilate': 2,
    'blur': 5,
    'min_area': 100,
    'max_candidates': 2,
}

# ============================================
# HSV THRESHOLDS - FIELD (GREEN)
# ============================================
# Untuk lapangan hijau
# Hue: 40-80 (green)

FIELD_HSV = {
    'lower': (40, 50, 50),
    'upper': (80, 255, 255),
    'erode': 2,
    'dilate': 2,
    'blur': 5,
}

# ============================================
# HSV THRESHOLDS - GOAL CYAN
# ============================================
# Untuk gawang cyan/biru muda
# Hue: 85-95 (cyan)

GOAL_CYAN_HSV = {
    'lower': (85, 100, 100),
    'upper': (95, 255, 255),
    'erode': 2,
    'dilate': 2,
    'blur': 5,
    'min_area': 500,
}

# ============================================
# HSV THRESHOLDS - GOAL MAGENTA
# ============================================
# Untuk gawang magenta/pink
# Hue: 140-170 (magenta)

GOAL_MAGENTA_HSV = {
    'lower': (140, 100, 100),
    'upper': (170, 255, 255),
    'erode': 2,
    'dilate': 2,
    'blur': 5,
    'min_area': 500,
}

# ============================================
# HSV THRESHOLDS - GOAL YELLOW (OPTIONAL)
# ============================================
# Jika ada gawang kuning
# Hue: 20-30 (yellow)

GOAL_YELLOW_HSV = {
    'lower': (20, 100, 100),
    'upper': (30, 255, 255),
    'erode': 2,
    'dilate': 2,
    'blur': 5,
    'min_area': 500,
}

# ============================================
# DISPLAY CONFIGURATION
# ============================================

DISPLAY_CONFIG = {
    'window_name': 'KRSBI-B Hybrid Detection',
    'show_hsv_candidates': True,     # Show HSV candidates sebelum YOLO verify
    'show_rejected': True,           # Show rejected detections dengan X merah
    'show_fps': True,                # Show FPS counter
    'show_crosshair': True,          # Show crosshair di center
    'show_masks': False,             # Show HSV masks (untuk debugging)
    'fps_window_size': 30,           # Window size untuk average FPS
}

# ============================================
# SERIAL COMMUNICATION (ARDUINO)
# ============================================

SERIAL_CONFIG = {
    'enabled': False,               # Enable/disable serial communication
    'port': '/dev/ttyUSB0',        # Serial port
    'baudrate': 115200,            # Baud rate
    'timeout': 0.1,                # Read timeout (seconds)
    'protocol': 'csv',             # Protocol: 'csv' atau 'json'
}

# ============================================
# ADVANCED SETTINGS
# ============================================

ADVANCED_CONFIG = {
    'auto_exposure': False,         # Auto exposure (may affect FPS)
    'auto_white_balance': False,    # Auto white balance
    'save_screenshots_dir': 'screenshots/',  # Directory untuk screenshots
    'log_level': 'INFO',           # Logging level: DEBUG, INFO, WARNING, ERROR
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_config():
    """
    Get complete configuration dictionary
    
    Returns:
        dict: Complete configuration
    """
    return {
        'camera_index': CAMERA_CONFIG['camera_index'],
        'im_width': CAMERA_CONFIG['frame_width'],
        'center_im': (CAMERA_CONFIG['frame_width'] // 2, CAMERA_CONFIG['frame_height'] // 2),
        
        # YOLO
        'model_path': YOLO_CONFIG['model_path'],
        'yolo_confidence': YOLO_CONFIG['confidence_threshold'],
        'yolo_roi_margin': YOLO_CONFIG['roi_margin'],
        'yolo_debug': YOLO_CONFIG['enable_debug'],
        'yolo_enabled': YOLO_CONFIG['enabled_by_default'],
        
        # Ball HSV
        'b_Lower_val': BALL_HSV['lower'],
        'b_Upper_val': BALL_HSV['upper'],
        'Ebsize': BALL_HSV['erode'],
        'Dbsize': BALL_HSV['dilate'],
        'b_blur': BALL_HSV['blur'],
        'ball_min_area': BALL_HSV['min_area'],
        'ball_max_candidates': BALL_HSV['max_candidates'],
        
        # Field HSV
        'f_Lower_val': FIELD_HSV['lower'],
        'f_Upper_val': FIELD_HSV['upper'],
        'Efsize': FIELD_HSV['erode'],
        'Dfsize': FIELD_HSV['dilate'],
        'f_blur': FIELD_HSV['blur'],
        
        # Goal Cyan HSV
        'c_Lower_val': GOAL_CYAN_HSV['lower'],
        'c_Upper_val': GOAL_CYAN_HSV['upper'],
        'Ecsize': GOAL_CYAN_HSV['erode'],
        'Dcsize': GOAL_CYAN_HSV['dilate'],
        'c_blur': GOAL_CYAN_HSV['blur'],
        
        # Goal Magenta HSV
        'm_Lower_val': GOAL_MAGENTA_HSV['lower'],
        'm_Upper_val': GOAL_MAGENTA_HSV['upper'],
        'Emsize': GOAL_MAGENTA_HSV['erode'],
        'Dmsize': GOAL_MAGENTA_HSV['dilate'],
        'm_blur': GOAL_MAGENTA_HSV['blur'],
        
        # Goal Yellow HSV
        'g_Lower_val': GOAL_YELLOW_HSV['lower'],
        'g_Upper_val': GOAL_YELLOW_HSV['upper'],
        'Egsize': GOAL_YELLOW_HSV['erode'],
        'Dgsize': GOAL_YELLOW_HSV['dilate'],
        'g_blur': GOAL_YELLOW_HSV['blur'],
        
        # Display
        'display_config': DISPLAY_CONFIG,
        
        # Serial
        'serial_config': SERIAL_CONFIG,
    }

def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    
    print("\nCAMERA:")
    for key, val in CAMERA_CONFIG.items():
        print(f"  {key}: {val}")
    
    print("\nYOLO:")
    for key, val in YOLO_CONFIG.items():
        print(f"  {key}: {val}")
    
    print("\nBALL HSV:")
    for key, val in BALL_HSV.items():
        print(f"  {key}: {val}")
    
    print("\nGOAL CYAN HSV:")
    for key, val in GOAL_CYAN_HSV.items():
        print(f"  {key}: {val}")
    
    print("\nGOAL MAGENTA HSV:")
    for key, val in GOAL_MAGENTA_HSV.items():
        print(f"  {key}: {val}")
    
    print("="*60 + "\n")


# ============================================
# HSV TUNING TIPS
# ============================================

"""
TIPS UNTUK TUNING HSV VALUES:
==============================

1. HUE (H) - Warna
   - Red/Orange: 0-15 atau 170-180
   - Yellow: 15-30
   - Green: 40-80
   - Cyan: 85-95
   - Blue: 100-130
   - Magenta: 140-170

2. SATURATION (S) - Kejenuhan warna
   - Low (0-100): Warna pucat/pastel
   - Medium (100-200): Warna normal
   - High (200-255): Warna sangat jenuh
   
   Tips: Untuk deteksi robust, gunakan lower=100 untuk filter warna pucat

3. VALUE (V) - Brightness
   - Low (0-100): Gelap
   - Medium (100-200): Normal
   - High (200-255): Terang
   
   Tips: Sesuaikan dengan pencahayaan ruangan Anda

4. EROSION & DILATION
   - Erode: Menghilangkan noise kecil (pixel-pixel terpisah)
   - Dilate: Mengisi gap dalam objek
   - Nilai 2-5 biasanya cukup
   
   Tips: Erode dulu untuk remove noise, baru dilate untuk fill gaps

5. CARA TUNING:
   a. Gunakan HSV color picker tool atau script terpisah
   b. Test di berbagai kondisi pencahayaan
   c. Test dengan objek distractor (cone, jeruk, dll)
   d. Adjust erode/dilate jika ada noise atau gap
   
6. TROUBLESHOOTING:
   - Banyak false positive? → Naikkan lower threshold S dan V
   - Objek tidak terdeteksi? → Turunkan lower threshold
   - Banyak noise? → Naikkan erode size
   - Objek terpotong-potong? → Naikkan dilate size
"""

if __name__ == "__main__":
    print_config()