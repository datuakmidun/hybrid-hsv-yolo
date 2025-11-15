"""
KRSBI-B Hybrid HSV + YOLO Ball Detection System - FIXED
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import imutils
import math

from optimizations import FrameSkipper

class YOLOBallVerifier:
    """Class untuk verify ball detection menggunakan YOLOv8"""
    
    def __init__(self, 
                 model_path="models/yolov8n.pt",
                 confidence_threshold=0.45,
                 sports_ball_class_id=32,
                 roi_margin=0.3,
                 enable_debug=False):
        """Initialize YOLO verifier"""
        print(f"[YOLO Verifier] Initializing...")
        print(f"[YOLO Verifier] Model: {model_path}")
        print(f"[YOLO Verifier] Confidence threshold: {confidence_threshold}")
        
        try:
            self.model = YOLO(model_path, verbose=False)
            print(f"[YOLO Verifier] Model loaded successfully!")
        except Exception as e:
            print(f"[YOLO Verifier] ERROR: Failed to load model - {e}")
            raise
        
        self.confidence_threshold = confidence_threshold
        self.sports_ball_class_id = sports_ball_class_id
        self.roi_margin = roi_margin
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
        """Expand bounding box dengan margin untuk safety"""
        margin_w = int(w * self.roi_margin)
        margin_h = int(h * self.roi_margin)
        
        x_new = max(0, x - margin_w)
        y_new = max(0, y - margin_h)
        w_new = min(frame_width - x_new, w + 2 * margin_w)
        h_new = min(frame_height - y_new, h + 2 * margin_h)
        
        return x_new, y_new, w_new, h_new
    
    def verify_region(self, frame, x, y, w, h):
        """Verify apakah region mengandung bola"""
        start_time = time.time()
        
        # Expand bbox untuk safety margin
        frame_h, frame_w = frame.shape[:2]
        x_exp, y_exp, w_exp, h_exp = self.expand_bbox(x, y, w, h, frame_w, frame_h)
        
        # Crop region of interest
        roi = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]
        
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
            print(f"[YOLO Verifier] ❌ Not a ball")
        
        self.stats['balls_rejected'] += 1
        return False, 0.0, None
    
    def print_stats(self):
        """Print verification statistics"""
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
        """Reset statistics"""
        self.stats = {
            'total_verifications': 0,
            'balls_confirmed': 0,
            'balls_rejected': 0,
            'avg_inference_time': 0,
            'total_inference_time': 0
        }


def putText(frame, text, x, y, r, g, b, scale, thickness):
    """Helper untuk put text dengan warna BGR"""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                scale, (b, g, r), thickness)

def draw_cross_lines(frame):
    """Draw crosshair di center frame"""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    cv2.line(frame, (0, center_y), (w, center_y), (255, 255, 255), 1)
    cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 1)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

def sudut(x1, y1, x2, y2, x3, y3):
    """Hitung sudut antara 3 titik"""
    try:
        deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
        deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
        angle = int(deg2 - deg1)
        if angle < 0:
            angle += 360
        return str(angle)
    except:
        return '500'

def jarak(cx, cy):
    """Hitung jarak dari center frame"""
    return math.sqrt(cx**2 + cy**2)

def parseField(hsv, lower, upper, erode_size, dilate_size):
    """Parse field dengan HSV threshold dan morphology"""
    mask = cv2.inRange(hsv, lower, upper)
    
    if erode_size > 0:
        kernel = np.ones((erode_size, erode_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    
    if dilate_size > 0:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def kirim(degree_ball, param2, param3, jarak_cyan, jarak_magenta):
    """Kirim data ke Arduino/Serial"""
    pass


def TrackTanding(camera, config):
    """Main tracking function dengan hybrid HSV + YOLO detection"""
    
    # Unpack config
    im_width = config.get('im_width', 640)
    center_im = config.get('center_im', (im_width // 2, 240))
    
    # HSV thresholds
    b_Lower_val = config.get('b_Lower_val', (0, 100, 100))
    b_Upper_val = config.get('b_Upper_val', (15, 255, 255))
    Ebsize = config.get('Ebsize', 2)
    Dbsize = config.get('Dbsize', 2)
    
    f_Lower_val = config.get('f_Lower_val', (40, 50, 50))
    f_Upper_val = config.get('f_Upper_val', (80, 255, 255))
    Efsize = config.get('Efsize', 2)
    Dfsize = config.get('Dfsize', 2)
    
    c_Lower_val = config.get('c_Lower_val', (85, 100, 100))
    c_Upper_val = config.get('c_Upper_val', (95, 255, 255))
    Ecsize = config.get('Ecsize', 2)
    Dcsize = config.get('Dcsize', 2)
    
    m_Lower_val = config.get('m_Lower_val', (140, 100, 100))
    m_Upper_val = config.get('m_Upper_val', (170, 255, 255))
    Emsize = config.get('Emsize', 2)
    Dmsize = config.get('Dmsize', 2)
    
    g_Lower_val = config.get('g_Lower_val', (20, 100, 100))
    g_Upper_val = config.get('g_Upper_val', (30, 255, 255))
    Egsize = config.get('Egsize', 2)
    Dgsize = config.get('Dgsize', 2)
    
    # INITIALIZE YOLO VERIFIER
    print("\n" + "="*60)
    print("INITIALIZING HYBRID HSV + YOLO SYSTEM")
    print("="*60)
    
    MODEL_PATH = config.get('model_path', "models/yolov8n.pt")
    
    try:
        yolo_verifier = YOLOBallVerifier(
            model_path=MODEL_PATH,
            confidence_threshold=0.45,
            roi_margin=0.3,
            enable_debug=False
        )
        print("✅ YOLO Verifier ready!")
        yolo_enabled = True
    except Exception as e:
        print(f"❌ YOLO Verifier initialization failed: {e}")
        print("⚠️  Falling back to HSV-only mode")
        yolo_enabled = False
        yolo_verifier = None
    
    print("="*60)
    print("CONTROLS:")
    print("  'q' - Quit")
    print("  'y' - Toggle YOLO verification ON/OFF")
    print("  's' - Show statistics")
    print("  'r' - Reset statistics")
    print("  'c' - Capture screenshot")
    print("="*60 + "\n")
    
    # FPS MONITORING
    frame_count = 0
    start_time = time.time()
    fps_history = []
    FPS_WINDOW = 30
    
    # Frame skipper
    frame_skipper = FrameSkipper(skip_frames=1)
    
    # MAIN LOOP
    while True:
        loop_start = time.time()
        
        grabbed, frame = camera.read()
        if not grabbed:
            print("ERROR: Failed to grab frame")
            break
        
        frame = imutils.resize(frame, width=im_width)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # HSV DETECTION
        mask_ball = parseField(hsv.copy(), b_Lower_val, b_Upper_val, Ebsize, Dbsize)
        mask_field = parseField(hsv.copy(), f_Lower_val, f_Upper_val, Efsize, Dfsize)
        mask_goal = parseField(hsv.copy(), g_Lower_val, g_Upper_val, Egsize, Dgsize)
        mask_cyan = parseField(hsv.copy(), c_Lower_val, c_Upper_val, Ecsize, Dcsize)
        mask_magenta = parseField(hsv.copy(), m_Lower_val, m_Upper_val, Emsize, Dmsize)
        
        # BALL DETECTION - HYBRID HSV + YOLO
        contours_ball = cv2.findContours(mask_ball.copy(), cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        ball_detected = False
        x_ball, y_ball, radius_ball = 0, 0, 0
        degree_ball = '500'
        yolo_confidence = 0.0
        
        if len(contours_ball) > 0:
            candidates = sorted(contours_ball, key=cv2.contourArea, reverse=True)[:2]
            
            for idx, candidate in enumerate(candidates):
                ((x_hsv, y_hsv), radius_hsv) = cv2.minEnclosingCircle(candidate)
                x_hsv, y_hsv, radius_hsv = int(x_hsv), int(y_hsv), int(radius_hsv)
                
                x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(candidate)
                
                cv2.circle(frame, (x_hsv, y_hsv), radius_hsv, (0, 255, 255), 1)
                putText(frame, f'HSV#{idx+1}', x_hsv-20, y_hsv-radius_hsv-10, 
                       0, 255, 255, 0.4, 1)
                
                # YOLO VERIFICATION - FIXED LOGIC
                if yolo_enabled and yolo_verifier:
                    # Cek apakah perlu verify atau reuse hasil sebelumnya
                    if frame_skipper.should_verify():
                        is_ball, confidence, yolo_bbox = yolo_verifier.verify_region(
                            frame, x_bbox, y_bbox, w_bbox, h_bbox
                        )
                        frame_skipper.update_result((is_ball, confidence, yolo_bbox))
                    else:
                        # Reuse hasil frame sebelumnya
                        is_ball, confidence, yolo_bbox = frame_skipper.get_last_result()
                    
                    if is_ball:
                        # YOLO confirmed!
                        ball_detected = True
                        yolo_confidence = confidence
                        
                        if yolo_bbox is not None:
                            x_yolo, y_yolo, w_yolo, h_yolo = yolo_bbox
                            x_ball = x_yolo + w_yolo // 2
                            y_ball = y_yolo + h_yolo // 2
                            radius_ball = max(w_yolo, h_yolo) // 2
                        else:
                            x_ball, y_ball, radius_ball = x_hsv, y_hsv, radius_hsv
                        
                        degree_ball = sudut(x_ball, y_ball, center_im[0], center_im[1], 
                                          center_im[0], 0)
                        
                        cv2.circle(frame, (x_ball, y_ball), radius_ball, (0, 255, 0), 2)
                        cv2.circle(frame, (x_ball, y_ball), 3, (0, 255, 255), -1)
                        
                        putText(frame, f'Ball {confidence:.2f}', 
                               x_ball-30, y_ball-radius_ball-5, 0, 255, 0, 0.5, 2)
                        
                        break
                    else:
                        # YOLO rejected
                        cv2.line(frame, 
                                (x_hsv-radius_hsv, y_hsv-radius_hsv),
                                (x_hsv+radius_hsv, y_hsv+radius_hsv),
                                (0, 0, 255), 2)
                        cv2.line(frame,
                                (x_hsv-radius_hsv, y_hsv+radius_hsv),
                                (x_hsv+radius_hsv, y_hsv-radius_hsv),
                                (0, 0, 255), 2)
                        putText(frame, 'REJECT', x_hsv-25, y_hsv, 0, 0, 255, 0.4, 1)
                
                else:
                    # YOLO disabled, fallback ke HSV only
                    ball_detected = True
                    x_ball, y_ball, radius_ball = x_hsv, y_hsv, radius_hsv
                    degree_ball = sudut(x_ball, y_ball, center_im[0], center_im[1], 
                                      center_im[0], 0)
                    
                    cv2.circle(frame, (x_ball, y_ball), radius_ball, (0, 0, 255), 2)
                    cv2.circle(frame, (x_ball, y_ball), 3, (0, 255, 255), -1)
                    putText(frame, 'Ball', x_ball, y_ball, 0, 0, 255, 0.5, 2)
                    break
        
        # GOAL DETECTION
        jarak_goal_cyan = 999
        contours_cyan = cv2.findContours(mask_cyan.copy(), cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contours_cyan) > 0:
            c = max(contours_cyan, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx_cyan = int(M["m10"] / M["m00"])
                cy_cyan = int(M["m01"] / M["m00"])
                cv2.drawContours(frame, [hull], 0, (255, 255, 0), 2)
                putText(frame, 'Goal_cyan', cx_cyan, cy_cyan, 255, 255, 0, 0.5, 2)
                jarak_goal_cyan = int(jarak(cx_cyan, cy_cyan))
        
        jarak_goal_magenta = 999
        contours_magenta = cv2.findContours(mask_magenta.copy(), cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contours_magenta) > 0:
            c = max(contours_magenta, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx_magenta = int(M["m10"] / M["m00"])
                cy_magenta = int(M["m01"] / M["m00"])
                cv2.drawContours(frame, [hull], 0, (255, 0, 255), 2)
                putText(frame, 'Goal_magenta', cx_magenta, cy_magenta, 255, 0, 255, 0.5, 2)
                jarak_goal_magenta = int(jarak(cx_magenta, cy_magenta))
        
        # FPS CALCULATION
        loop_time = time.time() - loop_start
        instant_fps = 1.0 / loop_time if loop_time > 0 else 0
        fps_history.append(instant_fps)
        
        if len(fps_history) > FPS_WINDOW:
            fps_history.pop(0)
        
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        
        frame_count += 1
        elapsed = time.time() - start_time
        overall_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # VISUALIZATION
        status_color = (0, 255, 0) if yolo_enabled else (0, 165, 255)
        status_text = "HYBRID MODE" if yolo_enabled else "HSV ONLY"
        putText(frame, status_text, 10, 20, status_color[0], status_color[1], 
               status_color[2], 0.6, 2)
        
        putText(frame, f'FPS: {avg_fps:.1f}', 10, 40, 255, 255, 255, 0.5, 2)
        
        text1 = f'Sudut_Ball= {str(degree_ball)}'
        putText(frame, text1, 10, 60, 0, 255, 0, 0.4, 2)
        
        if yolo_enabled and ball_detected and yolo_confidence > 0:
            text_conf = f'Confidence= {yolo_confidence:.2f}'
            putText(frame, text_conf, 10, 75, 0, 255, 0, 0.4, 2)
        
        text2 = f'Jarak_Cyan= {str(int(jarak_goal_cyan))}'
        putText(frame, text2, 10, 95, 0, 0, 255, 0.4, 2)
        
        text3 = f'Jarak_Magenta= {str(int(jarak_goal_magenta))}'
        putText(frame, text3, 10, 110, 0, 0, 255, 0.4, 2)
        
        kirim(degree_ball, 0, 0, jarak_goal_cyan, jarak_goal_magenta)
        
        draw_cross_lines(frame)
        
        if ball_detected:
            try:
                putText(frame, f'{degree_ball}', x_ball, y_ball-15, 255, 0, 0, 0.8, 2)
            except:
                pass
        
        cv2.imshow('KRSBI-B Hybrid Detection', frame)
        
        # KEYBOARD CONTROL
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):
            if yolo_enabled and yolo_verifier:
                print("\n" + "="*60)
                yolo_verifier.print_stats()
            break
        
        elif k == ord('y'):
            if yolo_verifier:
                yolo_enabled = not yolo_enabled
                status = "ENABLED" if yolo_enabled else "DISABLED"
                print(f"\n[INFO] YOLO Verification {status}")
        
        elif k == ord('s'):
            if yolo_enabled and yolo_verifier:
                yolo_verifier.print_stats()
            print(f"\nFPS Statistics:")
            print(f"  Current FPS: {avg_fps:.1f}")
            print(f"  Overall FPS: {overall_fps:.1f}")
            print(f"  Total frames: {frame_count}\n")
        
        elif k == ord('r'):
            if yolo_enabled and yolo_verifier:
                yolo_verifier.reset_stats()
                print("\n[INFO] Statistics reset")
            frame_count = 0
            start_time = time.time()
            fps_history = []
        
        elif k == ord('c'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n[INFO] Screenshot saved: {filename}")
    
    cv2.destroyAllWindows()
    return yolo_verifier


if __name__ == "__main__":
    print("="*60)
    print("KRSBI-B HYBRID HSV + YOLO DETECTION SYSTEM")
    print("="*60)
    
    config = {
        'camera_index': 0,
        'im_width': 640,
        'center_im': (320, 240),
        'model_path': 'models/yolov8n.pt',
        'b_Lower_val': (0, 100, 100),
        'b_Upper_val': (15, 255, 255),
        'Ebsize': 2,
        'Dbsize': 2,
        'f_Lower_val': (40, 50, 50),
        'f_Upper_val': (80, 255, 255),
        'Efsize': 2,
        'Dfsize': 2,
        'c_Lower_val': (85, 100, 100),
        'c_Upper_val': (95, 255, 255),
        'Ecsize': 2,
        'Dcsize': 2,
        'm_Lower_val': (140, 100, 100),
        'm_Upper_val': (170, 255, 255),
        'Emsize': 2,
        'Dmsize': 2,
        'g_Lower_val': (20, 100, 100),
        'g_Upper_val': (30, 255, 255),
        'Egsize': 2,
        'Dgsize': 2,
    }
    
    print(f"\nOpening camera at index {config['camera_index']}...")
    camera = cv2.VideoCapture(config['camera_index'])
    
    if not camera.isOpened():
        print("ERROR: Cannot open camera!")
        exit(1)
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['im_width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera opened successfully!")
    print(f"Resolution: {config['im_width']}x480")
    
    try:
        yolo_verifier = TrackTanding(camera, config)
        
        if yolo_verifier:
            print("\n" + "="*60)
            print("FINAL STATISTICS")
            print("="*60)
            yolo_verifier.print_stats()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")
        print("="*60)