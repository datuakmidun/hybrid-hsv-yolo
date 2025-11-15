"""
Redis Handler untuk KRSBI-B
Mengirim data sudut bola dan informasi tracking ke Redis Server

Flow: Kamera → Redis Server → USB TTL → Arduino Mega

Author: KRSBI-B Team
"""

import redis
import json
import time
from datetime import datetime


class RedisHandler:
    """Handler untuk publish data tracking ke Redis"""
    
    def __init__(self, 
                 host='localhost', 
                 port=6379, 
                 db=0,
                 password=None,
                 channel='krsbi_ball_tracking'):
        """
        Initialize Redis connection
        
        Args:
            host: Redis server host (default: localhost)
            port: Redis server port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password (default: None)
            channel: Redis channel name untuk publish
        """
        self.host = host
        self.port = port
        self.db = db
        self.channel = channel
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            print(f"[Redis] ✅ Connected to {host}:{port}")
            print(f"[Redis] Publishing to channel: {channel}")
            
        except redis.ConnectionError as e:
            self.connected = False
            print(f"[Redis] ❌ Connection failed: {e}")
            print(f"[Redis] ⚠️  Running in offline mode")
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'total_errors': 0,
            'last_send_time': 0,
            'avg_latency': 0,
            'total_latency': 0
        }
    
    def publish_ball_data(self, 
                          degree_ball,
                          x_ball=0,
                          y_ball=0,
                          confidence=0.0,
                          velocity_x=0.0,
                          velocity_y=0.0,
                          jarak_cyan=999,
                          jarak_magenta=999,
                          detection_mode='HYBRID',
                          kalman_active=True):
        """
        Publish data bola ke Redis
        
        Args:
            degree_ball: Sudut bola (string atau int)
            x_ball: Posisi X bola
            y_ball: Posisi Y bola
            confidence: YOLO confidence score
            velocity_x: Velocity X dari Kalman
            velocity_y: Velocity Y dari Kalman
            jarak_cyan: Jarak ke goal cyan
            jarak_magenta: Jarak ke goal magenta
            detection_mode: Mode deteksi (HYBRID/HSV_ONLY/YOLO_ONLY)
            kalman_active: Status Kalman filter
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if not self.connected:
            return False
        
        start_time = time.time()
        
        try:
            # Prepare data
            data = {
                # Data utama
                'sudut_bola': str(degree_ball),
                'x': int(x_ball),
                'y': int(y_ball),
                
                # Tracking info
                'confidence': float(confidence),
                'velocity_x': float(velocity_x),
                'velocity_y': float(velocity_y),
                'speed': float((velocity_x**2 + velocity_y**2)**0.5),
                
                # Goal info
                'jarak_cyan': int(jarak_cyan),
                'jarak_magenta': int(jarak_magenta),
                
                # System info
                'detection_mode': detection_mode,
                'kalman_active': kalman_active,
                'timestamp': datetime.now().isoformat(),
                'unix_time': int(time.time() * 1000)
            }
            
            # Convert to JSON
            json_data = json.dumps(data)
            
            # Publish to Redis channel
            subscribers = self.redis_client.publish(self.channel, json_data)
            
            # Also set as key (untuk backup / last value)
            self.redis_client.set('krsbi:ball:latest', json_data)
            self.redis_client.expire('krsbi:ball:latest', 5)  # Expire dalam 5 detik
            
            # Update statistics
            latency = (time.time() - start_time) * 1000  # ms
            self.stats['total_sent'] += 1
            self.stats['total_latency'] += latency
            self.stats['avg_latency'] = self.stats['total_latency'] / self.stats['total_sent']
            self.stats['last_send_time'] = time.time()
            
            return True
            
        except Exception as e:
            self.stats['total_errors'] += 1
            print(f"[Redis] ❌ Publish error: {e}")
            return False
    
    def publish_simple(self, degree_ball):
        """
        Publish hanya sudut bola (simple version)
        
        Args:
            degree_ball: Sudut bola (string atau int)
        
        Returns:
            bool: True jika berhasil
        """
        if not self.connected:
            return False
        
        try:
            # Format sederhana: hanya sudut
            self.redis_client.set('krsbi:sudut', str(degree_ball))
            self.redis_client.expire('krsbi:sudut', 2)
            
            # Publish ke channel
            self.redis_client.publish(self.channel, str(degree_ball))
            
            self.stats['total_sent'] += 1
            return True
            
        except Exception as e:
            self.stats['total_errors'] += 1
            return False
    
    def set_robot_status(self, status_data):
        """
        Set robot status ke Redis (untuk monitoring)
        
        Args:
            status_data: Dictionary dengan info status robot
        """
        if not self.connected:
            return False
        
        try:
            json_data = json.dumps(status_data)
            self.redis_client.set('krsbi:robot:status', json_data)
            self.redis_client.expire('krsbi:robot:status', 10)
            return True
        except:
            return False
    
    def get_command(self):
        """
        Get command dari Redis (jika ada control dari external)
        
        Returns:
            dict or None: Command data atau None jika tidak ada
        """
        if not self.connected:
            return None
        
        try:
            cmd = self.redis_client.get('krsbi:command')
            if cmd:
                self.redis_client.delete('krsbi:command')  # Clear setelah dibaca
                return json.loads(cmd)
            return None
        except:
            return None
    
    def print_stats(self):
        """Print Redis statistics"""
        print("\n" + "="*50)
        print("REDIS HANDLER STATISTICS")
        print("="*50)
        print(f"Connection: {'✅ Connected' if self.connected else '❌ Disconnected'}")
        print(f"Server: {self.host}:{self.port}")
        print(f"Channel: {self.channel}")
        print(f"Total sent: {self.stats['total_sent']}")
        print(f"Total errors: {self.stats['total_errors']}")
        
        if self.stats['total_sent'] > 0:
            success_rate = ((self.stats['total_sent'] - self.stats['total_errors']) / 
                          self.stats['total_sent'] * 100)
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average latency: {self.stats['avg_latency']:.2f}ms")
        
        print("="*50 + "\n")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_sent': 0,
            'total_errors': 0,
            'last_send_time': 0,
            'avg_latency': 0,
            'total_latency': 0
        }
    
    def close(self):
        """Close Redis connection"""
        if self.connected:
            try:
                self.redis_client.close()
                print("[Redis] Connection closed")
            except:
                pass


# ============================================
# TEST SCRIPT
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING REDIS HANDLER")
    print("="*60 + "\n")
    
    # Initialize Redis handler
    redis_handler = RedisHandler(
        host='localhost',
        port=6379,
        channel='krsbi_ball_tracking'
    )
    
    if not redis_handler.connected:
        print("\n[ERROR] Cannot connect to Redis!")
        print("\nTroubleshooting:")
        print("1. Install Redis: sudo apt install redis-server")
        print("2. Start Redis: sudo systemctl start redis")
        print("3. Check status: sudo systemctl status redis")
        print("4. Test connection: redis-cli ping")
        exit(1)
    
    print("\n[INFO] Starting test publish...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    try:
        # Simulate ball tracking data
        for i in range(10):
            degree = (i * 36) % 360  # 0, 36, 72, ..., 324
            x = 320 + (i * 10)
            y = 240 + (i * 5)
            confidence = 0.85 + (i * 0.01)
            vx = i * 0.5
            vy = i * 0.3
            
            success = redis_handler.publish_ball_data(
                degree_ball=degree,
                x_ball=x,
                y_ball=y,
                confidence=confidence,
                velocity_x=vx,
                velocity_y=vy,
                jarak_cyan=100 + i,
                jarak_magenta=200 - i,
                detection_mode='HYBRID',
                kalman_active=True
            )
            
            if success:
                print(f"[{i+1}/10] ✅ Published: sudut={degree}° x={x} y={y} conf={confidence:.2f}")
            else:
                print(f"[{i+1}/10] ❌ Failed to publish")
            
            time.sleep(0.5)
        
        print("\n[INFO] Test completed!")
        redis_handler.print_stats()
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted")
        redis_handler.print_stats()
    
    finally:
        redis_handler.close()
        print("\nDone!")