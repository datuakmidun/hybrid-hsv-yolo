"""
Redis to Arduino Serial Forwarder - KRSBI-B
============================================

Mengambil data sudut bola dari Redis dan mengirim ke Arduino Mega via USB TTL

Flow: Redis Server → Subscribe Channel → Parse Data → Serial USB → Arduino

Author: KRSBI-B Team
"""

import redis
import serial
import json
import time
import sys
from datetime import datetime


class RedisToSerial:
    """Forward data dari Redis ke Arduino via Serial"""
    
    def __init__(self,
                 redis_host='localhost',
                 redis_port=6379,
                 redis_channel='krsbi_ball_tracking',
                 serial_port='/dev/ttyUSB0',
                 baudrate=115200,
                 timeout=1):
        """
        Initialize Redis subscriber dan Serial connection
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_channel: Redis channel untuk subscribe
            serial_port: Serial port Arduino (Linux: /dev/ttyUSB0, Windows: COM3)
            baudrate: Serial baudrate (default: 115200)
            timeout: Serial timeout (default: 1 second)
        """
        self.redis_channel = redis_channel
        self.serial_port_name = serial_port
        self.baudrate = baudrate
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'errors': 0,
            'start_time': time.time(),
            'last_message_time': 0
        }
        
        # Initialize Redis
        self.redis_connected = False
        self.serial_connected = False
        
        self._connect_redis(redis_host, redis_port)
        self._connect_serial(serial_port, baudrate, timeout)
    
    def _connect_redis(self, host, port):
        """Connect to Redis server"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=2
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Create pubsub object
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe(self.redis_channel)
            
            self.redis_connected = True
            print(f"[Redis] ✅ Connected to {host}:{port}")
            print(f"[Redis] 📡 Subscribed to channel: {self.redis_channel}")
            
        except redis.ConnectionError as e:
            self.redis_connected = False
            print(f"[Redis] ❌ Connection failed: {e}")
            print(f"[Redis] Please check if Redis server is running")
    
    def _connect_serial(self, port, baudrate, timeout):
        """Connect to Arduino via Serial"""
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            
            # Wait for Arduino to reset
            time.sleep(2)
            
            # Flush buffers
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            self.serial_connected = True
            print(f"[Serial] ✅ Connected to {port} at {baudrate} baud")
            
        except serial.SerialException as e:
            self.serial_connected = False
            print(f"[Serial] ❌ Connection failed: {e}")
            print(f"[Serial] Available ports:")
            self._list_serial_ports()
    
    def _list_serial_ports(self):
        """List available serial ports"""
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("  No serial ports found")
        else:
            for port in ports:
                print(f"  - {port.device}: {port.description}")
    
    def parse_message(self, message):
        """
        Parse Redis message dan extract data untuk Arduino
        
        Args:
            message: Redis message (JSON string)
        
        Returns:
            str: Formatted string untuk Arduino atau None jika error
        """
        try:
            # Parse JSON
            data = json.loads(message)
            
            # Extract data
            sudut = data.get('sudut_bola', '500')
            x = data.get('x', 0)
            y = data.get('y', 0)
            confidence = data.get('confidence', 0.0)
            speed = data.get('speed', 0.0)
            jarak_cyan = data.get('jarak_cyan', 999)
            jarak_magenta = data.get('jarak_magenta', 999)
            
            # Format: SUDUT,X,Y,CONF,SPEED,CYAN,MAGENTA
            # Contoh: 45,320,240,0.87,2.5,150,200
            formatted = f"{sudut},{x},{y},{confidence:.2f},{speed:.1f},{jarak_cyan},{jarak_magenta}\n"
            
            return formatted
            
        except json.JSONDecodeError as e:
            print(f"[Parse] ❌ JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"[Parse] ❌ Parse error: {e}")
            return None
    
    def send_to_arduino(self, data):
        """
        Send data ke Arduino via Serial
        
        Args:
            data: String data untuk dikirim
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if not self.serial_connected:
            return False
        
        try:
            # Send data
            self.serial_port.write(data.encode('utf-8'))
            self.serial_port.flush()
            
            self.stats['messages_sent'] += 1
            self.stats['last_message_time'] = time.time()
            
            return True
            
        except serial.SerialException as e:
            self.stats['errors'] += 1
            print(f"[Serial] ❌ Write error: {e}")
            return False
    
    def read_arduino_response(self):
        """
        Read response dari Arduino (optional)
        
        Returns:
            str or None: Response dari Arduino
        """
        if not self.serial_connected:
            return None
        
        try:
            if self.serial_port.in_waiting > 0:
                response = self.serial_port.readline().decode('utf-8').strip()
                return response
            return None
        except:
            return None
    
    def run(self, verbose=True, show_arduino_response=False):
        """
        Main loop: Subscribe Redis → Send to Arduino
        
        Args:
            verbose: Print setiap message yang dikirim
            show_arduino_response: Print response dari Arduino
        """
        if not self.redis_connected:
            print("\n[ERROR] Cannot start: Redis not connected!")
            return
        
        if not self.serial_connected:
            print("\n[WARNING] Serial not connected, running in Redis-only mode")
        
        print("\n" + "="*60)
        print("REDIS TO ARDUINO FORWARDER - RUNNING")
        print("="*60)
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        try:
            # Listen for messages
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    self.stats['messages_received'] += 1
                    
                    # Parse message
                    raw_data = message['data']
                    formatted_data = self.parse_message(raw_data)
                    
                    if formatted_data:
                        # Send to Arduino
                        success = self.send_to_arduino(formatted_data)
                        
                        if verbose:
                            status = "✅" if success else "❌"
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            print(f"[{timestamp}] {status} {formatted_data.strip()}")
                        
                        # Read Arduino response
                        if show_arduino_response and self.serial_connected:
                            response = self.read_arduino_response()
                            if response:
                                print(f"  ← Arduino: {response}")
                    
                    # Print stats setiap 100 message
                    if self.stats['messages_received'] % 100 == 0:
                        self._print_inline_stats()
        
        except KeyboardInterrupt:
            print("\n\n[INFO] Stopping forwarder...")
            self.print_stats()
        
        except Exception as e:
            print(f"\n[ERROR] Runtime error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.close()
    
    def _print_inline_stats(self):
        """Print stats inline (without newline spam)"""
        uptime = time.time() - self.stats['start_time']
        rate = self.stats['messages_received'] / uptime if uptime > 0 else 0
        print(f"\n[Stats] Received: {self.stats['messages_received']} | "
              f"Sent: {self.stats['messages_sent']} | "
              f"Errors: {self.stats['errors']} | "
              f"Rate: {rate:.1f} msg/s\n")
    
    def print_stats(self):
        """Print detailed statistics"""
        print("\n" + "="*60)
        print("REDIS TO SERIAL STATISTICS")
        print("="*60)
        
        uptime = time.time() - self.stats['start_time']
        
        print(f"Uptime: {uptime:.1f} seconds")
        print(f"Messages received: {self.stats['messages_received']}")
        print(f"Messages sent: {self.stats['messages_sent']}")
        print(f"Errors: {self.stats['errors']}")
        
        if uptime > 0:
            rate = self.stats['messages_received'] / uptime
            print(f"Average rate: {rate:.2f} messages/second")
        
        if self.stats['messages_received'] > 0:
            success_rate = (self.stats['messages_sent'] / 
                          self.stats['messages_received'] * 100)
            print(f"Success rate: {success_rate:.1f}%")
        
        print("="*60 + "\n")
    
    def close(self):
        """Close connections"""
        print("\n[INFO] Closing connections...")
        
        if self.redis_connected:
            try:
                self.pubsub.unsubscribe()
                self.pubsub.close()
                self.redis_client.close()
                print("[Redis] Connection closed")
            except:
                pass
        
        if self.serial_connected:
            try:
                self.serial_port.close()
                print("[Serial] Connection closed")
            except:
                pass


# ============================================
# COMMAND LINE INTERFACE
# ============================================

def main():
    """Main entry point dengan argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Forward data dari Redis ke Arduino via USB TTL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (localhost Redis, /dev/ttyUSB0)
  python3 redis_to_serial.py
  
  # Custom serial port
  python3 redis_to_serial.py --port /dev/ttyUSB1
  
  # Custom Redis server
  python3 redis_to_serial.py --redis-host 192.168.1.100
  
  # Verbose mode dengan Arduino response
  python3 redis_to_serial.py --verbose --show-response
  
  # List available serial ports
  python3 redis_to_serial.py --list-ports
        """
    )
    
    parser.add_argument('--redis-host', default='localhost',
                       help='Redis server host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis server port (default: 6379)')
    parser.add_argument('--redis-channel', default='krsbi_ball_tracking',
                       help='Redis channel name (default: krsbi_ball_tracking)')
    parser.add_argument('--port', '-p', default='/dev/ttyUSB0',
                       help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baudrate', '-b', type=int, default=115200,
                       help='Serial baudrate (default: 115200)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print every message sent')
    parser.add_argument('--show-response', '-r', action='store_true',
                       help='Show Arduino response')
    parser.add_argument('--list-ports', '-l', action='store_true',
                       help='List available serial ports and exit')
    
    args = parser.parse_args()
    
    # List ports and exit
    if args.list_ports:
        print("\n" + "="*60)
        print("AVAILABLE SERIAL PORTS")
        print("="*60)
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found")
        else:
            for port in ports:
                print(f"\nPort: {port.device}")
                print(f"  Description: {port.description}")
                print(f"  Hardware ID: {port.hwid}")
        print("="*60 + "\n")
        return
    
    # Print configuration
    print("\n" + "="*60)
    print("REDIS TO ARDUINO FORWARDER")
    print("="*60)
    print(f"Redis Server: {args.redis_host}:{args.redis_port}")
    print(f"Redis Channel: {args.redis_channel}")
    print(f"Serial Port: {args.port}")
    print(f"Baudrate: {args.baudrate}")
    print(f"Verbose: {args.verbose}")
    print(f"Show Arduino Response: {args.show_response}")
    print("="*60 + "\n")
    
    # Initialize and run
    forwarder = RedisToSerial(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_channel=args.redis_channel,
        serial_port=args.port,
        baudrate=args.baudrate
    )
    
    if forwarder.redis_connected or forwarder.serial_connected:
        forwarder.run(verbose=args.verbose, show_arduino_response=args.show_response)
    else:
        print("\n[ERROR] Cannot start: No connections available!")
        print("\nTroubleshooting:")
        print("1. Check Redis: redis-cli ping")
        print("2. Check USB: ls -l /dev/ttyUSB*")
        print("3. Add user to dialout group: sudo usermod -a -G dialout $USER")
        print("4. Reconnect USB cable")


if __name__ == "__main__":
    main()