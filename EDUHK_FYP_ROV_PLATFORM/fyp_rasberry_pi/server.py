import time
import cv2
import socket
import pickle
import struct
import threading
import serial  # Correct import for pyserial

arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

def handle_commands(command_conn):
    """Handle incoming commands from the client."""
    while True:
        command = command_conn.recv(1024).decode()  # Buffer size is 1024 bytes
        if not command:
            break
        arduino.write((command + '\n').encode())
        print(f"Received command: {command}")

def start_video_stream(video_socket):
    """Stream video from the camera to the connected client."""
    cap = cv2.VideoCapture(0)

    # Set a smaller resolution to reduce latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Serialize the frame
        data = pickle.dumps(frame)
        message_size = struct.pack("Q", len(data))  # Use "Q" for 64-bit
        # Send the frame size followed by the frame data
        video_socket.sendall(message_size + data)

    cap.release()
    video_socket.close()

def wait_for_connections(video_socket, command_socket):
    """Wait for video and command connections."""
    while True:
        print("Waiting for a video connection...")
        conn, addr = video_socket.accept()
        print(f"Video connection from {addr}")

        print("Waiting for a command connection...")
        command_conn, addr = command_socket.accept()
        print(f"Command connection from {addr}")

        # Start a thread to handle incoming commands
        command_thread = threading.Thread(target=handle_commands, args=(command_conn,))
        command_thread.start()

        # Start streaming video in a separate thread
        video_thread = threading.Thread(target=start_video_stream, args=(conn,))
        video_thread.start()

        # Wait for threads to finish
        command_thread.join()
        video_thread.join()

        # Clean up
        command_conn.close()
        conn.close()
        print("Connections closed, waiting for new connections...")

def main():
    try:
        # Set up the server socket for video streaming
        video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        video_socket.bind(('192.168.1.100', 5000))
        video_socket.listen(1)

        # Set up the server socket for commands
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        command_socket.bind(('192.168.1.100', 5001))
        command_socket.listen(1)

        wait_for_connections(video_socket, command_socket)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up
        if 'video_socket' in locals():
            video_socket.close()
        if 'command_socket' in locals():
            command_socket.close()

if __name__ == "__main__":
    main()

