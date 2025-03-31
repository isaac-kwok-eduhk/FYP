from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import base64
import aiohttp
import asyncio
import numpy as np
import os
from datetime import datetime
from PIL import Image
import json
import pickle
import socket
import struct
import pygame
import threading

app = Flask(__name__)
CORS(app)

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
camera = cv2.VideoCapture(0)
is_recording = False
video_writer = None
session_folder = None 

def generate_frames(video_socket):
    global is_recording, video_writer
    data = b""
    payload_size = struct.calcsize("Q")

    while True:
        while len(data) < payload_size:
            packet = video_socket.recv(4 * 1024)
            if not packet:
                break
            data += packet

        if len(data) < payload_size:
            continue

        packed_size = data[:payload_size]
        data = data[payload_size:]
        frame_size = struct.unpack("Q", packed_size)[0]

        while len(data) < frame_size:
            packet = video_socket.recv(4 * 1024)
            if not packet:
                break
            data += packet

        if len(data) < frame_size:
            continue

        frame_data = data[:frame_size]
        data = data[frame_size:]
        frame = pickle.loads(frame_data)

        # Check for empty frame
        if frame is None or frame.size == 0:
            print("Empty frame received!")
            continue

        # Write frame to video if recording
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(video_socket), mimetype='multipart/x-mixed-replace; boundary=frame')

def create_session_folder():
    global session_folder
    if session_folder is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(r"C:\Users\kwoki\OneDrive\桌面\fyp_chat - Copy\log", current_time)
        os.makedirs(session_folder, exist_ok=True)

async def fetch_api_response(payload):
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json=payload) as response:
            return await response.json()
        
@app.route('/capture', methods=['POST'])
async def capture():
    global session_folder
    try:
        create_session_folder()
        data = request.json['image']
        print("Received image data:", data)

        img_data = base64.b64decode(data.split(',')[1])
        np_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        jpg_filename = os.path.join(session_folder, f"captured_image_{datetime.now().strftime('%H%M%S')}.jpg")
        pil_image.save(jpg_filename, format='JPEG')

        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "model": "llava-v1.5-7b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": -1,
        }

        response = await fetch_api_response(payload)

        if 'choices' in response and len(response['choices']) > 0:
            message_content = response['choices'][0]['message']['content']
            log_filename = os.path.join(session_folder, f"log.txt")
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{message_content}\n")
            return jsonify(message_content)
        else:
            return jsonify({"error": "Invalid response from API"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/text_response', methods=['POST'])
async def text_response():
    try:
        user_input = request.json['input']
        print("User input received:", user_input)

        payload = {
            "model": "llava-v1.5-7b",
            "messages": [
                {"role": "system", "content": "You are a helpful ocean scientist."},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7,
            "max_tokens": 50,
        }

        response = await fetch_api_response(payload)

        if 'choices' in response and len(response['choices']) > 0:
            message_content = response['choices'][0]['message']['content']
            return jsonify(message_content)
        else:
            return jsonify({"error": "Invalid response from API"}), 500

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, video_writer
    create_session_folder()

    if not is_recording:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        current_time = datetime.now().strftime("%H%M%S")
        video_filename = os.path.join(session_folder, f"recording_{current_time}.avi")
        
        # Ensure frame size matches the incoming video
        frame_width = 640  # Replace with actual width of your frames
        frame_height = 480  # Replace with actual height of your frames
        frame_rate = 20.0  # Adjust to match incoming video frame rate

        video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))
        is_recording = True
        return jsonify({"message": "Recording started."}), 200
    else:
        return jsonify({"message": "Recording is already in progress."}), 400

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, video_writer
    if is_recording:
        is_recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        return jsonify({"message": "Recording stopped."}), 200
    else:
        return jsonify({"message": "No recording in progress."}), 400
    
@app.teardown_appcontext
def shutdown_camera(exception=None):
    camera.release()

@app.route('/')
def home():
    return render_template('index.html')

class PS4Controller:
    """Class representing the PS4 controller."""

    button_names = {
        0: 'X',
        1: 'Circle',
        2: 'Square',
        3: 'Triangle',
        4: 'Share',
        5: 'Home',
        6: 'Options',
        7: 'R3',
        8: 'L3',
        9: 'L1',
        10: 'R1',
        11: 'Up',
        12: 'Down',
        13: 'Left',
        14: 'Right',
        15: 'Touch'
    }

    def __init__(self, command_socket):
        """Initialize the PS4 controller and the command socket."""
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        self.command_socket = command_socket
        self.centered = True  

    def send_command(self, command):
        """Helper function to send a command."""
        self.command_socket.send(command.encode())
        print(f"Command sent: {command}")

    def listen(self):
        """Listen for events from the PS4 controller and send commands."""
        axis_data = {i: 0 for i in range(self.controller.get_numaxes())}
        button_data = {i: False for i in range(self.controller.get_numbuttons())}

        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    axis_data[event.axis] = round(event.value, 2)
                    self.process_joystick_inputs(axis_data)
                elif event.type == pygame.JOYBUTTONDOWN:
                    button_data[event.button] = True
                    button_name = self.button_names.get(event.button, str(event.button))
                    self.send_command(f"Button {button_name} pressed")
                elif event.type == pygame.JOYBUTTONUP:
                    button_data[event.button] = False
                    button_name = self.button_names.get(event.button, str(event.button))
                    self.send_command(f"Button {button_name} released")
            print("sent")
    def process_joystick_inputs(self, axis_data):
        """Process all joystick inputs (left and right) and send commands."""
        # Left joystick axes
        joystick_x = axis_data[0] 
        joystick_y = axis_data[1] 

        # right joystick axes
        additional_joystick_y = axis_data[3]
        # Thresholds
        threshold = 0.5
        center_threshold = 0.2
         # Process right joystick (vertical movement commands)
        if additional_joystick_y is not None:
            if additional_joystick_y <= -threshold:
                self.send_command("U")
            elif additional_joystick_y >= threshold:
                self.send_command("D")

        # Process left joystick (movement commands)
        if abs(joystick_x) < center_threshold and abs(joystick_y) < center_threshold:
            if not self.centered:  # Only send "CENTER" if it wasn't sent previously
                self.send_command("C")
                self.centered = True
            return

        # Joystick is not centered
        self.centered = False

        # Map left joystick inputs to directions
        directions = {
            (joystick_y < -threshold, joystick_x < -threshold): "FL",
            (joystick_y < -threshold, joystick_x > threshold): "FR",
            (joystick_y > threshold, joystick_x < -threshold): "BL",
            (joystick_y > threshold, joystick_x > threshold): "BR",
            (joystick_y < -threshold, abs(joystick_x) <= threshold): "F",
            (joystick_y > threshold, abs(joystick_x) <= threshold): "B",
            (abs(joystick_y) <= threshold, joystick_x < -threshold): "L",
            (abs(joystick_y) <= threshold, joystick_x > threshold): "R",
        }

        for condition, command in directions.items():
            if all(condition):
                self.send_command(command)
                break

def run_ps4_controller(command_socket):
    ps4 = PS4Controller(command_socket)
    ps4.listen()

if __name__ == '__main__':
    video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_socket.connect(('192.168.1.100', 5000))

    command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    command_socket.connect(('192.168.1.100', 5001))

    # Start the PS4 controller in a separate thread
    ps4_thread = threading.Thread(target=run_ps4_controller, args=(command_socket,))
    ps4_thread.start()

    app.run(debug=True, use_reloader=False)