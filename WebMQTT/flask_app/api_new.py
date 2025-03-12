from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import base64
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to hold the latest frame and emotion counts
latest_frame = None
latest_emotion_counts = {}

# Endpoint to receive frames from the Raspberry Pi client
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame, latest_emotion_counts
    try:
        data = request.json
        frame_base64 = data.get("frame", "")
        latest_emotion_counts = data.get("emotion_counts", {})

        if frame_base64:
            # Decode the image for optional processing/storage
            img_bytes = base64.b64decode(frame_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            latest_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Broadcast the frame and emotion data to all connected SocketIO clients
            socketio.emit('new_frame', {'frame': frame_base64, 'emotion_counts': latest_emotion_counts})
        return "Frame received", 200
    except Exception as e:
        return f"Error: {e}", 500

# Optional endpoints (if still needed for non real-time purposes)
@app.route('/emotion_counts')
def get_emotion_counts():
    return jsonify(latest_emotion_counts), 200

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('frame')
def handle_frame(data):
    socketio.emit('new_frame', data)


@socketio.on('connect')
def handle_connect():
    print('Client connected.')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected.')

if __name__ == '__main__':
    # Use eventlet or gevent if available for production use
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
