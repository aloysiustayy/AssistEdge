
# Introduction & Problem Statement
AssistEdge is an edge computing-based solution designed to bridge communication gaps for individuals who rely on sign language, as well as to improve accessibility for the wider community. By leveraging real-time computer vision and audio processing techniques on affordable hardware, AssistEdge translates sign language gestures into text and concurrently converts spoken language into text and audio output. In addition, the system incorporates facial emotion recognition to display audience emotions, further enhancing interactive communication. 

Every day, individuals who rely on sign language or are visually impaired face significant communication challenges. This technology transforms communication for those who face challenges in environments where spoken language predominates, enhancing interactions by converting between visual, audio, and text modalities.

# Methodology and Proposed Solutions
AssistEdge aims to offer real-time sign language translation, speech-to-text and text-to-speech conversion, and facial emotion recognition.

1. Hardware Setup: Two Raspberry Pi devices are used, each equipped with a webcam for data collection. One of the Raspberry Pis is also connected to a speaker for audio output. A laptop runs the Flask server for processing.

2. Communication Protocol: An MQTT broker enables real-time, lightweight messaging between Raspberry Pi devices and the web server, while POST requests are used to send larger data payloads, such as processed video or transcriptions, for storage or further processing.

3. Data Collection and Processing: Each Raspberry Pi manages its own video and audio tasks. The webcams capture video data, while the Raspberry Pi connected to the speaker handles audio output. These devices process their respective data locally.

4. Data Transmission: For sign language translation and speech/ text conversion, MQTT is used to publish or subscribe to relevant topics. For facial emotion recognition, POST request is used to send data to the Flask server.

5. Web Interface: A web interface built using React will display the translation, conversion, and emotion recognition results in real time. There is also a dashboard detailing the usage, xx, and xx. 

## Architecture Diagram
![architecture](./Images/architecture.jpg)

# Installation
## Prerequisite
- Python version 3.9 - 3.10
- Install Mosquitto Broker on local computer
- Require 3 Raspberry Pi(s)

### 1. Clone the Repository
`https://github.com/aloysiustayy/AssistEdge.git`

### 2. Set Up Virtual Environment
Create and activate a virtual environment in the root folder on Raspberry Pi and local computer:\
`python -m venv .venv`\
`source .venv/bin/activate`   
`# On Windows, use .venv\Scripts\activate`

### 3. Install Dependencies
#### 3.1 On Raspberry Pis
After activating the virtual environment, do the following:\
`cd AssistEdge/SignLanguage`\
`pip install -r requirements.txt`\
`cd ../EmotionDetection`\
`pip install -r requirements.txt`\
`cd ../AudioDetection`\
`pip install -r requirements.txt`

#### 3.2 On Local computer
After activating the virtual environment, do the following:\
`cd AssistEdge/WebMQTT/flask_app`\
`pip install -r requirements.txt`\
`cd ../assistedge-frontend`\
`npm install --legacy-peer-deps`

### 4. Run the everything
#### 4.1 On Raspberry Pi 1 - For Sign Language
a. Start terminal inside the folders: `SignLanguage` and install packages by using the commands:\
`sudo apt-get install epeak-ng`\
`sudo apt install nmap`

b. Run the python script: \
`python infer.py --headless --ip <mqtt_broker_ip_address>`

Parameters:\
`--headless`: Disable GUI display\
`--ip`: MQTT Broker IP address

#### 4.2 On Raspberry Pi 2 and 3
Start another two termnials inside the `AudioDetection` and `EmotionDetection` folders and then:\
`python stt.py` - for AudioDetection\
`python video_streaming.py --headless --check <number> --fps <number> --ip <flask server ip address>` - for EmotionDetection

Parameters:\
`--headless`: Disable GUI display\
`--check`: Run DeepFace analysis every N frames (default: every frame)\
`--fps`: Set the target FPS for stable performance (default: 10)\
`--ip`: Flask Server IP address

#### 4.3 On Local computer
- Start Mosquitto Broker on root directory.\
    `mosquitto -c mosquitto.conf`

- Start two separate terminal inside WebMQTT folder\
        1. Backend (Flask)\
        `cd flask_app`\
        `python api.py`\
        2. Frontend (React)\
        `cd assistedge-frontend`\
        `npm start`

### 5. Results
Go to `http://<localhost-ip>:3000` and view the output