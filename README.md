
# Prerequisite
- Python version 3.9 - 3.10
- Install Mosquitto Broker on local computer

# Installation
### 1. Clone the Repository
`https://github.com/aloysiustayy/AssistEdge.git`

### 2. Set Up Virtual Environment
Create and activate a virtual environment in the root folder on Raspberry Pi and local computer:\
`python -m venv .venv`\
`source .venv/bin/activate`   
`# On Windows, use .venv\Scripts\activate`

### 3. Install Dependencies
#### 3.1 On Raspberry Pi
After activating the virtual environment, do the following:\
`cd AssistEdge/SignLanguage`\
`pip install -r requirements.txt`\
`cd ../EmotionDetection`\
`pip install -r requirements.txt`

#### 3.2 On Local computer
After activating the virtual environment, do the following:\
`cd AssistEdge/WebMQTT/flask_app`\
`pip install -r requirements.txt`\
`cd ../assistedge-frontend`\
`npm install`

### 4. Run the everything
#### 4.1 On Raspberry Pi
Start two separate terminal inside the two folders: SignLanguage, EmotionDetection, then:\
`python video_streaming.py`\
`python infer.py`

#### 4.2 On Local computer
- Start Mosquitto Broker on root directory.\
    `mosquitto -c mosquitto.conf`

- Start two separate terminal inside WebMQTT folder\
        1. Backend (Flask)\
        `cd flask_app`\
        `python api.py`\
        2. Frontend (React)\
        `cd assistedge-frontend`\
        `npm start`