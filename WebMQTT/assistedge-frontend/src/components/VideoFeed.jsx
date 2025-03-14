import React, { useEffect, useState } from "react";
import socketIOClient from "socket.io-client";

const SOCKET_SERVER_URL = "http://localhost:5001"; // Adjust if needed

const VideoFeed = () => {
  const [frame, setFrame] = useState(null);
  const [emotionCounts, setEmotionCounts] = useState({});

  useEffect(() => {
    const socket = socketIOClient(SOCKET_SERVER_URL);
    socket.on("new_frame", (data) => {
      setFrame(data.frame);
      setEmotionCounts(data.emotion_counts || {});
    });
    return () => socket.disconnect();
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Real-time Emotion Detection</h1>
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          alt="Live Video Stream"
          style={{ width: "640px", borderRadius: "10px", border: "2px solid black" }}
        />
      ) : (
        <p>Waiting for video stream...</p>
      )}
      <h2>People's Emotions:</h2>
      <ul style={{ listStyle: "none", fontSize: "20px" }}>
        {Object.entries(emotionCounts).map(([emotion, count]) => (
          <li key={emotion}>
            {emotion.charAt(0).toUpperCase() + emotion.slice(1)}: {count}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default VideoFeed;
