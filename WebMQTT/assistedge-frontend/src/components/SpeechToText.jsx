import React, { useEffect, useState } from "react";
import socketIOClient from "socket.io-client";

const SOCKET_SERVER_URL = "http://localhost:5001"; // Adjust if needed

const SpeechToText = () => {
  const [frame, setFrame] = useState(null);
  const [emotionCounts, setEmotionCounts] = useState({});

  useEffect(() => {
    const socket = socketIOClient(SOCKET_SERVER_URL);
    socket.on("new_frame", (data) => {
      setFrame(data.frame);
      // console.log(data)
      console.log(emotionCounts)
      setEmotionCounts(data.all_emotion || {});

    });
    return () => socket.disconnect();
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Speech To Text</h1>
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          alt="Live Video Stream"
          style={{ width: "640px", borderRadius: "10px", border: "2px solid black" }}
        />
      ) : (
        <p>Waiting for video stream...</p>
      )}
     
    </div>
  );
};

export default SpeechToText;
