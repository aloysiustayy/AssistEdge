import React, { useEffect, useState } from "react";

const VideoFeed = () => {
  const [emotionCounts, setEmotionCounts] = useState({});

  useEffect(() => {
    const fetchEmotionCounts = async () => {
      try {
        const response = await fetch("http://localhost:5001/emotion_counts");
        const data = await response.json();
        setEmotionCounts(data);
      } catch (error) {
        console.error("Error fetching emotion counts:", error);
      }
    };

    const interval = setInterval(fetchEmotionCounts, 1000); // Fetch emotion counts every second
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Real-time Emotion Detection</h1>

      {/* Display emotion counts */}
      <h2>People's Emotions:</h2>
      <ul style={{ listStyle: "none", fontSize: "20px" }}>
        {Object.entries(emotionCounts).map(([emotion, count]) => (
          <li key={emotion}>
            {emotion.charAt(0).toUpperCase() + emotion.slice(1)}: {count}
          </li>
        ))}
      </ul>

      {/* Video Stream */}
      <img
        src="http://localhost:5001/video_feed"
        alt="Live Video Stream"
        style={{ width: "640px", borderRadius: "10px", border: "2px solid black" }}
      />
    </div>
  );
};

export default VideoFeed;
