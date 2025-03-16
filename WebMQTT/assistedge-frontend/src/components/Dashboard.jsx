import React, { useEffect, useState } from "react";

const Dashboard = () => {
  const [all_data, setAllData] = useState({}); // Default to empty object

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await fetch("http://localhost:5001/dashboard");
        if (response.ok) {
          const data = await response.json();
          setAllData(data || {}); // Ensure it's an object
          console.log(data);
        } else {
          console.error("Failed to fetch messages. Status:", response.status);
        }
      } catch (error) {
        console.error("Error fetching messages:", error);
      }
    };

    // Fetch data every 1 second
    const intervalId = setInterval(fetchMessages, 1000);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Dashboard</h1>
      <ul style={{ listStyle: "none", fontSize: "20px" }}>
        {Object.entries(all_data).map(([timestamp, data]) => (
          <li key={timestamp}>
            <strong>{timestamp}</strong>
            <br />
            Recognised Word: {data.recognised_word || "N/A"}
            <br />
            Recognised Emotion: {data.recognised_emotion 
              ? Object.entries(data.recognised_emotion)
                .map(([emotion, count]) => `${emotion} (${count})`)
                .join(", ") 
              : "N/A"}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Dashboard;
