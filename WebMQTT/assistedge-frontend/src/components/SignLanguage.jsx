import React, { useEffect, useState } from "react";

const SignLanguagePage = () => {
  const [messages, setMessages] = useState([]);

  useEffect(() => {

    const fetchMessages = async () => {
      try {
        const response = await fetch("http://localhost:5001/data");
        if (response.ok) {
          const data = await response.json();
          setMessages(data.sign_language || []);
        } else {
          console.error("Failed to fetch messages. Status:", response.status);
        }
      } catch (error) {
        console.error("Error fetching messages:", error);
      }
    };

    // fetchMessages();
    const intervalId = setInterval(fetchMessages, 1000);

    return () => clearInterval(intervalId);
  }, []);

  const clearMessages = async () => {
    try {
      const response = await fetch("http://localhost:5001/clear", {
        method: "POST",
      });
      if (response.ok) {
        setMessages([]); // Also clear client state
      } else {
        console.error("Failed to clear messages. Status:", response.status);
      }
    } catch (error) {
      console.error("Error clearing messages:", error);
    }
  };

  return (
    <div>
      <h2>Sign Language to Text</h2>
      <div
        style={{
          border: "1px solid #ccc",
          height: "300px",
          overflowY: "scroll",
          padding: "10px",
          backgroundColor: "#f9f9f9",
        }}
      >
        {messages.map((msg, index) => (
          <p key={index} style={{ margin: "5px 0" }}>
            {msg}
          </p>
        ))}
      </div>
      <button onClick={clearMessages}>Clear Messages</button>
    </div>
  );
};

export default SignLanguagePage;
