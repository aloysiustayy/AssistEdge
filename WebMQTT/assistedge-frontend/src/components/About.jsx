import React from "react";

const AboutPage = () => {
  return (
    <div 
      style={{ 
        maxWidth: "800px", 
        margin: "auto", 
        padding: "40px", 
        textAlign: "center", 
        fontFamily: "Arial, sans-serif",
        height: "100vh", 
        overflowY: "auto" // Enables scrolling
      }}
    >
      <h1 style={{ marginBottom: "20px", color: "#333" }}>About AssistEdge</h1>

      <section style={{ marginBottom: "40px" }}>
        <h2 style={{ color: "#444" }}>Our Mission and Vision</h2>
        <p style={{ fontSize: "17px", lineHeight: "1.8", color: "#666" }}>
          Our mission is to create a more inclusive world by providing tools that empower individuals
          with disabilities to communicate more effectively.
        </p>
        <p style={{ fontSize: "17px", lineHeight: "1.8", color: "#666" }}>
          We aim to bridge the gap between sign language and digital communication by making recognition 
          systems more accessible and intuitive.
        </p>
      </section>

      <section style={{ marginBottom: "40px" }}>
        <h2 style={{ color: "#444" }}>Our Features</h2>

        <div style={{ textAlign: "left", margin: "auto", maxWidth: "600px" }}>
          <h3 style={{ color: "#555", marginBottom: "5px" }}>🔠 Sign Language to Text</h3>
          <p style={{ fontSize: "16px", lineHeight: "1.6", color: "#666" }}>
            Converts sign language to text in real-time, enhancing communication for individuals who are deaf or hard of hearing.
          </p>

          <h3 style={{ color: "#555", marginBottom: "5px" }}>🎤 Speech to Text</h3>
          <p style={{ fontSize: "16px", lineHeight: "1.6", color: "#666" }}>
            Converts spoken language to text instantly, making conversations more accessible.
          </p>

          <h3 style={{ color: "#555", marginBottom: "5px" }}>😊 Emotion Recogniser</h3>
          <p style={{ fontSize: "16px", lineHeight: "1.6", color: "#666" }}>
            Detects emotions from a video feed, providing real-time emotional insights.
          </p>

          <h3 style={{ color: "#555", marginBottom: "5px" }}>📊 Dashboard</h3>
          <p style={{ fontSize: "16px", lineHeight: "1.6", color: "#666" }}>
            Displays insights and analytics from the collected data, making trends easy to understand.
          </p>
        </div>
      </section>

      <section style={{ marginBottom: "20px" }}>
        <h2 style={{ color: "#444" }}>Contact Us</h2>
        <p style={{ fontSize: "16px", color: "#666" }}>📧 Email: <a href="mailto:admin@assistedge.com" style={{ color: "#007BFF", textDecoration: "none" }}>admin@assistedge.com</a></p>
      </section>

    </div>
  );
};

export default AboutPage;
