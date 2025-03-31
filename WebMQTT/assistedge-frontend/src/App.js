import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import SignLanguagePage from "./components/SignLanguage";
import VideoFeed from "./components/VideoFeed";
import SpeechToTextPage from "./components/SpeechToText";
import Dashboard from "./components/Dashboard";
import About from "./components/About";
import "./App.css";

function App() {
  return (
    <Router>
      <div className="app-container">
        {/* Navigation Bar */}
        <nav className="navbar">
          <h2 className="logo">AssistEdge</h2>
          <ul className="nav-links">
          <li><Link to="/about">About</Link></li>
            <li><Link to="/sign">Sign Language To Text</Link></li>
            <li><Link to="/speech">Speech To Text</Link></li>
            <li><Link to="/video">Emotion Recogniser</Link></li>
            <li><Link to="/dashboard">Dashboard</Link></li>
          </ul>
        </nav>

        {/* Main Content */}
        <div className="content-container">
          <Routes>
            <Route path="/sign" element={<SignLanguagePage />} />
            <Route path="/video" element={<VideoFeed />} />
            <Route path="/speech" element={<SpeechToTextPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
