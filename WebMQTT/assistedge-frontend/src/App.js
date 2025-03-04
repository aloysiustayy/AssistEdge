import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import SignLanguagePage from "./components/SignLanguage";
import VideoFeed from "./components/VideoFeed";

function App() {
  return (
    <Router>
      <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
        <h1>Assistive Edge Dashboard</h1>
        <nav>
          <ul>
            <li><Link to="/sign">Sign Language</Link></li>
            <li><Link to="/video">Video Feed</Link></li>
          </ul>
        </nav>

        <Routes>
          <Route path="/sign" element={<SignLanguagePage />} />
          <Route path="/video" element={<VideoFeed />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
