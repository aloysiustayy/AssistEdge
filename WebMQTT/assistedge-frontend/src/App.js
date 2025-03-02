// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

import React, { useEffect, useState } from 'react';

function App() {
  const [messages, setMessages] = useState([]);

  // Poll the Flask API every second to fetch the latest sign language texts
  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await fetch('http://localhost:5001/data');
        if (response.ok) {
          // Assuming the JSON response has a structure like: { sign_language: ["hello", "thanks", ...] }
          const data = await response.json();
          console.log(data)
          setMessages(data.sign_language || []);
        } else {
          console.error('Failed to fetch messages. Status:', response.status);
        }
      } catch (error) {
        console.error('Error fetching messages:', error);
      }
    };

    fetchMessages(); // initial fetch
    const intervalId = setInterval(fetchMessages, 1000); // fetch every 1 second

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Sign Language To Text</h1>
      <div 
        style={{
          border: '1px solid #ccc',
          height: '300px',
          overflowY: 'scroll',
          padding: '10px',
          backgroundColor: '#f9f9f9'
        }}
      >
        
        {messages.map((msg, index) => (
          <p key={index} style={{ margin: '5px 0' }}>
            {msg}
          </p>
        ))}
      </div>
    </div>
  );
}

export default App;
