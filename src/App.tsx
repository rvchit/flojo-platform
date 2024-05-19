import React, { useState } from 'react';
import './App.css';

function App() {
  const [flowboardName, setFlowboardName] = useState('');

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFlowboardName(event.target.value);
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="left-section">
          <button className="header-button">Athletes</button>
        </div>
        <div className="center-section">
          <button className="header-button">Flowboard</button>
        </div>
        <div className="right-section">
          <button className="header-button">Sign In</button>
        </div>
      </header>
      <main className="App-main">
        <div className="overlay-text">
          <h1>Help your community unlock their potential through exclusive content</h1>
        </div>
        <div className="input-container">
          <p>Create a flowboard:</p>
          <div className="input-group">
            <span>flowboard.me/</span>
            <input
              type="text"
              value={flowboardName}
              onChange={handleInputChange}
              placeholder="your-custom-link"
            />
          </div>
        </div>
        <div className="suggestions">
          <h1 className="create-button">Have suggestions or concerns email me @rbisht</h1>  
        </div>
      </main>
    </div>
  );
}

export default App;
