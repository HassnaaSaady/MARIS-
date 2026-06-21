import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Reset browser defaults
const style = document.createElement("style");
style.textContent = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0D1117; color: #E6EDF3; overflow: hidden; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #161B22; }
  ::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
`;
document.head.appendChild(style);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
