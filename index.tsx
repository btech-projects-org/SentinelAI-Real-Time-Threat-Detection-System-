
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import logger from './utils/logger';

// Global error handlers - CRITICAL for zero-error policy
window.addEventListener('error', (event) => {
  logger.error('Global error:', event.error || event.message);
  // Prevent default browser error display in production
  if (import.meta.env.MODE === 'production') {
    event.preventDefault();
  }
});

window.addEventListener('unhandledrejection', (event) => {
  logger.error('Unhandled promise rejection:', event.reason);
  // Prevent default browser warning
  if (import.meta.env.MODE === 'production') {
    event.preventDefault();
  }
});

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("CRITICAL: Root element not found in DOM");
}

try {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} catch (renderError) {
  logger.error('CRITICAL: React render failed:', renderError);
  // Display fallback UI
  rootElement.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: center; min-height: 100vh; background: #020617; color: white; font-family: sans-serif; padding: 2rem;">
      <div style="max-width: 600px; text-align: center;">
        <h1 style="color: #ef4444; font-size: 2rem; margin-bottom: 1rem;">Application Failed to Start</h1>
        <p style="color: #94a3b8; margin-bottom: 2rem;">A critical error prevented the application from loading. Please refresh the page or contact support.</p>
        <button onclick="location.reload()" style="background: #ef4444; color: white; padding: 1rem 2rem; border: none; border-radius: 0.5rem; font-weight: bold; cursor: pointer;">
          RELOAD APPLICATION
        </button>
      </div>
    </div>
  `;
}
