
import React from 'react';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastProvider } from './components/ToastContainer';
import Dashboard from './pages/Dashboard/Dashboard';
import Alerts from './pages/Alerts/Alerts';
import Admin from './pages/Admin/Admin';

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <HashRouter>
          <div className="min-h-screen flex flex-col">
            <Navbar />
            <main className="flex-1 container mx-auto p-4 md:p-6">
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/alerts" element={<Alerts />} />
                <Route path="/admin" element={<Admin />} />
              </Routes>
            </main>
          </div>
        </HashRouter>
      </ToastProvider>
    </ErrorBoundary>
  );
};

export default App;
