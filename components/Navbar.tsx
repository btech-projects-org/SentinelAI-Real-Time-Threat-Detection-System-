
import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const location = useLocation();

  const navLinks = [
    { path: '/dashboard', label: 'Monitor', icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
    { path: '/alerts', label: 'Threats', icon: 'M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9' },
    { path: '/admin', label: 'Biometrics', icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z' },
  ];

  return (
    <nav className="glass sticky top-0 z-50 border-b border-white/5">
      <div className="container mx-auto px-6 h-18 flex items-center justify-between">
        <div className="flex items-center space-x-3 group cursor-pointer">
          <div className="relative">
            <div className="absolute inset-0 bg-red-600 blur-lg opacity-20 group-hover:opacity-40 transition-opacity"></div>
            <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-red-700 rounded-xl flex items-center justify-center shadow-2xl relative border border-white/10">
              <span className="font-black text-white text-xl">S</span>
            </div>
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-black tracking-tighter text-white leading-none">SENTINEL<span className="text-red-500 italic">AI</span></span>
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-[0.2em]">Command Center</span>
          </div>
        </div>
        
        <div className="flex space-x-2 bg-black/20 p-1 rounded-2xl border border-white/5">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`relative flex items-center px-4 py-2.5 rounded-xl text-sm font-bold transition-all ${
                  isActive
                    ? 'text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {isActive && (
                  <div className="absolute inset-0 bg-white/5 rounded-xl border border-white/10 shadow-inner"></div>
                )}
                <svg className={`w-4 h-4 mr-2 ${isActive ? 'text-red-500' : 'text-slate-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d={link.icon} />
                </svg>
                <span className="relative">{link.label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;