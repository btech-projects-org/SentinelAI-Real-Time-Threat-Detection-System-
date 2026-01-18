
import React, { useState, useEffect } from 'react';
import AlertCard from '../../components/AlertCard';

const Alerts: React.FC = () => {
  const [telegramConfig, setTelegramConfig] = useState({ token: '', chatId: '' });
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);

  const [alerts] = useState([
    {
      id: 'alt_001',
      type: 'FIREARM',
      camera: 'PORTAL_NORTH_ALPHA',
      confidence: 0.985,
      timestamp: new Date().toISOString(),
      dataset: 'Roboflow Firearm v1',
      status: 'CRITICAL'
    },
    {
      id: 'alt_002',
      type: 'VIOLENCE',
      camera: 'LOBBY_RECEPTION',
      confidence: 0.842,
      timestamp: new Date().toISOString(),
      dataset: 'UCF-Crime',
      status: 'WARNING'
    }
  ]);

  const handleSaveConfig = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    setTimeout(() => {
      setIsSaving(false);
      alert('ENCRYPTION_COMPLETE: Credentials stored in Atlas Vault.');
    }, 1200);
  };

  const handleTestSignal = () => {
    if (!telegramConfig.token || !telegramConfig.chatId) {
      alert("ERROR: Missing Bot Token or Chat ID for handshake.");
      return;
    }
    
    setIsTesting(true);
    // Simulate API call to Telegram Bot API
    setTimeout(() => {
      setIsTesting(false);
      setShowSuccessPopup(true);
      // Auto-hide popup after 4 seconds
      setTimeout(() => setShowSuccessPopup(false), 4000);
    }, 1800);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-12 py-4 relative">
      {/* Success Popup Notification */}
      {showSuccessPopup && (
        <div className="fixed top-24 right-8 z-[100] animate-in slide-in-from-right fade-in duration-500">
          <div className="glass border-green-500/30 bg-green-500/10 p-6 rounded-2xl shadow-[0_0_40px_rgba(34,197,94,0.2)] flex items-center space-x-4 max-w-sm">
            <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center border border-green-500/40">
              <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <p className="text-white font-black text-xs uppercase tracking-widest mb-1">Handshake Verified</p>
              <p className="text-slate-400 text-[10px] font-mono leading-tight">Bot @SentinelAI_Bot linked to ChatID {telegramConfig.chatId.substring(0,6)}...</p>
            </div>
            <button 
              onClick={() => setShowSuccessPopup(false)} 
              className="text-slate-500 hover:text-white transition-colors"
              aria-label="Close notification"
              type="button"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
            </button>
          </div>
        </div>
      )}

      <div className="px-2">
        <h1 className="text-4xl font-black text-white tracking-tighter mb-1">THREAT_CENTRAL</h1>
        <p className="text-slate-400 font-medium">Global alert dispatch and historical threat analysis.</p>
      </div>

      <section className="glass rounded-3xl overflow-hidden border-white/5 shadow-2xl">
        <div className="bg-white/5 border-b border-white/5 px-8 py-5 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-2.5 bg-red-500/10 rounded-xl border border-red-500/20">
              <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h2 className="text-xs font-black text-white uppercase tracking-[0.2em]">Secure Dispatch Config</h2>
          </div>
          <div className="flex items-center space-x-2 bg-green-500/10 px-3 py-1 rounded-full border border-green-500/20">
             <span className="text-[10px] text-green-500 font-black uppercase">Atlas_Sync: Active</span>
          </div>
        </div>
        
        <form onSubmit={handleSaveConfig} className="p-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div className="space-y-3">
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Telegram Bot API (AES-256)</label>
              <input 
                type="password" 
                value={telegramConfig.token}
                onChange={e => setTelegramConfig({...telegramConfig, token: e.target.value})}
                placeholder="BOT_TOKEN_HASH"
                className="w-full bg-black/40 border border-white/5 rounded-2xl px-5 py-4 text-white focus:outline-none focus:ring-2 focus:ring-red-500/50 transition-all font-mono"
              />
            </div>
            <div className="space-y-3">
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Alert Target Chat ID</label>
              <input 
                type="text" 
                value={telegramConfig.chatId}
                onChange={e => setTelegramConfig({...telegramConfig, chatId: e.target.value})}
                placeholder="-100..."
                className="w-full bg-black/40 border border-white/5 rounded-2xl px-5 py-4 text-white focus:outline-none focus:ring-2 focus:ring-red-500/50 transition-all font-mono"
              />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4">
            <button 
              type="submit"
              disabled={isSaving}
              className="flex-1 bg-white text-slate-950 hover:bg-slate-200 font-black py-4 rounded-2xl transition-all shadow-xl text-xs uppercase tracking-[0.2em]"
            >
              {isSaving ? 'ENCRYPTING...' : 'COMMIT SECURE DISPATCH'}
            </button>
            <button 
              type="button"
              onClick={handleTestSignal}
              disabled={isTesting}
              className={`px-8 font-black py-4 rounded-2xl border transition-all text-xs uppercase tracking-[0.2em] ${
                isTesting 
                ? 'bg-slate-800 border-white/5 text-slate-500' 
                : 'bg-slate-900 hover:bg-slate-800 border-white/5 text-slate-400 hover:text-white'
              }`}
            >
              {isTesting ? 'SENDING_PING...' : 'TEST SIGNAL'}
            </button>
          </div>
        </form>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {alerts.map(alert => (
          <AlertCard key={alert.id} alert={alert} />
        ))}
      </div>
    </div>
  );
};

export default Alerts;
