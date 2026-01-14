
import React, { useState } from 'react';

const Admin: React.FC = () => {
  const [formData, setFormData] = useState({
    criminalId: '',
    fullName: '',
    category: 'Violent Crimes',
    notes: ''
  });
  const [image, setImage] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastRegistered, setLastRegistered] = useState<any>(null);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!image) {
      alert('Please select an image file');
      return;
    }
    
    if (!formData.criminalId || !formData.fullName) {
      alert('Please fill in all required fields');
      return;
    }

    setLoading(true);
    
    const uploadWithRetry = async (retries = 3): Promise<void> => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
      
      try {
        const formDataToSend = new FormData();
        formDataToSend.append('criminal_id', formData.criminalId);
        formDataToSend.append('name', formData.fullName);
        formDataToSend.append('threat_level', formData.category);
        formDataToSend.append('description', formData.notes);
        formDataToSend.append('image', image);

        const response = await fetch('http://localhost:8000/api/v1/criminals/register', {
          method: 'POST',
          body: formDataToSend,
          signal: controller.signal,
          headers: {
            'Accept': 'application/json',
          }
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          throw new Error(`Invalid response type: ${contentType}`);
        }

        const result = await response.json();

        if (result.status === 'success') {
          setLoading(false);
          setLastRegistered(formData);
          alert(`✅ SUCCESS: Criminal '${formData.fullName}' registered successfully!\n\nID: ${formData.criminalId}\nThreat Level: ${formData.category}`);
          setFormData({ criminalId: '', fullName: '', category: 'Violent Crimes', notes: '' });
          setImage(null);
        } else {
          throw new Error(result.message || 'Registration failed');
        }
      } catch (error) {
        clearTimeout(timeoutId);
        
        // Handle specific error types
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            if (retries > 0) {
              await new Promise(resolve => setTimeout(resolve, 1000));
              return uploadWithRetry(retries - 1);
            }
            throw new Error('Request timeout after 30 seconds');
          }
          
          if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch')) {
            if (retries > 0) {
              await new Promise(resolve => setTimeout(resolve, 2000));
              return uploadWithRetry(retries - 1);
            }
            throw new Error('Network connection failed. Please check backend server.');
          }
        }
        
        throw error;
      }
    };
    
    try {
      await uploadWithRetry();
    } catch (error) {
      setLoading(false);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`❌ ERROR: ${errorMessage}`);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-10 py-4">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 px-2">
        <div>
          <h1 className="text-4xl font-black text-white tracking-tighter mb-1 uppercase">Vault_Access</h1>
          <p className="text-slate-500 font-medium">Index subjects into the Vector Database for recognition logic.</p>
        </div>
        
        {lastRegistered && (
          <div className="animate-pulse">
            <button 
              onClick={() => alert(`To test this target, go to 'Monitor' and use 'Inject Watchlist Hit'. Registered ID: ${lastRegistered.criminalId}`)}
              className="bg-cyan-500/10 border border-cyan-500/50 text-cyan-500 px-6 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest"
            >
              Latest Entry Verified: {lastRegistered.criminalId}
            </button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        <div className="lg:col-span-2">
          <form onSubmit={handleUpload} className="glass p-8 rounded-3xl space-y-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-6 opacity-10">
              <svg className="w-16 h-16 text-white" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" clipRule="evenodd" /></svg>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-3">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] flex items-center">
                  Subject_Identifier
                  <span className="ml-2 text-cyan-500 font-mono text-[8px] bg-cyan-500/10 px-1 rounded border border-cyan-500/20 tracking-normal">VECTOR_ID</span>
                </label>
                <input 
                  type="text" 
                  value={formData.criminalId}
                  onChange={e => setFormData({...formData, criminalId: e.target.value})}
                  className="w-full bg-black/40 border border-white/5 rounded-xl px-5 py-4 text-white focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-all font-mono"
                  placeholder="e.g. ID_X_001"
                  required
                />
              </div>
              <div className="space-y-3">
                <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Full_Identity_Name</label>
                <input 
                  type="text" 
                  value={formData.fullName}
                  onChange={e => setFormData({...formData, fullName: e.target.value})}
                  className="w-full bg-black/40 border border-white/5 rounded-xl px-5 py-4 text-white focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-all"
                  placeholder="e.g. ALEX_REED"
                  required
                />
              </div>
            </div>

            <div className="space-y-3">
              <label htmlFor="threat-classification" className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Criminal_Classification</label>
              <select 
                id="threat-classification"
                aria-label="Criminal threat classification level"
                value={formData.category}
                onChange={e => setFormData({...formData, category: e.target.value})}
                className="w-full bg-black/40 border border-white/5 rounded-xl px-5 py-4 text-white focus:outline-none focus:ring-1 focus:ring-cyan-500 appearance-none"
              >
                <option>Violent Crimes</option>
                <option>Wanted Fugitive</option>
                <option>Larceny / Theft</option>
                <option>Organized Crime</option>
              </select>
            </div>

            <div className="space-y-3">
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Intelligence_Brief</label>
              <textarea 
                rows={3}
                value={formData.notes}
                onChange={e => setFormData({...formData, notes: e.target.value})}
                className="w-full bg-black/40 border border-white/5 rounded-xl px-5 py-4 text-white focus:outline-none focus:ring-1 focus:ring-cyan-500 resize-none"
                placeholder="Last seen coordinates, behavior patterns..."
              ></textarea>
            </div>

            <div className="space-y-3">
              <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Neural_Source_Mapping (Face)</label>
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-white/5 border-dashed rounded-xl cursor-pointer bg-black/20 hover:bg-white/5 transition-all">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <p className="text-xs text-slate-500 font-bold uppercase tracking-widest">{image ? image.name : 'UPLOAD_IMAGE_FOR_VECTORIZING'}</p>
                </div>
                <input type="file" className="hidden" accept="image/*" onChange={e => setImage(e.target.files ? e.target.files[0] : null)} />
              </label>
            </div>

            <button 
              type="submit" 
              disabled={loading}
              className={`w-full py-5 rounded-xl font-black transition-all shadow-2xl tracking-[0.3em] uppercase text-xs ${loading ? 'bg-slate-800 text-slate-500' : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-[0_0_30px_rgba(6,182,212,0.3)] active:scale-[0.98]'}`}
            >
              {loading ? 'PROCESSING_BIOMETRICS...' : 'EXECUTE_DB_COMMIT'}
            </button>
          </form>
        </div>

        <div className="space-y-8">
          <div className="glass p-8 rounded-3xl border-white/5">
             <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] border-b border-white/5 pb-4 mb-6 flex justify-between">
               Database_Status
               <span className="text-green-500">OPTIMAL</span>
             </h3>
             <div className="space-y-4">
                <div className="flex justify-between items-center bg-white/5 p-4 rounded-xl">
                   <span className="text-[9px] font-black text-slate-500 uppercase">Registered_Targets</span>
                   <span className="text-white font-mono font-bold">1,249</span>
                </div>
                <div className="flex justify-between items-center bg-white/5 p-4 rounded-xl">
                   <span className="text-[9px] font-black text-slate-500 uppercase">Search_Latency</span>
                   <span className="text-cyan-500 font-mono font-bold">142ms</span>
                </div>
                <div className="flex justify-between items-center bg-white/5 p-4 rounded-xl">
                   <span className="text-[9px] font-black text-slate-500 uppercase">Storage_Provider</span>
                   <span className="text-white font-mono font-bold">Atlas_C0</span>
                </div>
             </div>
          </div>
          
          <div className="glass p-8 rounded-3xl border-l-4 border-amber-600 bg-amber-600/5">
            <p className="text-[9px] text-amber-600 font-black uppercase tracking-widest leading-relaxed mb-4">
              Biometric Data Compliance Notice
            </p>
            <p className="text-[10px] text-slate-400 font-medium leading-relaxed">
              Indexing identities requires Tier-2 security clearance. All vector hashes are immutable once stored in the secure cluster partition.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Admin;
