import React from 'react';
import { Waves, Cpu, Database } from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  targetFloatId: string;
  predictedCycle: number;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  targetFloatId,
  predictedCycle,
}) => {
  const tabs = [
    { id: 'forecast', label: 'Profile Forecast & UQ' },
    { id: 'chat', label: 'Ask ArgoCast' },
    { id: 'trajectory', label: 'Sea Map' },
    { id: 'xai', label: 'XAI & TEOS-10 Physics' },
    { id: 'evaluation', label: 'Model Benchmarks' },
  ];

  return (
    <header className="border-b border-slate-200 bg-slate-900 text-white sticky top-0 z-30 shadow-md">
      {/* Top Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-cyan-600/30 border border-cyan-400/40 flex items-center justify-center text-cyan-400">
              <Waves className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                  ArgoCast
                </h1>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Explainable, Evidence-Linked Forecasting for Argo Ocean Temperature & Salinity Profiles
              </p>
            </div>
          </div>

          <div className="flex items-center flex-wrap gap-2 text-xs text-slate-300">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span>Target: <strong>WMO {targetFloatId}</strong> (Cyc {predictedCycle})</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span>Domain: <strong>Arabian Sea</strong></span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 border border-slate-700">
              <Cpu className="w-3.5 h-3.5 text-purple-400" />
              <span>UQ: <strong>MC Dropout (50 iters)</strong></span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 sm:space-x-2 mt-3 pt-2 border-t border-slate-800 overflow-x-auto no-scrollbar text-sm">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 rounded-md font-medium whitespace-nowrap transition-colors flex items-center gap-1.5 ${
                activeTab === tab.id
                  ? 'bg-cyan-500 text-slate-950 font-semibold shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/80'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
};
