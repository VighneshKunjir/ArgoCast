import React from 'react';
import { Play, Compass, Calendar, Layers, RefreshCw } from 'lucide-react';

interface FloatItem {
  wmo_id: string;
  name: string;
  baseLat: number;
  baseLon: number;
  cycles: number[];
  defaultCycle: number;
  firstDate?: string;
  lastDate?: string;
}

interface ControlBarProps {
  floats: FloatItem[];
  selectedWmo: string;
  onSelectWmo: (wmo: string) => void;
  selectedCycle: number;
  onSelectCycle: (cycle: number) => void;
  variableMode: 'temperature' | 'salinity' | 'both';
  setVariableMode: (mode: 'temperature' | 'salinity' | 'both') => void;
  isLoading: boolean;
  onRefreshForecast: () => void;
  currentLat?: number;
  currentLon?: number;
  forecastDate?: string;
  lastRefresh?: string | null;
  refreshCount?: number;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  floats,
  selectedWmo,
  onSelectWmo,
  selectedCycle,
  onSelectCycle,
  variableMode,
  setVariableMode,
  isLoading,
  onRefreshForecast,
  currentLat,
  currentLon,
  forecastDate,
  lastRefresh,
  refreshCount
}) => {
  const currentFloat = floats.find((f) => f.wmo_id === selectedWmo);
  const allCycles = currentFloat?.cycles || [];
  // The selected base cycle is t-1; the model also needs the two profiles before it
  // (t-3, t-2), so the first two recorded cycles of a float cannot be a base cycle.
  const availableCycles = allCycles.length > 2 ? allCycles.slice(2) : allCycles;

  return (
    <div className="bg-white border-b border-slate-200 px-4 sm:px-6 py-3 shadow-xs">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Float and Cycle Selection */}
        <div className="flex flex-wrap items-center gap-3">
          <div>
            <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Select Argo Float {floats.length > 0 ? `(${floats.length} available)` : '(Arabian Sea)'}
            </label>
            <select
              value={selectedWmo}
              onChange={(e) => onSelectWmo(e.target.value)}
              className="bg-slate-50 border border-slate-300 text-slate-900 text-xs rounded-lg focus:ring-cyan-500 focus:border-cyan-500 block w-64 p-2 font-medium"
            >
              {floats.map((f) => (
                <option key={f.wmo_id} value={f.wmo_id}>
                  WMO {f.wmo_id}{f.firstDate && f.lastDate ? ` · ${f.firstDate.slice(0, 4)}-${f.lastDate.slice(0, 4)}` : ''} · {f.cycles.length} profiles
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Base Cycle (Horizon: Next +1)
            </label>
            <div className="flex items-center space-x-1">
              <select
                value={selectedCycle}
                onChange={(e) => onSelectCycle(Number(e.target.value))}
                className="bg-slate-50 border border-slate-300 text-slate-900 text-xs rounded-lg focus:ring-cyan-500 focus:border-cyan-500 p-2 font-medium"
              >
                {availableCycles.map((c) => {
                  const pos = allCycles.indexOf(c);
                  const next = pos + 1 < allCycles.length ? allCycles[pos + 1] : null;
                  return (
                    <option key={c} value={c}>
                      Cycle {c} → {next !== null ? `Cyc ${next}` : 'next (not observed yet)'}
                    </option>
                  );
                })}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Profile Variable
            </label>
            <div className="inline-flex rounded-lg shadow-xs border border-slate-300 bg-slate-50 p-0.5 text-xs">
              <button
                type="button"
                onClick={() => setVariableMode('temperature')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  variableMode === 'temperature'
                    ? 'bg-rose-500 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Temp (°C)
              </button>
              <button
                type="button"
                onClick={() => setVariableMode('salinity')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  variableMode === 'salinity'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Salinity (PSU)
              </button>
              <button
                type="button"
                onClick={() => setVariableMode('both')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  variableMode === 'both'
                    ? 'bg-cyan-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Dual View
              </button>
            </div>
          </div>
        </div>

        {/* Location coordinates & Action button */}
        <div className="flex items-center justify-between lg:justify-end gap-3 pt-2 lg:pt-0 border-t lg:border-t-0 border-slate-100">
          <div className="flex items-center space-x-3 text-xs text-slate-500">
            {currentLat != null && currentLon != null && typeof currentLat === 'number' && typeof currentLon === 'number' && (
              <div className="flex items-center gap-1">
                <Compass className="w-3.5 h-3.5 text-cyan-600" />
                <span>{currentLat.toFixed(2)}°N, {currentLon.toFixed(2)}°E</span>
              </div>
            )}
            {forecastDate && (
              <div className="hidden sm:flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                <span>{forecastDate}</span>
              </div>
            )}
            {lastRefresh && !isLoading && (
              <div className="hidden md:flex items-center gap-1 text-emerald-600 font-medium" title={`Forecast recomputed ${refreshCount ?? 1} time(s) this session`}>
                <span>Updated {lastRefresh}</span>
              </div>
            )}
          </div>

          <button
            onClick={onRefreshForecast}
            disabled={isLoading}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-lg shadow-xs transition-colors disabled:opacity-50"
          >
            {isLoading ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />
            ) : (
              <Play className="w-3.5 h-3.5 text-cyan-400 fill-cyan-400" />
            )}
            <span>{isLoading ? 'Computing...' : 'Run Forecast'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
