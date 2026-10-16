import React, { useState, useEffect } from 'react';
import { Header } from './components/Header.tsx';
import { ControlBar } from './components/ControlBar.tsx';
import { ProfilePlot } from './components/ProfilePlot.tsx';
import { XaiDiagnostics } from './components/XaiDiagnostics.tsx';
import { EvidenceLinkViewer } from './components/EvidenceLinkViewer.tsx';
import { ChatPanel } from './components/ChatPanel.tsx';
import { EvaluationBenchmarks } from './components/EvaluationBenchmarks.tsx';
import { TrajectoryMap } from './components/TrajectoryMap.tsx';
import { ForecastResult, ArgoProfile } from './types.ts';
import { apiService, FloatSummary } from './services/api.ts';
import { RefreshCw, AlertCircle } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('forecast');
  const [floats, setFloats] = useState<FloatSummary[]>([]);
  const [selectedWmo, setSelectedWmo] = useState<string>('');
  const [selectedCycle, setSelectedCycle] = useState<number>(0);
  const [variableMode, setVariableMode] = useState<'temperature' | 'salinity' | 'both'>('both');

  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const [historicalProfiles, setHistoricalProfiles] = useState<ArgoProfile[]>([]);
  const [floatTrack, setFloatTrack] = useState<{ cycle: number; lat: number; lon: number; date: string }[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);
  const [refreshCount, setRefreshCount] = useState<number>(0);

  // Load available floats on initial mount
  useEffect(() => {
    apiService.getFloats()
      .then((data) => {
        setFloats(data);
        if (data.length > 0) {
          // Start on a float with a long record that is still reporting, so the default
          // forecast falls in the held-out period (2019 onwards) of the served model.
          const candidates = data.filter((f) => f.cycles.length >= 30);
          const pool = candidates.length ? candidates : data;
          const start = [...pool].sort((a, b) => (b.lastDate ?? '').localeCompare(a.lastDate ?? ''))[0];
          setSelectedWmo(start.wmo_id);
          setSelectedCycle(start.defaultCycle || start.cycles[start.cycles.length - 2]);
        }
      })
      .catch((err) => {
        console.error('Failed to load float list:', err);
      });
  }, []);

  // Run forecast whenever selectedWmo or selectedCycle changes
  const loadForecast = async (wmo: string, cycle: number) => {
    setIsLoading(true);
    setError(null);
    try {
      // 1. Run forecast
      const forecastData = await apiService.runForecast(wmo, cycle);
      setForecast(forecastData);
      setLastRefresh(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
      setRefreshCount((c) => c + 1);
      if (forecastData.target_cycle && forecastData.target_cycle !== cycle) {
        setSelectedCycle(forecastData.target_cycle);
      }

      // 2. Fetch historical profiles for sequence display
      try {
        const profilesData = await apiService.getProfiles(wmo);
        // Keep profiles prior to or up to cycle
        const prior = profilesData.filter((p: ArgoProfile) => p.cycle_number <= cycle).slice(-3);
        setHistoricalProfiles(prior);
        // Real positions: the four profiles up to the base cycle plus the forecast target if observed.
        const idx = profilesData.findIndex((p: ArgoProfile) => p.cycle_number === cycle);
        const around = idx >= 0 ? profilesData.slice(Math.max(0, idx - 3), idx + 2) : [];
        setFloatTrack(around.map((p: ArgoProfile) => ({ cycle: p.cycle_number, lat: p.latitude, lon: p.longitude, date: p.date })));
      } catch (profilesErr) {
        console.error('Historical profiles retrieval error:', profilesErr);
      }
    } catch (err: any) {
      console.error('Forecast retrieval error:', err);
      setError(err.message || 'Failed to generate forecast');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (selectedWmo && selectedCycle) {
      loadForecast(selectedWmo, selectedCycle);
    }
  }, [selectedWmo, selectedCycle]);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 flex flex-col font-sans">
      {/* App Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        targetFloatId={selectedWmo}
        predictedCycle={forecast ? forecast.predicted_cycle : selectedCycle + 1}
      />

      {/* Control & Query Bar */}
      <ControlBar
        floats={floats}
        selectedWmo={selectedWmo}
        onSelectWmo={(wmo) => {
          setSelectedWmo(wmo);
          const found = floats.find((f) => f.wmo_id === wmo);
          if (found && found.cycles) {
            setSelectedCycle(found.defaultCycle || found.cycles[found.cycles.length - 2]);
          }
        }}
        selectedCycle={selectedCycle}
        onSelectCycle={(c) => setSelectedCycle(c)}
        variableMode={variableMode}
        setVariableMode={setVariableMode}
        isLoading={isLoading}
        onRefreshForecast={() => loadForecast(selectedWmo, selectedCycle)}
        currentLat={forecast?.latitude}
        currentLon={forecast?.longitude}
        forecastDate={forecast?.forecast_date}
        lastRefresh={lastRefresh}
        refreshCount={refreshCount}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6">
        {error && (
          <div className="mb-4 bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-xl flex items-center gap-3 text-xs">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
            <div>
              <strong>Error calculating ocean profile forecast:</strong> {error}
            </div>
          </div>
        )}

        {isLoading && !forecast ? (
          <div className="flex flex-col items-center justify-center py-24 space-y-3">
            <RefreshCw className="w-8 h-8 animate-spin text-cyan-600" />
            <p className="text-xs font-semibold text-slate-500">
              Loading forecast...
            </p>
          </div>
        ) : forecast ? (
          <div>
            {(activeTab === 'forecast' || activeTab === 'evidence') && (
              <ProfilePlot
                forecast={forecast}
                historicalProfiles={historicalProfiles}
                variableMode={variableMode}
              />
            )}

            {activeTab === 'xai' && <XaiDiagnostics forecast={forecast} />}

            {activeTab === 'chat' && (
              <ChatPanel
                forecast={forecast}
                onNavigateToEvidence={() => {
                  setActiveTab('forecast');
                  setTimeout(() => {
                    document.getElementById('evidence-link-protocol-section')?.scrollIntoView({ behavior: 'smooth' });
                  }, 120);
                }}
                onNavigateToXai={() => setActiveTab('xai')}
              />
            )}

            {activeTab === 'evaluation' && <EvaluationBenchmarks forecast={forecast} />}

            {activeTab === 'trajectory' && (
              <TrajectoryMap
                forecast={forecast}
                allFloats={floats}
                track={floatTrack}
                onSelectFloat={(wmo) => {
                  setSelectedWmo(wmo);
                  const found = floats.find((f) => f.wmo_id === wmo);
                  if (found) setSelectedCycle(found.defaultCycle);
                }}
              />
            )}
          </div>
        ) : null}
      </main>

    </div>
  );
}
