import React, { useState, useEffect, useRef } from 'react';
import { ForecastResult } from '../types.ts';
import {
  Compass,
  MapPin,
  Navigation,
  Waves,
  Layers,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Play,
  Pause,
  SkipForward,
  Info,
  Thermometer,
  Droplets,
  Anchor,
  Activity,
  ArrowUpRight,
  Maximize2
} from 'lucide-react';

interface TrajectoryMapProps {
  forecast: ForecastResult;
  allFloats: Array<{
    wmo_id: string;
    name: string;
    baseLat: number;
    baseLon: number;
    cycles?: number[];
    defaultCycle?: number;
    firstDate?: string;
    lastDate?: string;
  }>;
  onSelectFloat: (wmoId: string) => void;
  /** Real recorded positions of the selected float around the forecast, oldest first. */
  track?: Array<{ cycle: number; lat: number; lon: number; date?: string }>;
}

export const TrajectoryMap: React.FC<TrajectoryMapProps> = ({
  forecast,
  allFloats,
  onSelectFloat,
  track = []
}) => {
  // Map View Mode: Bathymetry, SST (Thermal), SSS (Haline)
  const [mapLayer, setMapLayer] = useState<'bathymetry' | 'sst' | 'salinity'>('bathymetry');
  const [showEvidenceLinks, setShowEvidenceLinks] = useState(true);
  const [showTrajectoryHistory, setShowTrajectoryHistory] = useState(true);
  const [showRangeRings, setShowRangeRings] = useState(true);
  const [showBathymetricContours, setShowBathymetricContours] = useState(true);
  const [selectedFloatDetails, setSelectedFloatDetails] = useState<string | null>(forecast.target_float_id);
  const [hoverCoord, setHoverCoord] = useState<{ lat: number; lon: number } | null>(null);

  // Zoom & Pan state
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  // Playback animation state for drift trajectory
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackCycleIndex, setPlaybackCycleIndex] = useState(0);

  // Map coordinate bounds: 6°N to 26°N, 53°E to 79°E
  const minLat = 6.0;
  const maxLat = 26.0;
  const minLon = 53.0;
  const maxLon = 79.0;

  const baseWidth = 860;
  const baseHeight = 540;

  const lonToX = (lon: number) => ((lon - minLon) / (maxLon - minLon)) * baseWidth;
  const latToY = (lat: number) => baseHeight - ((lat - minLat) / (maxLat - minLat)) * baseHeight;
  const xToLon = (x: number) => minLon + (x / baseWidth) * (maxLon - minLon);
  const yToLat = (y: number) => maxLat - (y / baseHeight) * (maxLat - minLat);

  // Generate historical drift track for current target float
  const targetLat = forecast.latitude;
  const targetLon = forecast.longitude;
  // Recorded positions (from the profile archive). The forecast target is shown at its
  // observed position when it exists; position itself is not forecast by the model.
  const driftTrack: { cycle: number; lat: number; lon: number; label: string; isForecast?: boolean }[] =
    track.length > 0
      ? track.map((p) => {
          const isTarget = p.cycle === forecast.predicted_cycle;
          const isBase = p.cycle === forecast.target_cycle;
          return {
            cycle: p.cycle,
            lat: p.lat,
            lon: p.lon,
            label: `Cycle ${p.cycle}${p.date ? ` (${p.date})` : ''}${isTarget ? ' - forecast target' : isBase ? ' - base cycle' : ''}`,
            isForecast: isTarget,
          };
        })
      : [{ cycle: forecast.target_cycle, lat: targetLat, lon: targetLon, label: `Cycle ${forecast.target_cycle} - base cycle` }];

  // Animation timer
  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setPlaybackCycleIndex((prev) => (prev >= driftTrack.length - 1 ? 0 : prev + 1));
      }, 1200);
    }
    return () => clearInterval(interval);
  }, [isPlaying, driftTrack.length]);

  // Inspect current selected float
  const activeInspectorFloat = allFloats.find((f) => f.wmo_id === selectedFloatDetails) || {
    wmo_id: forecast.target_float_id,
    name: `Target Argo Float #${forecast.target_float_id}`,
    baseLat: forecast.latitude,
    baseLon: forecast.longitude
  };

  const isCurrentTarget = activeInspectorFloat.wmo_id === forecast.target_float_id;
  const inspectorRecord = allFloats.find((f) => f.wmo_id === activeInspectorFloat.wmo_id);

  // Drift between the last two recorded positions of the target float (real track only)
  const drift = (() => {
    const pts = track.filter((p) => p.cycle <= forecast.target_cycle);
    if (pts.length < 2) return null;
    const a = pts[pts.length - 2];
    const b = pts[pts.length - 1];
    if (!a.date || !b.date) return null;
    const days = (new Date(b.date).getTime() - new Date(a.date).getTime()) / 86400000;
    if (days <= 0) return null;
    const R = 6371000;
    const dLat = ((b.lat - a.lat) * Math.PI) / 180;
    const dLon = ((b.lon - a.lon) * Math.PI) / 180 * Math.cos((((a.lat + b.lat) / 2) * Math.PI) / 180);
    const metres = R * Math.sqrt(dLat * dLat + dLon * dLon);
    const bearing = (Math.atan2(dLon, dLat) * 180) / Math.PI;
    const dirs = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west'];
    const dir = dirs[Math.round(((bearing + 360) % 360) / 45) % 8];
    return { cms: (metres / (days * 86400)) * 100, dir, days };
  })();

  // Compute depth estimate from coordinates
  const getEstimatedDepth = (lat: number, lon: number): { depth: number; basin: string } => {
    // Distance from coastlines
    if (lat > 22 && lon > 68) return { depth: 85, basin: 'Gulf of Kutch / Saurashtra Shelf' };
    if (lon > 73.5 && lat < 18) return { depth: 150, basin: 'Konkan-Malabar Continental Shelf' };
    if (lon < 58 && lat > 22) return { depth: 420, basin: 'Gulf of Oman Slope' };
    if (lat < 12 && lon < 55) return { depth: 2100, basin: 'Gulf of Aden Deep Channel' };
    if (lat >= 12 && lat <= 18 && lon >= 62 && lon <= 70) return { depth: 4150, basin: 'Central Arabian Basin Abyssal Plain' };
    if (lat > 18 && lon >= 62 && lon <= 66) return { depth: 3250, basin: 'Northern Arabian Basin' };
    if (lat < 12 && lon >= 64 && lon <= 72) return { depth: 3900, basin: 'Carlsberg Ridge Flank' };
    return { depth: 3100, basin: 'Arabian Sea Pelagic Zone' };
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const normX = (mouseX / rect.width) * baseWidth;
    const normY = (mouseY / rect.height) * baseHeight;
    const lat = Number(yToLat(normY).toFixed(2));
    const lon = Number(xToLon(normX).toFixed(2));
    if (lat >= minLat && lat <= maxLat && lon >= minLon && lon <= maxLon) {
      setHoverCoord({ lat, lon });
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Map Control Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          {/* Map Layer Mode Selector */}
          <div className="inline-flex rounded-lg bg-slate-100 p-1 text-xs font-semibold">
            <button
              onClick={() => setMapLayer('bathymetry')}
              className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 transition-all ${
                mapLayer === 'bathymetry'
                  ? 'bg-cyan-600 text-white shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Waves className="w-3.5 h-3.5" />
              Bathymetry & Topography
            </button>
            <button
              onClick={() => setMapLayer('sst')}
              className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 transition-all ${
                mapLayer === 'sst'
                  ? 'bg-rose-600 text-white shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Thermometer className="w-3.5 h-3.5" />
              SST Thermal Layer
            </button>
            <button
              onClick={() => setMapLayer('salinity')}
              className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 transition-all ${
                mapLayer === 'salinity'
                  ? 'bg-blue-600 text-white shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Droplets className="w-3.5 h-3.5" />
              Salinity Haline Core
            </button>
          </div>
        </div>

        {/* Layer Feature Toggles */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            onClick={() => setShowTrajectoryHistory(!showTrajectoryHistory)}
            className={`px-2.5 py-1 rounded-md border font-medium transition-colors flex items-center gap-1 ${
              showTrajectoryHistory
                ? 'bg-cyan-50 border-cyan-300 text-cyan-800'
                : 'bg-slate-50 border-slate-200 text-slate-500'
            }`}
          >
            <Activity className="w-3 h-3" />
            Drift Track
          </button>
          <button
            onClick={() => setShowEvidenceLinks(!showEvidenceLinks)}
            className={`px-2.5 py-1 rounded-md border font-medium transition-colors flex items-center gap-1 ${
              showEvidenceLinks
                ? 'bg-emerald-50 border-emerald-300 text-emerald-800'
                : 'bg-slate-50 border-slate-200 text-slate-500'
            }`}
          >
            <Layers className="w-3 h-3" />
            Evidence Vectors
          </button>
          <button
            onClick={() => setShowRangeRings(!showRangeRings)}
            className={`px-2.5 py-1 rounded-md border font-medium transition-colors flex items-center gap-1 ${
              showRangeRings
                ? 'bg-purple-50 border-purple-300 text-purple-800'
                : 'bg-slate-50 border-slate-200 text-slate-500'
            }`}
          >
            <Compass className="w-3 h-3" />
            Range Rings
          </button>
          <button
            onClick={() => setShowBathymetricContours(!showBathymetricContours)}
            className={`px-2.5 py-1 rounded-md border font-medium transition-colors flex items-center gap-1 ${
              showBathymetricContours
                ? 'bg-blue-50 border-blue-300 text-blue-800'
                : 'bg-slate-50 border-slate-200 text-slate-500'
            }`}
          >
            <Anchor className="w-3 h-3" />
            Ridges & Trenches
          </button>
        </div>
      </div>

      {/* Main Interactive Map Stage */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* SVG Sea Map (3 Columns) */}
        <div className="lg:col-span-3 bg-slate-950 border border-slate-800 rounded-xl overflow-hidden relative shadow-lg">
          {/* Map Top Floating Header */}
          <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-slate-900/85 backdrop-blur-xs px-3 py-1.5 rounded-lg border border-slate-700/80 text-xs text-white">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            <span className="font-semibold text-slate-200">Northern Indian Ocean Hydrographic Basin</span>
            <span className="text-[10px] text-slate-400 font-mono">6°N–26°N, 53°E–79°E</span>
          </div>

          {/* Zoom & View Controls Overlay */}
          <div className="absolute top-3 right-3 z-10 flex flex-col gap-1 bg-slate-900/85 backdrop-blur-xs p-1 rounded-lg border border-slate-700/80 shadow-md">
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.0, Number((z + 0.2).toFixed(1))))}
              title="Zoom In"
              className="w-7 h-7 flex items-center justify-center text-slate-200 hover:text-white hover:bg-slate-800 rounded"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => setZoomLevel((z) => Math.max(0.8, Number((z - 0.2).toFixed(1))))}
              title="Zoom Out"
              className="w-7 h-7 flex items-center justify-center text-slate-200 hover:text-white hover:bg-slate-800 rounded"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={() => {
                setZoomLevel(1);
                setPanOffset({ x: 0, y: 0 });
              }}
              title="Reset View"
              className="w-7 h-7 flex items-center justify-center text-slate-200 hover:text-white hover:bg-slate-800 rounded"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Map Canvas SVG */}
          <div className="relative overflow-hidden w-full aspect-[860/540]">
            <svg
              viewBox={`0 0 ${baseWidth} ${baseHeight}`}
              className="w-full h-full select-none cursor-crosshair transition-transform duration-300"
              style={{
                transform: `scale(${zoomLevel}) translate(${panOffset.x}px, ${panOffset.y}px)`,
                transformOrigin: 'center center'
              }}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverCoord(null)}
            >
              <defs>
                {/* Bathymetry Depth Gradients */}
                <linearGradient id="deepBasinGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#081426" />
                  <stop offset="50%" stopColor="#0a1d37" />
                  <stop offset="100%" stopColor="#061221" />
                </linearGradient>

                {/* SST Heatmap Gradient */}
                <linearGradient id="sstGradient" x1="0%" y1="100%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#0284c7" stopOpacity="0.8" />
                  <stop offset="35%" stopColor="#10b981" stopOpacity="0.75" />
                  <stop offset="70%" stopColor="#f59e0b" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#e11d48" stopOpacity="0.85" />
                </linearGradient>

                {/* SSS Salinity Gradient */}
                <linearGradient id="salGradient" x1="0%" y1="100%" x2="50%" y2="0%">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.7" />
                  <stop offset="50%" stopColor="#6366f1" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#a855f7" stopOpacity="0.85" />
                </linearGradient>

                {/* Glow Filter for Active Float */}
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* 1. Ocean Water Base Canvas */}
              <rect width={baseWidth} height={baseHeight} fill="url(#deepBasinGrad)" />

              {/* 2. Bathymetry Depth Zones & Oceanic Layers */}
              {mapLayer === 'bathymetry' && (
                <g className="bathymetry-layers">
                  {/* Continental Shelf (0-200m depth) Shading off Western India & Oman */}
                  <path
                    d={`M ${lonToX(72)} ${latToY(24)} 
                        L ${lonToX(71.5)} ${latToY(21)} 
                        L ${lonToX(72)} ${latToY(19)} 
                        L ${lonToX(72.8)} ${latToY(15)} 
                        L ${lonToX(74.2)} ${latToY(11)} 
                        L ${lonToX(76.5)} ${latToY(7.5)}
                        L ${lonToX(77.5)} ${latToY(7.5)}
                        L ${lonToX(77)} ${latToY(8.5)}
                        L ${lonToX(75)} ${latToY(12)}
                        L ${lonToX(73.5)} ${latToY(15.5)}
                        L ${lonToX(72.8)} ${latToY(19)}
                        L ${lonToX(72.5)} ${latToY(21.5)}
                        L ${lonToX(72.8)} ${latToY(24)} Z`}
                    fill="#0e3a53"
                    opacity={0.65}
                  />

                  {/* Central Arabian Abyssal Plain (>4000m) Deep Blue Core */}
                  <ellipse
                    cx={lonToX(65.5)}
                    cy={latToY(15.2)}
                    rx={140}
                    ry={95}
                    fill="#040b17"
                    opacity={0.7}
                  />

                  {/* Carlsberg Mid-Ocean Ridge (Seafloor Spreading Center) */}
                  {showBathymetricContours && (
                    <g>
                      <path
                        d={`M ${lonToX(58)} ${latToY(10.5)} 
                            Q ${lonToX(64)} ${latToY(8.5)} ${lonToX(69)} ${latToY(6.5)}`}
                        fill="none"
                        stroke="#1e3a5f"
                        strokeWidth={24}
                        strokeLinecap="round"
                        opacity={0.5}
                      />
                      <path
                        d={`M ${lonToX(58)} ${latToY(10.5)} 
                            Q ${lonToX(64)} ${latToY(8.5)} ${lonToX(69)} ${latToY(6.5)}`}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth={1.5}
                        strokeDasharray="4 6"
                        opacity={0.6}
                      />
                      <text
                        x={lonToX(63.5)}
                        y={latToY(8.8)}
                        className="text-[9px] fill-cyan-300/60 font-mono font-semibold tracking-wider"
                      >
                        Carlsberg Ridge (Spreading Center)
                      </text>
                    </g>
                  )}

                  {/* Murray Ridge (Tectonic boundary near Oman) */}
                  {showBathymetricContours && (
                    <g>
                      <path
                        d={`M ${lonToX(61.5)} ${latToY(22.8)} L ${lonToX(64.5)} ${latToY(18.5)}`}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth={1.2}
                        strokeDasharray="3 3"
                        opacity={0.5}
                      />
                      <text
                        x={lonToX(62.8)}
                        y={latToY(20.5)}
                        className="text-[8px] fill-cyan-300/50 font-mono"
                      >
                        Murray Ridge
                      </text>
                    </g>
                  )}
                </g>
              )}

              {/* 3. Sea Surface Temperature (SST) Thermal Gradient Layer */}
              {mapLayer === 'sst' && (
                <g className="sst-layer">
                  <rect width={baseWidth} height={baseHeight} fill="url(#sstGradient)" />
                  {/* Upwelling cool tongue near Oman & Somalia (24-26°C) */}
                  <ellipse
                    cx={lonToX(57.5)}
                    cy={latToY(18)}
                    rx={65}
                    ry={90}
                    fill="#0284c7"
                    opacity={0.85}
                  />
                  <text
                    x={lonToX(57.5)}
                    y={latToY(18)}
                    textAnchor="middle"
                    className="text-[10px] fill-white font-bold tracking-wider"
                  >
                    Coastal Upwelling (24.5°C)
                  </text>
                  {/* Warm Pool in Eastern Arabian Sea (29.5°C) */}
                  <ellipse
                    cx={lonToX(71.5)}
                    cy={latToY(13)}
                    rx={75}
                    ry={100}
                    fill="#e11d48"
                    opacity={0.7}
                  />
                  <text
                    x={lonToX(71.5)}
                    y={latToY(13)}
                    textAnchor="middle"
                    className="text-[10px] fill-white font-bold tracking-wider"
                  >
                    Warm Pool (29.8°C)
                  </text>
                </g>
              )}

              {/* 4. Sea Surface Salinity (SSS) Haline Overlay */}
              {mapLayer === 'salinity' && (
                <g className="salinity-layer">
                  <rect width={baseWidth} height={baseHeight} fill="url(#salGradient)" />
                  <ellipse
                    cx={lonToX(63)}
                    cy={latToY(21)}
                    rx={110}
                    ry={65}
                    fill="#a855f7"
                    opacity={0.8}
                  />
                  <text
                    x={lonToX(63)}
                    y={latToY(21)}
                    textAnchor="middle"
                    className="text-[10px] fill-white font-bold tracking-wider"
                  >
                    ASHSW High Salinity Core (&gt;36.6 PSU)
                  </text>
                </g>
              )}

              {/* 5. Geographic Coordinate Grid (Graticules) */}
              {[10, 15, 20, 25].map((lat) => (
                <g key={`graticule-lat-${lat}`}>
                  <line
                    x1={0}
                    y1={latToY(lat)}
                    x2={baseWidth}
                    y2={latToY(lat)}
                    stroke="#334155"
                    strokeDasharray="4 6"
                    strokeWidth={0.7}
                    opacity={0.7}
                  />
                  <text x={8} y={latToY(lat) - 4} className="text-[10px] fill-slate-400 font-mono font-semibold">
                    {lat}°N
                  </text>
                </g>
              ))}

              {[55, 60, 65, 70, 75].map((lon) => (
                <g key={`graticule-lon-${lon}`}>
                  <line
                    x1={lonToX(lon)}
                    y1={0}
                    x2={lonToX(lon)}
                    y2={baseHeight}
                    stroke="#334155"
                    strokeDasharray="4 6"
                    strokeWidth={0.7}
                    opacity={0.7}
                  />
                  <text x={lonToX(lon) + 4} y={baseHeight - 10} className="text-[10px] fill-slate-400 font-mono font-semibold">
                    {lon}°E
                  </text>
                </g>
              ))}

              {/* 6. Detailed Geographic Landmass Silhouettes */}
              {/* Landmass Fill and Borders */}
              <g className="landmasses">
                {/* Western India & Gujarat Peninsula */}
                <path
                  d={`M ${lonToX(72.8)} ${latToY(24.5)} 
                      L ${lonToX(69.8)} ${latToY(23.8)} 
                      L ${lonToX(69.0)} ${latToY(22.8)} 
                      L ${lonToX(70.2)} ${latToY(21.5)} 
                      L ${lonToX(72.2)} ${latToY(21.2)} 
                      L ${lonToX(72.8)} ${latToY(19.2)} 
                      L ${lonToX(73.6)} ${latToY(16.0)} 
                      L ${lonToX(74.8)} ${latToY(12.5)} 
                      L ${lonToX(76.2)} ${latToY(9.5)} 
                      L ${lonToX(77.5)} ${latToY(8.1)} 
                      L ${lonToX(78.5)} ${latToY(9.2)} 
                      L ${baseWidth} ${latToY(9.2)} 
                      L ${baseWidth} 0 
                      L ${lonToX(72.8)} 0 Z`}
                  fill="#1e293b"
                  stroke="#475569"
                  strokeWidth={1.8}
                />
                <text x={lonToX(75)} y={latToY(17)} className="text-xs fill-slate-300 font-bold tracking-widest">
                  INDIA
                </text>
                <text x={lonToX(73.2)} y={latToY(19.2)} className="text-[9px] fill-slate-400 font-medium">
                  Mumbai
                </text>
                <circle cx={lonToX(72.8)} cy={latToY(19.0)} r={2.5} fill="#94a3b8" />

                {/* Sri Lanka */}
                <path
                  d={`M ${lonToX(79.5)} ${latToY(9.2)} 
                      Q ${lonToX(81.2)} ${latToY(7.8)} ${lonToX(80.5)} ${latToY(6.2)} 
                      Q ${lonToX(79.5)} ${latToY(7.2)} ${lonToX(79.5)} ${latToY(9.2)} Z`}
                  fill="#1e293b"
                  stroke="#475569"
                  strokeWidth={1.5}
                />

                {/* Arabian Peninsula (Oman, Yemen, Gulf of Oman, UAE, Saudi Arabia) */}
                <path
                  d={`M 0 0 
                      L ${lonToX(60.5)} 0 
                      L ${lonToX(60.0)} ${latToY(25.5)} 
                      L ${lonToX(59.2)} ${latToY(24.2)} 
                      L ${lonToX(59.8)} ${latToY(22.5)} 
                      L ${lonToX(58.5)} ${latToY(20.5)} 
                      L ${lonToX(56.0)} ${latToY(17.5)} 
                      L ${lonToX(54.0)} ${latToY(16.0)} 
                      L ${lonToX(53.0)} ${latToY(15.2)} 
                      L 0 ${latToY(12)} Z`}
                  fill="#1e293b"
                  stroke="#475569"
                  strokeWidth={1.8}
                />
                <text x={lonToX(56.5)} y={latToY(22.2)} className="text-xs fill-slate-300 font-bold tracking-widest">
                  OMAN
                </text>
                <text x={lonToX(54.2)} y={latToY(16.8)} className="text-[10px] fill-slate-400 font-semibold tracking-wider">
                  YEMEN
                </text>

                {/* Pakistan Coast (Makran Coast / Karachi) */}
                <path
                  d={`M ${lonToX(61)} ${latToY(25.3)} 
                      L ${lonToX(64)} ${latToY(25.2)} 
                      L ${lonToX(67)} ${latToY(25.0)} 
                      L ${lonToX(67.2)} ${latToY(24.8)} 
                      L ${lonToX(68.5)} ${latToY(23.8)} 
                      L ${lonToX(69.8)} ${latToY(23.8)} 
                      L ${lonToX(61)} 0 Z`}
                  fill="#1e293b"
                  stroke="#475569"
                  strokeWidth={1.5}
                />
                <text x={lonToX(65.5)} y={latToY(25.5)} className="text-[11px] fill-slate-400 font-bold">
                  PAKISTAN
                </text>

                {/* Horn of Africa / Somalia */}
                <path
                  d={`M 0 ${latToY(12)} 
                      L ${lonToX(51.2)} ${latToY(11.8)} 
                      L ${lonToX(51.4)} ${latToY(10.5)} 
                      L ${lonToX(50.5)} ${latToY(8.0)} 
                      L 0 ${latToY(6.5)} Z`}
                  fill="#1e293b"
                  stroke="#475569"
                  strokeWidth={1.8}
                />
                <text x={lonToX(53.2)} y={latToY(10.5)} className="text-[10px] fill-slate-400 font-bold">
                  HORN OF AFRICA
                </text>

                {/* Socotra Island */}
                <ellipse cx={lonToX(53.9)} cy={latToY(12.5)} rx={14} ry={6} fill="#1e293b" stroke="#475569" strokeWidth={1.2} />
                <text x={lonToX(54.8)} y={latToY(12.8)} className="text-[8px] fill-slate-400 font-mono">
                  Socotra
                </text>

                {/* Lakshadweep Archipelago */}
                {[
                  { lat: 10.57, lon: 72.63 },
                  { lat: 11.2, lon: 72.8 },
                  { lat: 9.8, lon: 72.9 },
                  { lat: 8.3, lon: 73.0 }
                ].map((island, idx) => (
                  <circle
                    key={`island-${idx}`}
                    cx={lonToX(island.lon)}
                    cy={latToY(island.lat)}
                    r={2.2}
                    fill="#38bdf8"
                    opacity={0.8}
                  />
                ))}
                <text x={lonToX(71.8)} y={latToY(10.8)} className="text-[8px] fill-cyan-400/80 font-mono font-medium">
                  Lakshadweep
                </text>
              </g>

              {/* 7. Oceanographic Basin & Water Body Typography */}
              <text x={lonToX(64.5)} y={latToY(15.5)} textAnchor="middle" className="text-sm fill-cyan-300/35 font-extrabold tracking-[0.25em] uppercase pointer-events-none">
                Arabian Sea Basin
              </text>
              <text x={lonToX(61.0)} y={latToY(23.5)} textAnchor="middle" className="text-[10px] fill-cyan-300/40 font-bold uppercase tracking-wider pointer-events-none">
                Gulf of Oman
              </text>
              <text x={lonToX(55.0)} y={latToY(13.2)} textAnchor="middle" className="text-[10px] fill-cyan-300/40 font-bold uppercase tracking-wider pointer-events-none">
                Gulf of Aden Entrance
              </text>

              {/* 8. Distance Range Rings around Target Float */}
              {showRangeRings && (
                <g className="range-rings pointer-events-none">
                  {[150, 300, 450].map((radiusKm) => {
                    // Approximate ~111 km per degree latitude
                    const rPixels = (radiusKm / 111) * ((baseHeight) / (maxLat - minLat));
                    const currentX = lonToX(forecast.longitude);
                    const currentY = latToY(forecast.latitude);
                    return (
                      <g key={`ring-${radiusKm}`}>
                        <circle
                          cx={currentX}
                          cy={currentY}
                          r={rPixels}
                          fill="none"
                          stroke="#06b6d4"
                          strokeWidth={1}
                          strokeDasharray="4 6"
                          opacity={0.3}
                        />
                        <text
                          x={currentX + rPixels - 24}
                          y={currentY - 4}
                          className="text-[9px] fill-cyan-400/60 font-mono"
                        >
                          {radiusKm} km
                        </text>
                      </g>
                    );
                  })}
                </g>
              )}

              {/* 9. Evidence Link Vectors from Target Float to Cited Analogues */}
              {showEvidenceLinks && (
                <g className="evidence-links">
                  {forecast.evidence_citations.map((c, i) => {
                    const citeX = lonToX(c.longitude);
                    const citeY = latToY(c.latitude);
                    const currX = lonToX(forecast.longitude);
                    const currY = latToY(forecast.latitude);
                    return (
                      <g key={`evidence-line-${c.citation_id}`} className="group">
                        <line
                          x1={currX}
                          y1={currY}
                          x2={citeX}
                          y2={citeY}
                          stroke="#10b981"
                          strokeWidth={1.8}
                          strokeDasharray="4 4"
                          strokeOpacity={0.7}
                        />
                        <circle cx={citeX} cy={citeY} r={5} fill="#10b981" stroke="#ffffff" strokeWidth={1.5} />
                        <text
                          x={citeX + 8}
                          y={citeY + 3}
                          className="text-[10px] fill-emerald-300 font-mono font-bold drop-shadow-xs"
                        >
                          Cite #{i + 1} ({c.distance_km}km • {(c.similarity_score * 100).toFixed(0)}%)
                        </text>
                      </g>
                    );
                  })}
                </g>
              )}

              {/* 10. Multi-Cycle Drift Trajectory Track */}
              {showTrajectoryHistory && (
                <g className="trajectory-track">
                  {/* Polyline connecting cycles */}
                  {driftTrack.map((pt, i) => {
                    if (i === 0) return null;
                    const prevPt = driftTrack[i - 1];
                    const isProjected = pt.isForecast;
                    return (
                      <line
                        key={`track-seg-${i}`}
                        x1={lonToX(prevPt.lon)}
                        y1={latToY(prevPt.lat)}
                        x2={lonToX(pt.lon)}
                        y2={latToY(pt.lat)}
                        stroke={isProjected ? '#a855f7' : '#06b6d4'}
                        strokeWidth={2.2}
                        strokeDasharray={isProjected ? '4 4' : 'none'}
                        strokeOpacity={0.85}
                      />
                    );
                  })}

                  {/* Waypoint nodes along drift path */}
                  {driftTrack.map((pt, i) => {
                    const cx = lonToX(pt.lon);
                    const cy = latToY(pt.lat);
                    const isForecastNode = pt.isForecast;
                    const isPlaybackActive = i === playbackCycleIndex;
                    return (
                      <g key={`waypoint-${i}`} className="cursor-pointer" onClick={() => setPlaybackCycleIndex(i)}>
                        {isPlaybackActive && (
                          <circle cx={cx} cy={cy} r={14} fill="none" stroke="#22d3ee" strokeWidth={1.5} opacity={0.5} className="animate-ping" />
                        )}
                        <circle
                          cx={cx}
                          cy={cy}
                          r={isForecastNode ? 5 : 4}
                          fill={isForecastNode ? '#a855f7' : '#06b6d4'}
                          stroke="#ffffff"
                          strokeWidth={1.2}
                        />
                        <text
                          x={cx + 8}
                          y={cy - 4}
                          className={`text-[9px] font-mono font-bold ${
                            isForecastNode ? 'fill-purple-300' : 'fill-cyan-300'
                          }`}
                        >
                          Cyc {pt.cycle}
                        </text>
                      </g>
                    );
                  })}
                </g>
              )}

              {/* 11. Regional Neighbor Floats */}
              {allFloats.map((f) => {
                if (f.wmo_id === forecast.target_float_id) return null;
                const fx = lonToX(f.baseLon);
                const fy = latToY(f.baseLat);
                const isSelected = selectedFloatDetails === f.wmo_id;
                return (
                  <g
                    key={`float-${f.wmo_id}`}
                    className="cursor-pointer group"
                    onClick={() => {
                      setSelectedFloatDetails(f.wmo_id);
                    }}
                  >
                    <circle
                      cx={fx}
                      cy={fy}
                      r={isSelected ? 6 : 4.5}
                      fill={isSelected ? '#38bdf8' : '#64748b'}
                      stroke="#ffffff"
                      strokeWidth={1.2}
                      className="transition-transform group-hover:scale-125"
                    />
                    <text
                      x={fx + 7}
                      y={fy + 3}
                      className={`text-[10px] font-mono transition-opacity ${isSelected ? 'fill-cyan-300 opacity-100' : 'fill-cyan-300 opacity-0 group-hover:opacity-100'}`}
                    >
                      WMO {f.wmo_id}
                    </text>
                  </g>
                );
              })}

              {/* 12. Primary Active Target Float Marker (with Radar Sweep Ring) */}
              <g className="target-marker cursor-pointer" onClick={() => setSelectedFloatDetails(forecast.target_float_id)}>
                <circle
                  cx={lonToX(forecast.longitude)}
                  cy={latToY(forecast.latitude)}
                  r={22}
                  fill="none"
                  stroke="#22d3ee"
                  strokeWidth={1.2}
                  opacity={0.4}
                  className="animate-ping"
                />
                <circle
                  cx={lonToX(forecast.longitude)}
                  cy={latToY(forecast.latitude)}
                  r={10}
                  fill="#06b6d4"
                  stroke="#ffffff"
                  strokeWidth={2}
                  filter="url(#glow)"
                />
                <circle
                  cx={lonToX(forecast.longitude)}
                  cy={latToY(forecast.latitude)}
                  r={3}
                  fill="#ffffff"
                />

                {/* Target Float Label Box */}
                <rect
                  x={lonToX(forecast.longitude) + 14}
                  y={latToY(forecast.latitude) - 20}
                  width={150}
                  height={32}
                  rx={6}
                  fill="#0f172a"
                  fillOpacity={0.9}
                  stroke="#38bdf8"
                  strokeWidth={1}
                />
                <text
                  x={lonToX(forecast.longitude) + 20}
                  y={latToY(forecast.latitude) - 7}
                  className="text-[11px] fill-white font-bold font-mono"
                >
                  TARGET: WMO {forecast.target_float_id}
                </text>
                <text
                  x={lonToX(forecast.longitude) + 20}
                  y={latToY(forecast.latitude) + 6}
                  className="text-[9px] fill-cyan-300 font-mono"
                >
                  Cyc {forecast.predicted_cycle} • {forecast.latitude}°N, {forecast.longitude}°E
                </text>
              </g>

              {/* 13. Nautical Compass Rose (Upper Right) */}
              <g transform={`translate(${baseWidth - 55}, 55)`} className="pointer-events-none opacity-80">
                <circle cx={0} cy={0} r={22} fill="#0f172a" stroke="#334155" strokeWidth={1} />
                <line x1={0} y1={-20} x2={0} y2={20} stroke="#38bdf8" strokeWidth={1.5} />
                <line x1={-20} y1={0} x2={20} y2={0} stroke="#38bdf8" strokeWidth={1.5} />
                <polygon points="0,-18 -4,-5 0,-8" fill="#e11d48" />
                <polygon points="0,-18 4,-5 0,-8" fill="#f43f5e" />
                <polygon points="0,18 -4,5 0,8" fill="#38bdf8" />
                <polygon points="0,18 4,5 0,8" fill="#0284c7" />
                <text x={0} y={-23} textAnchor="middle" className="text-[10px] fill-white font-bold font-mono">N</text>
              </g>

              {/* 14. Nautical Distance Scale Bar (Bottom Right) */}
              <g transform={`translate(${baseWidth - 140}, ${baseHeight - 25})`} className="pointer-events-none opacity-90">
                <rect x={-5} y={-14} width={135} height={22} rx={4} fill="#0f172a" fillOpacity={0.8} />
                <line x1={0} y1={0} x2={120} y2={0} stroke="#ffffff" strokeWidth={2} />
                <line x1={0} y1={-4} x2={0} y2={4} stroke="#ffffff" strokeWidth={2} />
                <line x1={60} y1={-3} x2={60} y2={3} stroke="#ffffff" strokeWidth={1.5} />
                <line x1={120} y1={-4} x2={120} y2={4} stroke="#ffffff" strokeWidth={2} />
                <text x={0} y={-6} className="text-[8px] fill-slate-300 font-mono">0</text>
                <text x={60} y={-6} textAnchor="middle" className="text-[8px] fill-slate-300 font-mono">150 km</text>
                <text x={120} y={-6} textAnchor="end" className="text-[8px] fill-slate-300 font-mono">300 km</text>
              </g>
            </svg>
          </div>

          {/* Map Bottom Status Bar / Cursor HUD */}
          <div className="bg-slate-900 border-t border-slate-800 p-2.5 px-4 flex flex-wrap items-center justify-between text-xs text-slate-300 gap-2">
            <div className="flex items-center gap-2">
              <Compass className="w-4 h-4 text-cyan-400" />
              {hoverCoord ? (
                <div className="flex items-center gap-3 font-mono">
                  <span>Cursor: <strong>{hoverCoord.lat}°N, {hoverCoord.lon}°E</strong></span>
                  <span className="text-slate-500">|</span>
                  <span>Est. Depth: <strong className="text-cyan-300">{getEstimatedDepth(hoverCoord.lat, hoverCoord.lon).depth} m</strong></span>
                  <span className="text-slate-500">|</span>
                  <span className="text-slate-400">{getEstimatedDepth(hoverCoord.lat, hoverCoord.lon).basin}</span>
                </div>
              ) : (
                <span className="text-slate-400">Hover cursor anywhere on the ocean canvas for precision bathymetric telemetry</span>
              )}
            </div>

            {/* Trajectory Playback Controls */}
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-slate-400">Track playback:</span>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="p-1 px-2 rounded bg-slate-800 hover:bg-slate-700 text-white font-semibold flex items-center gap-1 text-[11px]"
              >
                {isPlaying ? <Pause className="w-3 h-3 text-amber-400" /> : <Play className="w-3 h-3 text-emerald-400" />}
                <span>{isPlaying ? 'Pause' : 'Play Drift'}</span>
              </button>
              <button
                onClick={() => setPlaybackCycleIndex((p) => (p >= driftTrack.length - 1 ? 0 : p + 1))}
                className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                title="Next Cycle"
              >
                <SkipForward className="w-3 h-3" />
              </button>
              <span className="font-mono text-[11px] text-cyan-400 font-bold ml-1">
                {driftTrack[playbackCycleIndex]?.label || `Step ${playbackCycleIndex + 1}`}
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Float Hydrographic Telemetry & Inspector Panel */}
        <div className="space-y-3">
          {/* Selected Float Inspector Card */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <div className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                  isCurrentTarget ? 'bg-cyan-100 text-cyan-800' : 'bg-slate-100 text-slate-700'
                }`}>
                  <Navigation className="w-4 h-4" />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-slate-900 font-mono">
                    WMO {activeInspectorFloat.wmo_id}
                  </h4>
                  <span className={`text-[10px] font-semibold ${isCurrentTarget ? 'text-cyan-700' : 'text-slate-500'}`}>
                    {isCurrentTarget ? '● Active Target Float' : '● Regional Network Float'}
                  </span>
                </div>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
                QC Pass
              </span>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-50">
                <span className="text-slate-500">Record:</span>
                <span className="font-semibold text-slate-800 font-mono">
                  {inspectorRecord?.firstDate && inspectorRecord?.lastDate
                    ? `${inspectorRecord.firstDate.slice(0, 7)} to ${inspectorRecord.lastDate.slice(0, 7)}`
                    : '--'}
                  {inspectorRecord?.cycles ? ` · ${inspectorRecord.cycles.length} profiles` : ''}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-50">
                <span className="text-slate-500">Coordinates:</span>
                <span className="font-semibold font-mono text-slate-800">
                  {activeInspectorFloat.baseLat.toFixed(3)}°N, {activeInspectorFloat.baseLon.toFixed(3)}°E
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-50">
                <span className="text-slate-500">Region (approx.):</span>
                <span className="font-semibold text-slate-800">
                  {getEstimatedDepth(activeInspectorFloat.baseLat, activeInspectorFloat.baseLon).basin}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-50">
                <span className="text-slate-500">Recent drift:</span>
                <span className="font-semibold text-slate-800 font-mono">
                  {isCurrentTarget && drift ? `${drift.cms.toFixed(1)} cm/s toward ${drift.dir} (${drift.days.toFixed(0)} d)` : '--'}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-50">
                <span className="text-slate-500">Profile Reach:</span>
                <span className="font-semibold text-slate-800 font-mono">0 to 1000 dbar</span>
              </div>
            </div>

            {!isCurrentTarget && (
              <button
                onClick={() => onSelectFloat(activeInspectorFloat.wmo_id)}
                className="w-full mt-3 py-2 px-3 rounded-lg bg-cyan-600 hover:bg-cyan-700 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-1.5 shadow-xs"
              >
                <ArrowUpRight className="w-3.5 h-3.5" />
                Set as Active Target Float
              </button>
            )}
          </div>

          {/* Map Legend Card */}
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs space-y-3">
            <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-cyan-600" />
              Oceanographic Map Legend
            </h4>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-cyan-400 border border-white shrink-0"></span>
                <span className="text-slate-700">Target Float (Active Cycle {forecast.predicted_cycle})</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-emerald-500 border border-white shrink-0"></span>
                <span className="text-slate-700">Historical Analogue Evidence Floats</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-slate-500 border border-white shrink-0"></span>
                <span className="text-slate-700">Regional Monitoring Floats</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-0.5 bg-cyan-400 border-dashed shrink-0"></span>
                <span className="text-slate-700">Recorded track (recent profiles and forecast target)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-4 h-0.5 bg-emerald-400 border-dashed shrink-0"></span>
                <span className="text-slate-700">Cosine Distance / Evidence Citation Vector</span>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-400">
              Covers the Arabian Sea basin, Oman upwelling zone, and Indian continental shelf with TEOS-10 calibrated physical boundaries.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
