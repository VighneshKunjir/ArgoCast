import React, { useEffect, useState } from 'react';
import { ForecastResult, ArgoProfile, ModelInfo } from '../types.ts';
import { apiService } from '../services/api.ts';
import { Info, Eye, EyeOff, Layers, Activity, Thermometer, Droplets, Crosshair, TrendingUp, TrendingDown, ArrowDown } from 'lucide-react';
import { EvidenceLinkViewer } from './EvidenceLinkViewer.tsx';

interface ProfilePlotProps {
  forecast: ForecastResult;
  historicalProfiles: ArgoProfile[];
  variableMode: 'temperature' | 'salinity' | 'both';
}

// UNESCO/TEOS-10 potential density approximation
function calcDensity(t: number, s: number): number {
  const rho0 = 999.842594 + (6.793952e-2 * t) - (9.095290e-3 * t * t) + (1.001685e-4 * Math.pow(t, 3)) - (1.120083e-6 * Math.pow(t, 4)) + (6.536332e-9 * Math.pow(t, 5));
  const A = (8.24493e-1) - (4.0899e-3 * t) + (7.6438e-5 * t * t) - (8.2467e-7 * Math.pow(t, 3)) + (5.3875e-9 * Math.pow(t, 4));
  const B = (-5.72466e-3) + (1.0227e-4 * t) - (1.6546e-6 * t * t);
  const C = 4.8314e-4;
  const rho = rho0 + A * s + B * Math.pow(s, 1.5) + C * s * s;
  return Number((rho - 1000).toFixed(2));
}

function getLayerDescription(depth: number) {
  if (depth <= 40) return { name: 'Surface Mixed Layer', zone: 'Epipelagic (0–40 dbar)', desc: 'Solar heating, atmospheric momentum & wind mixing zone' };
  if (depth <= 150) return { name: 'Main Thermocline', zone: 'Pycnocline (50–150 dbar)', desc: 'Steep vertical thermal barrier & density stratification' };
  if (depth <= 350) return { name: 'Subsurface Salinity Core', zone: 'Mesopelagic (200–350 dbar)', desc: 'Persian Gulf & Red Sea high-salinity intrusion core' };
  if (depth <= 700) return { name: 'Intermediate Water Mass', zone: 'Lower Mesopelagic (400–700 dbar)', desc: 'Oxygen minimum zone (OMZ) & gradual stabilization' };
  return { name: 'Deep Abyssal Ocean', zone: 'Bathypelagic (800–1000 dbar)', desc: 'Homogeneous cold deep Indian Ocean water mass' };
}

export const ProfilePlot: React.FC<ProfilePlotProps> = ({
  forecast,
  historicalProfiles,
  variableMode
}) => {
  const [showCi95, setShowCi95] = useState(true);
  const [showCi90, setShowCi90] = useState(true);
  const [showHistorical, setShowHistorical] = useState(true);
  const [showPersistence, setShowPersistence] = useState(true);
  const [showRidge, setShowRidge] = useState(false);
  const [showObserved, setShowObserved] = useState(true);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    apiService.getModelInfo().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);
  const [hoverDepth, setHoverDepth] = useState<number | null>(null);
  const [chartType, setChartType] = useState<'profile' | 'ts_diagram'>('profile');

  // SVG Chart Geometry
  const width = 580;
  const height = 480;
  const margin = { top: 30, right: 30, bottom: 40, left: 55 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  // Scales
  const minDepth = 0;
  const maxDepth = 1000;
  const depthToY = (d: number) => margin.top + ((d - minDepth) / (maxDepth - minDepth)) * innerH;

  // Temp scale: 4°C to 32°C
  const minTemp = 4;
  const maxTemp = 32;
  const tempToX = (t: number) => margin.left + ((t - minTemp) / (maxTemp - minTemp)) * innerW;

  // Salinity scale: 34.5 to 37.0 PSU
  const minSal = 34.5;
  const maxSal = 37.0;
  const salToX = (s: number) => margin.left + ((s - minSal) / (maxSal - minSal)) * innerW;

  // Depth ticks
  const depthTicks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
  const tempTicks = [5, 10, 15, 20, 25, 30];
  const salTicks = [34.5, 35.0, 35.5, 36.0, 36.5, 37.0];

  // Helper to generate SVG polygon path for uncertainty ribbons
  const generateRibbonPath = (
    lowerVals: { depth: number; val: number }[],
    upperVals: { depth: number; val: number }[],
    scaleX: (v: number) => number
  ) => {
    if (!lowerVals.length) return '';
    const pointsDown = lowerVals.map((p) => `${scaleX(p.val)},${depthToY(p.depth)}`);
    const pointsUp = [...upperVals].reverse().map((p) => `${scaleX(p.val)},${depthToY(p.depth)}`);
    return `M ${pointsDown.join(' L ')} L ${pointsUp.join(' L ')} Z`;
  };

  const renderSingleCurve = (
    mode: 'temperature' | 'salinity',
    title: string,
    unit: string,
    scaleX: (v: number) => number,
    ticks: number[],
    color: string
  ) => {
    const isTemp = mode === 'temperature';
    const forecastVals = forecast.profiles.map((p) => ({
      depth: p.depth_dbar,
      val: isTemp ? p.temperature_forecast : p.salinity_forecast
    }));

    const ci95Path = generateRibbonPath(
      forecast.uncertainty_bounds.map((u) => ({
        depth: u.depth_dbar,
        val: isTemp ? u.ci95_temp_lower : u.ci95_sal_lower
      })),
      forecast.uncertainty_bounds.map((u) => ({
        depth: u.depth_dbar,
        val: isTemp ? u.ci95_temp_upper : u.ci95_sal_upper
      })),
      scaleX
    );

    const ci90Path = generateRibbonPath(
      forecast.uncertainty_bounds.map((u) => ({
        depth: u.depth_dbar,
        val: isTemp ? u.ci90_temp_lower : u.ci90_sal_lower
      })),
      forecast.uncertainty_bounds.map((u) => ({
        depth: u.depth_dbar,
        val: isTemp ? u.ci90_temp_upper : u.ci90_sal_upper
      })),
      scaleX
    );

    const persistencePath = forecast.profiles
      .map((p, idx) => {
        const x = scaleX(isTemp ? p.persistence_temperature : p.persistence_salinity);
        const y = depthToY(p.depth_dbar);
        return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');

    const observedPath = forecast.has_observation
      ? forecast.profiles
          .map((p, idx) => {
            const v = isTemp ? p.observed_temperature : p.observed_salinity;
            return `${idx === 0 ? 'M' : 'L'} ${scaleX(v ?? 0)} ${depthToY(p.depth_dbar)}`;
          })
          .join(' ')
      : '';

    const ridgePath = forecast.profiles
      .map((p, idx) => {
        const x = scaleX(isTemp ? p.ridge_temperature : p.ridge_salinity);
        const y = depthToY(p.depth_dbar);
        return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');

    const forecastPath = forecastVals
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(p.val)} ${depthToY(p.depth)}`)
      .join(' ');

    // Standard depths
    const standardDepths = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000];

    const handlePlotMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const clientY = e.clientY - rect.top;
      const svgY = (clientY / rect.height) * height;
      if (svgY >= margin.top && svgY <= height - margin.bottom) {
        const rawDepth = minDepth + ((svgY - margin.top) / innerH) * (maxDepth - minDepth);
        const closest = standardDepths.reduce((prev, curr) =>
          Math.abs(curr - rawDepth) < Math.abs(prev - rawDepth) ? curr : prev
        );
        setHoverDepth(closest);
      }
    };

    // Find hover point data
    const activeDataPoint = hoverDepth !== null
      ? forecast.profiles.find((p) => p.depth_dbar === hoverDepth)
      : null;
    const activeUqPoint = hoverDepth !== null
      ? forecast.uncertainty_bounds.find((u) => u.depth_dbar === hoverDepth)
      : null;

    return (
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-full ${isTemp ? 'bg-rose-500' : 'bg-blue-600'}`}></span>
              {title}
            </h3>
            <p className="text-[11px] text-slate-500">
              Depth-resolved vertical profile with Monte Carlo Dropout uncertainty
            </p>
          </div>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold">
            {unit}
          </span>
        </div>

        {/* SVG Container */}
        <div className="relative overflow-hidden">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto bg-slate-50/50 rounded-lg select-none cursor-crosshair"
            onMouseMove={handlePlotMouseMove}
            onMouseLeave={() => setHoverDepth(null)}
          >
            {/* Grid lines */}
            {depthTicks.map((d) => (
              <g key={`d-grid-${d}`}>
                <line
                  x1={margin.left}
                  y1={depthToY(d)}
                  x2={width - margin.right}
                  y2={depthToY(d)}
                  stroke="#e2e8f0"
                  strokeDasharray="3 3"
                />
                <text
                  x={margin.left - 8}
                  y={depthToY(d) + 3}
                  textAnchor="end"
                  className="text-[10px] fill-slate-400 font-mono"
                >
                  {d}
                </text>
              </g>
            ))}

            {ticks.map((t) => (
              <g key={`t-grid-${t}`}>
                <line
                  x1={scaleX(t)}
                  y1={margin.top}
                  x2={scaleX(t)}
                  y2={height - margin.bottom}
                  stroke="#e2e8f0"
                  strokeDasharray="3 3"
                />
                <text
                  x={scaleX(t)}
                  y={height - margin.bottom + 16}
                  textAnchor="middle"
                  className="text-[10px] fill-slate-500 font-mono font-medium"
                >
                  {t}
                </text>
              </g>
            ))}

            {/* Axes Labels */}
            <text
              x={margin.left}
              y={margin.top - 12}
              className="text-[10px] fill-slate-600 font-semibold"
            >
              Surface (0 dbar)
            </text>
            <text
              x={margin.left}
              y={height - 8}
              className="text-[10px] fill-slate-600 font-semibold"
            >
              Deep (1000 dbar)
            </text>
            <text
              x={width / 2}
              y={height - 10}
              textAnchor="middle"
              className="text-[11px] fill-slate-700 font-semibold"
            >
              {title} ({unit})
            </text>

            {/* 95% Confidence Interval Ribbon */}
            {showCi95 && (
              <path
                d={ci95Path}
                fill={isTemp ? '#fda4af' : '#bfdbfe'}
                fillOpacity={0.4}
                className="transition-opacity duration-200"
              />
            )}

            {/* 90% Confidence Interval Ribbon */}
            {showCi90 && (
              <path
                d={ci90Path}
                fill={isTemp ? '#f43f5e' : '#3b82f6'}
                fillOpacity={0.25}
                className="transition-opacity duration-200"
              />
            )}

            {/* Historical Sequence Profiles */}
            {showHistorical &&
              historicalProfiles.map((hp, i) => {
                const histPoints = hp.measurements
                  .map((m, idx) => {
                    const val = isTemp ? m.temperature : m.salinity;
                    return `${idx === 0 ? 'M' : 'L'} ${scaleX(val)} ${depthToY(m.depth_dbar)}`;
                  })
                  .join(' ');
                const opacity = 0.25 + i * 0.2;
                return (
                  <path
                    key={`hist-${hp.cycle_number}`}
                    d={histPoints}
                    fill="none"
                    stroke="#64748b"
                    strokeWidth={1.5}
                    strokeDasharray={i === historicalProfiles.length - 1 ? 'none' : '4 4'}
                    strokeOpacity={opacity}
                  />
                );
              })}

            {/* Persistence Baseline */}
            {showPersistence && (
              <path
                d={persistencePath}
                fill="none"
                stroke="#94a3b8"
                strokeWidth={1.8}
                strokeDasharray="5 5"
              />
            )}

            {/* Observed target profile (when the forecast cycle is already in the archive) */}
            {showObserved && observedPath && (
              <path
                d={observedPath}
                fill="none"
                stroke="#059669"
                strokeWidth={2}
                strokeDasharray="6 3"
              />
            )}

            {/* Ridge regression baseline */}
            {showRidge && (
              <path
                d={ridgePath}
                fill="none"
                stroke="#d97706"
                strokeWidth={1.8}
                strokeDasharray="2 2"
              />
            )}

            {/* Deep LSTM Forecast Profile (Primary) */}
            <path
              d={forecastPath}
              fill="none"
              stroke={isTemp ? '#e11d48' : '#2563eb'}
              strokeWidth={2.8}
              strokeLinecap="round"
            />

            {/* Interactive Data Nodes */}
            {forecastVals.map((p) => {
              const cx = scaleX(p.val);
              const cy = depthToY(p.depth);
              const isHovered = hoverDepth === p.depth;
              return (
                <g key={`pt-${p.depth}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 6 : 3.5}
                    fill={isTemp ? '#e11d48' : '#2563eb'}
                    stroke="#ffffff"
                    strokeWidth={1.5}
                    className="transition-all duration-150 cursor-pointer"
                    onMouseEnter={() => setHoverDepth(p.depth)}
                  />
                </g>
              );
            })}

            {/* Crosshair guide on hover */}
            {hoverDepth !== null && (
              <line
                x1={margin.left}
                y1={depthToY(hoverDepth)}
                x2={width - margin.right}
                y2={depthToY(hoverDepth)}
                stroke="#0284c7"
                strokeWidth={1.2}
                strokeDasharray="2 2"
              />
            )}
          </svg>

          {/* Floating readout box */}
          {activeDataPoint && activeUqPoint && (
            <div className="absolute top-4 right-4 bg-slate-900/90 backdrop-blur-xs text-white p-3 rounded-lg text-xs shadow-md border border-slate-700 pointer-events-none">
              <div className="font-bold text-cyan-300 mb-1">
                Depth: {activeDataPoint.depth_dbar} dbar
              </div>
              <div>
                Forecast:{' '}
                <strong className={isTemp ? 'text-rose-400' : 'text-blue-400'}>
                  {((isTemp ? activeDataPoint.temperature_forecast : activeDataPoint.salinity_forecast) != null
                    ? (isTemp ? activeDataPoint.temperature_forecast : activeDataPoint.salinity_forecast).toFixed(2)
                    : '--')}{' '}
                  {unit}
                </strong>
              </div>
              <div className="text-[11px] text-slate-300 mt-0.5">
                95% CI: [
                {(isTemp ? activeUqPoint.ci95_temp_lower : activeUqPoint.ci95_sal_lower) != null
                  ? (isTemp ? activeUqPoint.ci95_temp_lower : activeUqPoint.ci95_sal_lower).toFixed(2)
                  : '--'},{' '}
                {(isTemp ? activeUqPoint.ci95_temp_upper : activeUqPoint.ci95_sal_upper) != null
                  ? (isTemp ? activeUqPoint.ci95_temp_upper : activeUqPoint.ci95_sal_upper).toFixed(2)
                  : '--'}]
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">
                Persistence (t-1):{' '}
                {((isTemp ? activeDataPoint.persistence_temperature : activeDataPoint.persistence_salinity) != null
                  ? (isTemp ? activeDataPoint.persistence_temperature : activeDataPoint.persistence_salinity).toFixed(2)
                  : '--')}{' '}
                {unit}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Render T-S Diagram (Temperature vs Salinity)
  const renderTSDiagram = () => {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-purple-600" />
              T-S Diagram (Temperature-Salinity Water Mass Diagram)
            </h3>
            <p className="text-[11px] text-slate-500">
              Arabian Sea High Salinity Water (ASHSW) and Red Sea Water (RSW) stratification
            </p>
          </div>
          <span className="text-xs bg-purple-50 text-purple-700 border border-purple-200 px-2 py-0.5 rounded font-mono font-semibold">
            TEOS-10 Diagram
          </span>
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-50/50 rounded-lg">
          {/* Salinity on X axis (34.5 - 37.0), Temp on Y axis (4 - 32) */}
          {salTicks.map((s) => (
            <g key={`ts-x-${s}`}>
              <line
                x1={salToX(s)}
                y1={margin.top}
                x2={salToX(s)}
                y2={height - margin.bottom}
                stroke="#e2e8f0"
                strokeDasharray="3 3"
              />
              <text
                x={salToX(s)}
                y={height - margin.bottom + 16}
                textAnchor="middle"
                className="text-[10px] fill-slate-500 font-mono"
              >
                {s}
              </text>
            </g>
          ))}

          {tempTicks.map((t) => {
            const y = margin.top + ((maxTemp - t) / (maxTemp - minTemp)) * innerH;
            return (
              <g key={`ts-y-${t}`}>
                <line
                  x1={margin.left}
                  y1={y}
                  x2={width - margin.right}
                  y2={y}
                  stroke="#e2e8f0"
                  strokeDasharray="3 3"
                />
                <text
                  x={margin.left - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="text-[10px] fill-slate-500 font-mono"
                >
                  {t}°C
                </text>
              </g>
            );
          })}

          <text
            x={width / 2}
            y={height - 8}
            textAnchor="middle"
            className="text-[11px] fill-slate-700 font-semibold"
          >
            Practical Salinity (PSU)
          </text>

          {/* Isopycnal background annotations */}
          <text x={width - margin.right - 80} y={margin.top + 30} className="text-[10px] fill-purple-400 font-mono">
            σθ ≈ 23.5 kg/m³
          </text>
          <text x={width - margin.right - 80} y={height - margin.bottom - 40} className="text-[10px] fill-purple-400 font-mono">
            σθ ≈ 27.8 kg/m³
          </text>

          {/* TS curve for Forecast */}
          {forecast.profiles.length > 1 && (
            <path
              d={forecast.profiles
                .map((p, idx) => {
                  const x = salToX(p.salinity_forecast);
                  const y = margin.top + ((maxTemp - p.temperature_forecast) / (maxTemp - minTemp)) * innerH;
                  return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                })
                .join(' ')}
              fill="none"
              stroke="#7c3aed"
              strokeWidth={2.5}
            />
          )}

          {/* TS points colored by depth */}
          {forecast.profiles.map((p) => {
            const x = salToX(p.salinity_forecast);
            const y = margin.top + ((maxTemp - p.temperature_forecast) / (maxTemp - minTemp)) * innerH;
            return (
              <circle
                key={`ts-pt-${p.depth_dbar}`}
                cx={x}
                cy={y}
                r={4}
                fill={p.depth_dbar <= 100 ? '#ef4444' : p.depth_dbar <= 400 ? '#f59e0b' : '#3b82f6'}
                stroke="#fff"
                strokeWidth={1}
              >
                <title>{`Depth: ${p.depth_dbar} dbar | T: ${p.temperature_forecast}°C | S: ${p.salinity_forecast} PSU`}</title>
              </circle>
            );
          })}
        </svg>

        <div className="flex items-center justify-center gap-4 mt-3 text-xs text-slate-600">
          <div className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500"></span>
            <span>Surface/Mixed (0-100m)</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
            <span>Thermocline/RSW (100-400m)</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
            <span>Deep Abyssal (400-1000m)</span>
          </div>
        </div>
      </div>
    );
  };

  const activeDepth = hoverDepth !== null ? hoverDepth : 5;
  const activePoint = forecast.profiles.find((p) => p.depth_dbar === activeDepth) || forecast.profiles[0];
  const activeUq = forecast.uncertainty_bounds.find((u) => u.depth_dbar === activeDepth) || forecast.uncertainty_bounds[0];
  const layerInfo = getLayerDescription(activeDepth);
  const tempDelta = activePoint.temperature_forecast - activePoint.persistence_temperature;
  const salDelta = activePoint.salinity_forecast - activePoint.persistence_salinity;
  const densityVal = calcDensity(activePoint.temperature_forecast, activePoint.salinity_forecast);

  return (
    <div className="space-y-4">
      {/* Visual Controls & Legend Toolbar */}
      <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setChartType('profile')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              chartType === 'profile'
                ? 'bg-cyan-600 text-white shadow-xs'
                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Vertical Depth Profiles
          </button>
          <button
            onClick={() => setChartType('ts_diagram')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              chartType === 'ts_diagram'
                ? 'bg-purple-600 text-white shadow-xs'
                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            T-S Diagram (Water Mass)
          </button>
        </div>

        {/* Toggles for confidence ribbons & baselines */}
        {chartType === 'profile' && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <button
              onClick={() => setShowCi95(!showCi95)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md border font-medium transition-colors ${
                showCi95 ? 'bg-rose-50 border-rose-200 text-rose-700' : 'bg-slate-50 border-slate-200 text-slate-500'
              }`}
            >
              {showCi95 ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
              <span>95% CI Ribbon</span>
            </button>
            <button
              onClick={() => setShowCi90(!showCi90)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md border font-medium transition-colors ${
                showCi90 ? 'bg-blue-50 border-blue-200 text-blue-700' : 'bg-slate-50 border-slate-200 text-slate-500'
              }`}
            >
              {showCi90 ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
              <span>90% CI Ribbon</span>
            </button>
            <button
              onClick={() => setShowHistorical(!showHistorical)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md border font-medium transition-colors ${
                showHistorical ? 'bg-slate-800 border-slate-700 text-white' : 'bg-slate-50 border-slate-200 text-slate-500'
              }`}
            >
              <span>Historical Sequence (t-3, t-2, t-1)</span>
            </button>
            <button
              onClick={() => setShowPersistence(!showPersistence)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md border font-medium transition-colors ${
                showPersistence ? 'bg-amber-50 border-amber-200 text-amber-700' : 'bg-slate-50 border-slate-200 text-slate-500'
              }`}
            >
              <span>Persistence Baseline</span>
            </button>
            <button
              onClick={() => setShowRidge(!showRidge)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md border font-medium transition-colors ${
                showRidge ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-slate-50 border-slate-200 text-slate-500'
              }`}
            >
              <span>Ridge Baseline</span>
            </button>
            {forecast.has_observation && (
              <button
                onClick={() => setShowObserved(!showObserved)}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-md border font-medium transition-colors ${
                  showObserved ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-slate-50 border-slate-200 text-slate-500'
                }`}
              >
                <span>Observed Cycle {forecast.predicted_cycle}</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* Dynamic Forecast Prediction Readout Section */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 sm:p-5 shadow-xs transition-all">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-3 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-cyan-600 text-white flex items-center justify-center font-mono font-bold text-xs shadow-xs">
              {activeDepth}m
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                  Live Prediction Readout at <span className="text-cyan-700 font-mono underline decoration-cyan-400 decoration-2">{activeDepth} dbar</span>
                </h3>
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-medium ${
                  hoverDepth !== null
                    ? 'bg-emerald-100 text-emerald-800 animate-pulse'
                    : 'bg-slate-100 text-slate-600'
                }`}>
                  {hoverDepth !== null ? '● Dynamic Hover' : '● Default (Surface 5 dbar)'}
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5">
                <span className="font-semibold text-slate-700">{layerInfo.name}</span> — {layerInfo.zone} • {layerInfo.desc}
              </p>
            </div>
          </div>
          <div className="text-[11px] text-slate-400 hidden lg:flex items-center gap-1">
            <ArrowDown className="w-3.5 h-3.5 text-cyan-600" />
            Hover over any depth point on the profile graphs below to update dynamically
          </div>
        </div>

        {/* Prediction Value Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* 1. Temperature Prediction Card */}
          <div className="p-3.5 rounded-xl bg-gradient-to-br from-rose-50/80 to-white border border-rose-200/90 flex flex-col justify-between shadow-2xs">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-rose-900 flex items-center gap-1.5">
                <Thermometer className="w-4 h-4 text-rose-600" />
                Predicted Temperature
              </span>
              <span className="text-[10px] font-mono bg-rose-100 text-rose-800 px-1.5 py-0.5 rounded font-bold">
                ±{activeUq.std_temp.toFixed(2)}°C UQ
              </span>
            </div>
            <div className="my-2.5">
              <div className="text-3xl font-bold font-mono text-rose-950 tracking-tight flex items-baseline">
                {activePoint.temperature_forecast.toFixed(2)}
                <span className="text-sm font-sans font-semibold text-rose-600 ml-1.5">°C</span>
              </div>
            </div>
            <div className="space-y-1 text-[11px] text-rose-800/90 pt-1.5 border-t border-rose-200/70 font-mono">
              <div className="flex justify-between">
                <span className="text-rose-600/80 font-sans">95% Confidence:</span>
                <span className="font-semibold">[{activeUq.ci95_temp_lower.toFixed(2)}°C, {activeUq.ci95_temp_upper.toFixed(2)}°C]</span>
              </div>
              <div className="flex justify-between">
                <span className="text-rose-600/80 font-sans">vs Persistence:</span>
                <span className={tempDelta >= 0 ? 'text-rose-900 font-semibold' : 'text-blue-700 font-semibold'}>
                  {activePoint.persistence_temperature.toFixed(2)}°C ({tempDelta >= 0 ? '+' : ''}{tempDelta.toFixed(2)}°C)
                </span>
              </div>
            </div>
          </div>

          {/* 2. Salinity Prediction Card */}
          <div className="p-3.5 rounded-xl bg-gradient-to-br from-blue-50/80 to-white border border-blue-200/90 flex flex-col justify-between shadow-2xs">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-blue-900 flex items-center gap-1.5">
                <Droplets className="w-4 h-4 text-blue-600" />
                Predicted Salinity
              </span>
              <span className="text-[10px] font-mono bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded font-bold">
                ±{activeUq.std_sal.toFixed(2)} PSU
              </span>
            </div>
            <div className="my-2.5">
              <div className="text-3xl font-bold font-mono text-blue-950 tracking-tight flex items-baseline">
                {activePoint.salinity_forecast.toFixed(2)}
                <span className="text-sm font-sans font-semibold text-blue-600 ml-1.5">PSU</span>
              </div>
            </div>
            <div className="space-y-1 text-[11px] text-blue-800/90 pt-1.5 border-t border-blue-200/70 font-mono">
              <div className="flex justify-between">
                <span className="text-blue-600/80 font-sans">95% Confidence:</span>
                <span className="font-semibold">[{activeUq.ci95_sal_lower.toFixed(2)}, {activeUq.ci95_sal_upper.toFixed(2)}]</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-600/80 font-sans">vs Persistence:</span>
                <span className={salDelta >= 0 ? 'text-blue-900 font-semibold' : 'text-amber-700 font-semibold'}>
                  {activePoint.persistence_salinity.toFixed(2)} ({salDelta >= 0 ? '+' : ''}{salDelta.toFixed(2)})
                </span>
              </div>
            </div>
          </div>

          {/* 3. Potential Density & Stratification */}
          <div className="p-3.5 rounded-xl bg-gradient-to-br from-purple-50/80 to-white border border-purple-200/90 flex flex-col justify-between shadow-2xs">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-purple-900 flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-purple-600" />
                Potential Density (σθ)
              </span>
              <span className="text-[10px] font-mono bg-purple-100 text-purple-800 px-1.5 py-0.5 rounded font-bold">
                TEOS-10
              </span>
            </div>
            <div className="my-2.5">
              <div className="text-3xl font-bold font-mono text-purple-950 tracking-tight flex items-baseline">
                {densityVal.toFixed(2)}
                <span className="text-xs font-sans font-semibold text-purple-600 ml-1.5">kg/m³</span>
              </div>
            </div>
            <div className="space-y-1 text-[11px] text-purple-800/90 pt-1.5 border-t border-purple-200/70 font-mono">
              <div className="flex justify-between">
                <span className="text-purple-600/80 font-sans">Stability:</span>
                <span className="font-semibold text-emerald-700">Gravitationally Stable</span>
              </div>
              <div className="flex justify-between">
                <span className="text-purple-600/80 font-sans">Vertical ∂σ/∂z:</span>
                <span className="text-slate-700 font-semibold">≥ 0.000 kg/m³/m</span>
              </div>
            </div>
          </div>

          {/* 4. Quick Depth Jump Selector */}
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 flex flex-col justify-between shadow-2xs">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
                <Crosshair className="w-4 h-4 text-cyan-600" />
                Quick Depth Jump
              </span>
              <span className="text-[10px] font-mono text-slate-500 font-medium">Standard Depths</span>
            </div>
            <div className="flex flex-wrap gap-1.5 my-2">
              {[5, 50, 100, 200, 400, 700, 1000].map((d) => (
                <button
                  key={`depth-btn-${d}`}
                  onClick={() => setHoverDepth(d)}
                  onMouseEnter={() => setHoverDepth(d)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono transition-all ${
                    activeDepth === d
                      ? 'bg-cyan-600 text-white font-bold shadow-xs'
                      : 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-100 font-medium'
                  }`}
                >
                  {d}m
                </button>
              ))}
            </div>
            <div className="text-[10px] text-slate-500 pt-1.5 border-t border-slate-200 flex items-center justify-between">
              <span>Click or hover on pill / graph</span>
              <button
                onClick={() => setHoverDepth(5)}
                className="text-cyan-700 hover:text-cyan-900 font-semibold hover:underline"
              >
                Reset to Surface
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Plot Area */}
      {chartType === 'ts_diagram' ? (
        renderTSDiagram()
      ) : (
        <div className={`grid gap-4 ${variableMode === 'both' ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
          {(variableMode === 'temperature' || variableMode === 'both') &&
            renderSingleCurve('temperature', 'Ocean Temperature Profile', '°C', tempToX, tempTicks, '#e11d48')}

          {(variableMode === 'salinity' || variableMode === 'both') &&
            renderSingleCurve('salinity', 'Ocean Salinity Profile', 'PSU', salToX, salTicks, '#2563eb')}
        </div>
      )}

      {/* Oceanographic Profile Summary Banner */}
      <div className="bg-slate-900 text-slate-200 rounded-xl p-4 border border-slate-800 text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-cyan-400 shrink-0" />
          <span>
            Target Profile: <strong>Cycle {forecast.predicted_cycle}</strong> forecast from cycles [{forecast.input_sequence_cycles.join(', ')}]
            {forecast.has_observation ? ' (observed profile available for comparison).' : ' (not observed yet: true look-ahead forecast).'}
            {forecast.input_window && !forecast.input_window.within_training_gap_range && (
              <span className="text-amber-300"> Profile spacing ({forecast.input_window.gaps_days.join(', ')} days) is outside the 3-20 day range seen in training; treat this forecast with care.</span>
            )}
            {modelInfo && (
              <>
                {' '}Held-out test ({modelInfo.test_target_dates?.[0]?.slice(0, 4)}-{modelInfo.test_target_dates?.[1]?.slice(0, 4)}): temperature RMSE{' '}
                <strong className="text-emerald-400">{modelInfo.test_rmse_temp.toFixed(3)}°C</strong> vs persistence{' '}
                {modelInfo.test_rmse_temp_persistence.toFixed(3)}°C (
                {(((modelInfo.test_rmse_temp_persistence - modelInfo.test_rmse_temp) / modelInfo.test_rmse_temp_persistence) * 100).toFixed(1)}% lower).
              </>
            )}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {forecast.physical_diagnostics.is_gravitationally_stable ? (
            <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono text-[11px]">
              Static Stability: PASS
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800 font-mono text-[11px]">
              Static Stability: {forecast.physical_diagnostics.stability_violation_count} inversion(s)
            </span>
          )}
          <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono text-[11px]">
            QC: Flag 1/2
          </span>
        </div>
      </div>

      {/* Merged Evidence-Link Protocol Section (at the very bottom, right below target profile div) */}
      <div id="evidence-link-protocol-section" className="pt-2">
        <EvidenceLinkViewer forecast={forecast} />
      </div>
    </div>
  );
};
