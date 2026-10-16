import React, { useEffect, useMemo, useState } from 'react';
import { BenchmarkResults, BenchmarkRow, ForecastResult } from '../types.ts';
import { apiService } from '../services/api.ts';
import {
  Award, ShieldCheck, Target, TrendingUp, AlertTriangle, Layers, BookOpen, FlaskConical, Brain, Loader2,
} from 'lucide-react';
import { DepthLineChart, HBarChart, NEUTRAL, ReliabilityChart, ScatterChart, SERIES } from './benchmark/charts.tsx';

interface EvaluationBenchmarksProps {
  forecast: ForecastResult;
}

type SplitKey = 'chronological' | 'unseen_floats';
const PROPOSED = 'FloatChat PI-BiLSTM';
const NO_PHYS = 'Bi-LSTM (no physics loss)';
const CLIM = ['Climatology (regional)', 'Climatology (local)'];

const f3 = (v: number | null | undefined) => (v == null ? '--' : v.toFixed(3));
const f2 = (v: number | null | undefined) => (v == null ? '--' : v.toFixed(2));
const f1 = (v: number | null | undefined) => (v == null ? '--' : v.toFixed(1));
const pct = (v: number | null | undefined, d = 1) => (v == null ? '--' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`);

const GLOSSARY: [string, string][] = [
  ['RMSE', 'Root mean square error: the typical size of a forecast error, in °C for temperature and PSU for salinity. Lower is better.'],
  ['Skill vs persistence', 'How much smaller the error is than simply repeating the last profile. +10% means 10% smaller error.'],
  ['Density inversion', 'Heavier water forecast above lighter water (TEOS-10 potential density decreasing with depth). The real ocean removes these quickly, so they are physics errors. Counted when the drop exceeds 0.03 kg/m³ between neighbouring levels.'],
  ['Mixed layer depth (MLD)', 'Depth of the well-stirred surface layer: first level where density is 0.03 kg/m³ above its 5 dbar value.'],
  ['D20', 'Depth of the 20 °C isotherm, a marker of the thermocline.'],
  ['OHC 0-700 m', 'Heat stored in the upper 700 m of the ocean (GJ/m²).'],
  ['95% coverage', 'Share of observations that fall inside the forecast 95% uncertainty interval. Close to 95% is ideal.'],
  ['CRPS', 'Continuous ranked probability score: judges the forecast and its uncertainty together (°C). Lower is better.'],
  ['ACC', 'Anomaly correlation: correlation between forecast and observed departures from climatology. Higher is better.'],
  ['Faithfulness', 'For explanation methods: how well the attribution predicts the change in output when inputs are removed. Closer to 1 is better.'],
];

type SortKey = 'temp_rmse' | 'sal_rmse' | 'skill_vs_persistence' | 'therm_rmse' | 'inv_003_pct' | 'inv_any_pct' | 'mld_rmse' | 'acc_T' | 'crps_point_T';
const COLUMNS: { key: SortKey; label: string; fmt: (r: BenchmarkRow) => string; higherBetter?: boolean }[] = [
  { key: 'temp_rmse', label: 'Temp RMSE (°C)', fmt: (r) => f3(r.temp_rmse) },
  { key: 'sal_rmse', label: 'Sal RMSE (PSU)', fmt: (r) => f3(r.sal_rmse) },
  { key: 'skill_vs_persistence', label: 'Skill vs persistence', fmt: (r) => pct(r.skill_vs_persistence), higherBetter: true },
  { key: 'therm_rmse', label: 'Thermocline RMSE (°C)', fmt: (r) => f3(r.therm_rmse) },
  { key: 'acc_T', label: 'ACC (T)', fmt: (r) => f3(r.acc_T), higherBetter: true },
  { key: 'crps_point_T', label: 'CRPS (°C)', fmt: (r) => f3(r.crps_point_T) },
  { key: 'inv_003_pct', label: 'Inversions > 0.03 (%)', fmt: (r) => f2(r.inv_003_pct) },
  { key: 'inv_any_pct', label: 'Any inversion (%)', fmt: (r) => f1(r.inv_any_pct) },
  { key: 'mld_rmse', label: 'MLD RMSE (m)', fmt: (r) => f1(r.mld_rmse) },
];

function Card({ title, icon, subtitle, children }: { title: string; icon?: React.ReactNode; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="bg-white rounded-xl border border-slate-200 p-4 sm:p-5 shadow-xs">
      <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
        {icon}
        {title}
      </h3>
      {subtitle && <p className="text-[11px] text-slate-500 mt-0.5 mb-3 leading-relaxed">{subtitle}</p>}
      {!subtitle && <div className="mb-3" />}
      {children}
    </section>
  );
}

function Stat({ icon, label, value, detail }: { icon: React.ReactNode; label: string; value: string; detail: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
        {icon}
        {label}
      </div>
      <div className="mt-1.5 text-2xl font-bold font-mono text-slate-900">{value}</div>
      <div className="mt-1 text-[11px] leading-relaxed text-slate-600">{detail}</div>
    </div>
  );
}

export const EvaluationBenchmarks: React.FC<EvaluationBenchmarksProps> = ({ forecast }) => {
  const [data, setData] = useState<BenchmarkResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState<SplitKey>('chronological');
  const [sortKey, setSortKey] = useState<SortKey>('temp_rmse');
  const [sortAsc, setSortAsc] = useState(true);

  useEffect(() => {
    apiService.getBenchmark().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  const view = useMemo(() => {
    if (!data) return null;
    const s = data.splits[split];
    const rows = s.rows;
    const names = data.models.map((m) => m.name);
    const short = Object.fromEntries(data.models.map((m) => [m.name, m.short]));
    const learned = names.filter((n) => n !== 'Persistence' && !CLIM.includes(n) && n !== PROPOSED);
    const fc = rows[PROPOSED];
    const best = [...names].sort((a, b) => rows[a].temp_rmse - rows[b].temp_rmse)[0];
    const invOthers = learned.map((n) => rows[n].inv_003_pct);
    return { s, rows, names, short, learned, fc, best, invMin: Math.min(...invOthers), invMax: Math.max(...invOthers) };
  }, [data, split]);

  if (error) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-800 flex items-center gap-2">
        <AlertTriangle className="w-4 h-4" /> {error}
      </div>
    );
  }
  if (!data || !view) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 text-xs text-slate-500 flex items-center justify-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" /> Loading benchmark results...
      </div>
    );
  }

  const { s, rows, names, short, learned, fc, best, invMin, invMax } = view;
  const persistence = rows['Persistence'];
  const noPhys = rows[NO_PHYS];
  const uqFc = s.uq[PROPOSED];
  const sorted = [...names].sort((a, b) => {
    const va = rows[a][sortKey] ?? Infinity;
    const vb = rows[b][sortKey] ?? Infinity;
    return sortAsc ? (va as number) - (vb as number) : (vb as number) - (va as number);
  });
  const bestOf = (key: SortKey, higher?: boolean) => {
    // climatologies are long-term averages (smooth by construction), so they do not compete
    const vals = names.filter((n) => !CLIM.includes(n)).map((n) => rows[n][key]).filter((v): v is number => v != null);
    return higher ? Math.max(...vals) : Math.min(...vals);
  };
  const tipRow = (n: string) => [
    n,
    `Temp RMSE ${f3(rows[n].temp_rmse)} °C (95% CI ${f3(rows[n].temp_rmse_ci[0])}-${f3(rows[n].temp_rmse_ci[1])})`,
    `Salinity RMSE ${f3(rows[n].sal_rmse)} PSU`,
    `Inversions > 0.03: ${f2(rows[n].inv_003_pct)}%`,
  ];
  const byRmse = [...names].sort((a, b) => rows[a].temp_rmse - rows[b].temp_rmse);
  const byInv = [...names].sort((a, b) => rows[a].inv_003_pct - rows[b].inv_003_pct);
  const depthSeries = [PROPOSED, 'Persistence', 'LightGBM', NO_PHYS].map((n, i) => ({
    name: short[n], color: SERIES[i], values: s.per_depth_temp_rmse[n],
  }));
  const seasons = Object.keys(s.season_temp_rmse[PROPOSED] || {});
  const mld = data.mld_fix;
  const xai = data.xai;

  return (
    <div className="space-y-5">
      {/* ---------------------------------------------------------------- header */}
      <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Award className="w-5 h-5 text-indigo-600" /> Model Benchmarks: {data.models.length} forecasting models
            </h2>
            <p className="text-xs text-slate-600 mt-1 max-w-3xl leading-relaxed">
              {data.dataset.task}. Every model gets the same input and is scored on the same held-out forecasts.
              Data: {data.dataset.n_profiles.toLocaleString()} quality-controlled Argo profiles from {data.dataset.n_floats} floats,
              {' '}{data.dataset.region}, {data.dataset.date_range[0].slice(0, 4)}-{data.dataset.date_range[1].slice(0, 4)}, {data.dataset.levels}.
              Neural models were trained with 3 seeds; the values shown are for seed 42 and confidence intervals come from a float-level bootstrap.
            </p>
          </div>
          <div className="shrink-0">
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Test set</div>
            <div className="inline-flex rounded-lg border border-slate-300 bg-slate-50 p-0.5 text-xs">
              {(['chronological', 'unseen_floats'] as SplitKey[]).map((k) => (
                <button
                  key={k}
                  onClick={() => setSplit(k)}
                  className={`px-3 py-1.5 rounded-md font-medium transition-colors ${split === k ? 'bg-white shadow-xs text-slate-900' : 'text-slate-500 hover:text-slate-800'}`}
                >
                  {data.splits[k].label}
                </button>
              ))}
            </div>
            <div className="text-[11px] text-slate-500 mt-1.5 max-w-xs">
              {s.description} {s.n_windows.test.toLocaleString()} test forecasts from {s.n_floats.test} floats.
            </div>
          </div>
        </div>
      </section>

      {/* ----------------------------------------------------------- key findings */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <Stat
          icon={<TrendingUp className="w-3.5 h-3.5 text-emerald-600" />}
          label="Accuracy"
          value={`${f3(fc.temp_rmse)} °C`}
          detail={<>Temperature RMSE, {pct(fc.skill_vs_persistence)} vs persistence ({f3(persistence.temp_rmse)} °C). Most accurate model: {short[best]} ({f3(rows[best].temp_rmse)} °C).</>}
        />
        <Stat
          icon={<ShieldCheck className="w-3.5 h-3.5 text-cyan-600" />}
          label="Physical stability"
          value={`${f2(fc.inv_003_pct)}%`}
          detail={<>Forecasts with a density inversion &gt; 0.03 kg/m³. Other learned models: {f1(invMin)}-{f1(invMax)}%. Observed profiles: {f2(s.observed.inv_003_pct)}%.</>}
        />
        <Stat
          icon={<Target className="w-3.5 h-3.5 text-indigo-600" />}
          label="Honest uncertainty"
          value={uqFc ? `${f1(uqFc.coverage95_cal)}%` : '--'}
          detail={<>Observations inside the 95% interval after one validation-fitted scale factor (raw dropout: {uqFc ? f1(uqFc.coverage95_raw) : '--'}%).</>}
        />
        <Stat
          icon={<AlertTriangle className="w-3.5 h-3.5 text-amber-600" />}
          label="Known trade-off"
          value={`+${f1(fc.mld_rmse - noPhys.mld_rmse)} m`}
          detail={<>Extra mixed-layer-depth error vs the same network without the physics loss ({f1(fc.mld_rmse)} vs {f1(noPhys.mld_rmse)} m); forecasts are too shallow.</>}
        />
      </div>

      {/* --------------------------------------------------------- this forecast */}
      <Card
        title={`This forecast: float ${forecast.target_float_id}, cycle ${forecast.predicted_cycle}`}
        icon={<Layers className="w-4 h-4 text-cyan-600" />}
        subtitle="Scores of the forecast currently loaded, against the observed profile of the target cycle. A single forecast is noisy; the study below is what measures the models."
      >
        {forecast.metrics_comparison.length === 0 ? (
          <p className="text-xs text-slate-600">
            Cycle {forecast.predicted_cycle} has not been observed yet, so this is a true look-ahead forecast and cannot be scored.
            Pick an earlier base cycle to compare against an observed profile.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50 text-slate-600">
                <tr>
                  {['Model', 'Temp RMSE (°C)', 'Temp MAE (°C)', 'Thermocline 50-200 dbar (°C)', 'Deep 500-1000 dbar (°C)', 'Salinity RMSE (PSU)', 'Levels with inversion (%)'].map((h, i) => (
                    <th key={h} className={`px-3 py-2 font-semibold ${i === 0 ? 'text-left' : 'text-right'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {forecast.metrics_comparison.map((m) => {
                  const ours = m.model.includes('FloatChat');
                  return (
                    <tr key={m.model} className={ours ? 'bg-cyan-50/60 font-semibold' : ''}>
                      <td className="px-3 py-2 text-left font-sans text-slate-900">{m.model}</td>
                      <td className="px-3 py-2 text-right">{f3(m.profile_rmse)}</td>
                      <td className="px-3 py-2 text-right">{f3(m.profile_mae)}</td>
                      <td className="px-3 py-2 text-right">{f3(m.thermocline_rmse)}</td>
                      <td className="px-3 py-2 text-right">{f3(m.deep_rmse)}</td>
                      <td className="px-3 py-2 text-right">{m.salinity_rmse != null ? m.salinity_rmse.toFixed(4) : '--'}</td>
                      <td className="px-3 py-2 text-right">{f1(m.physical_violation_rate)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* ------------------------------------------------------------- charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <Card
          title="Temperature error by model"
          icon={<TrendingUp className="w-4 h-4 text-emerald-600" />}
          subtitle="RMSE over all 16 levels (lower is better). The dashed line is persistence; bars marked → run past the axis (the long-term averages are much worse)."
        >
          <HBarChart
            items={byRmse.map((n) => ({ label: short[n], value: rows[n].temp_rmse, highlight: n === PROPOSED, tooltip: tipRow(n) }))}
            domain={[0.5, 0.95]}
            format={(v) => v.toFixed(3)}
            axisLabel="Temperature RMSE (°C)"
            refLine={{ value: persistence.temp_rmse, label: 'persistence' }}
          />
        </Card>
        <Card
          title="Physically impossible forecasts"
          icon={<ShieldCheck className="w-4 h-4 text-cyan-600" />}
          subtitle="Share of forecasts with a TEOS-10 density inversion larger than 0.03 kg/m³ (log scale, lower is better). The dashed line is the rate in the observed profiles."
        >
          <HBarChart
            items={byInv.map((n) => ({ label: short[n], value: rows[n].inv_003_pct, highlight: n === PROPOSED, tooltip: tipRow(n) }))}
            domain={[0.01, 20]}
            log
            format={(v) => `${v.toFixed(2)}%`}
            axisLabel="Forecasts with an inversion > 0.03 kg/m³ (%)"
            refLine={{ value: s.observed.inv_003_pct, label: 'observed' }}
          />
        </Card>
        <Card
          title="Accuracy vs physical consistency"
          icon={<Target className="w-4 h-4 text-indigo-600" />}
          subtitle="Each dot is a model (hover for details); down-left is better on both. The learned models reach similar accuracy, but only FloatChat also stays physically consistent. The two climatologies are left out (temperature RMSE above 0.8 °C)."
        >
          <ScatterChart
            points={names.filter((n) => !CLIM.includes(n)).map((n) => ({
              label: short[n],
              x: Math.max(rows[n].inv_003_pct, 0.011),
              y: rows[n].temp_rmse,
              highlight: n === PROPOSED,
              showLabel: [PROPOSED, 'Persistence', 'DLinear', 'TCN'].includes(n),
              labelSide: n === 'DLinear' ? 'left' : 'right',
              tooltip: tipRow(n),
            }))}
            xLog
            xDomain={[0.01, 20]}
            yDomain={[0.56, 0.66]}
            xLabel="Forecasts with an inversion > 0.03 kg/m³ (%, log scale)"
            yLabel="Temperature RMSE (°C)"
            annotations={[{ x: 4.2, y: 0.5865, text: 'Ridge, LightGBM and 5 neural models' }]}
          />
        </Card>
        <Card
          title="Where the errors are: RMSE by depth"
          icon={<Layers className="w-4 h-4 text-rose-600" />}
          subtitle="Temperature RMSE at each of the 16 levels. Errors peak in the thermocline (75-200 dbar), where the water column changes fastest. Hover for values."
        >
          <DepthLineChart depths={data.depths_dbar} series={depthSeries} axisLabel="Temperature RMSE (°C)" format={(v) => `${v.toFixed(3)} °C`} />
        </Card>
      </div>

      {/* -------------------------------------------------------- uncertainty */}
      <Card
        title="Uncertainty: are the forecast intervals honest?"
        icon={<Target className="w-4 h-4 text-indigo-600" />}
        subtitle="Monte Carlo dropout runs the network 50 times with dropout switched on; the spread gives the interval. Raw spread is too narrow, so one scale factor per variable, fitted on validation data, is applied."
      >
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 items-start">
          <div className="lg:col-span-2">
            <div className="text-[11px] font-semibold text-slate-700 mb-1">FloatChat PI-BiLSTM, temperature</div>
            {uqFc && <ReliabilityChart nominal={s.reliability_nominal} raw={uqFc.reliability_raw} cal={uqFc.reliability_cal} />}
          </div>
          <div className="lg:col-span-3 overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50 text-slate-600">
                <tr>
                  {['Model', '95% coverage raw', '95% coverage calibrated', 'Scale factor', 'CRPS raw (°C)', 'CRPS calibrated (°C)'].map((h, i) => (
                    <th key={h} className={`px-3 py-2 font-semibold ${i === 0 ? 'text-left' : 'text-right'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {Object.entries(s.uq).sort((a, b) => a[1].crps_cal - b[1].crps_cal).map(([n, u]) => (
                  <tr key={n} className={n === PROPOSED ? 'bg-cyan-50/60 font-semibold' : ''}>
                    <td className="px-3 py-1.5 text-left font-sans text-slate-900">{short[n]}</td>
                    <td className="px-3 py-1.5 text-right">{f1(u.coverage95_raw)}%</td>
                    <td className="px-3 py-1.5 text-right">{f1(u.coverage95_cal)}%</td>
                    <td className="px-3 py-1.5 text-right">{f2(u.scale_k)}</td>
                    <td className="px-3 py-1.5 text-right">{f3(u.crps_raw)}</td>
                    <td className="px-3 py-1.5 text-right">{f3(u.crps_cal)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[11px] text-slate-500 mt-2">
              Sorted by calibrated CRPS (lower is better). Models without dropout layers give point forecasts only and are not listed.
            </p>
          </div>
        </div>
      </Card>

      {/* --------------------------------------------------------- full table */}
      <Card
        title="All metrics"
        icon={<BookOpen className="w-4 h-4 text-slate-600" />}
        subtitle="Click a column header to sort. The best value in each column (climatologies excluded) is in bold green; FloatChat is the highlighted row. Hover a temperature RMSE to see its 95% bootstrap interval."
      >
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-slate-100 text-slate-700">
              <tr>
                <th className="px-3 py-2 text-left font-semibold">Model</th>
                <th className="px-3 py-2 text-left font-semibold">Type</th>
                {COLUMNS.map((c) => (
                  <th key={c.key} className="px-3 py-2 text-right font-semibold whitespace-nowrap">
                    <button
                      className="hover:text-slate-950"
                      onClick={() => {
                        if (sortKey === c.key) setSortAsc(!sortAsc);
                        else { setSortKey(c.key); setSortAsc(!c.higherBetter); }
                      }}
                    >
                      {c.label}{sortKey === c.key ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {sorted.map((n) => {
                const r = rows[n];
                const model = data.models.find((m) => m.name === n)!;
                return (
                  <tr key={n} className={n === PROPOSED ? 'bg-cyan-50/60' : 'hover:bg-slate-50'}>
                    <td className={`px-3 py-1.5 text-left font-sans whitespace-nowrap ${n === PROPOSED ? 'font-bold text-slate-900' : 'text-slate-800'}`}>{n}</td>
                    <td className="px-3 py-1.5 text-left font-sans text-slate-500 whitespace-nowrap">{model.family}</td>
                    {COLUMNS.map((c) => {
                      const v = r[c.key];
                      const isBest = v != null && !CLIM.includes(n) && v === bestOf(c.key, c.higherBetter);
                      return (
                        <td
                          key={c.key}
                          title={c.key === 'temp_rmse' ? `95% CI ${f3(r.temp_rmse_ci[0])}-${f3(r.temp_rmse_ci[1])}` : undefined}
                          className={`px-3 py-1.5 text-right ${isBest ? 'font-bold text-emerald-700' : 'text-slate-700'}`}
                        >
                          {c.fmt(r)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
              <tr className="text-slate-500 italic">
                <td className="px-3 py-1.5 font-sans">Observed profiles</td>
                <td className="px-3 py-1.5 font-sans">reference</td>
                <td colSpan={6} />
                <td className="px-3 py-1.5 text-right">{f2(s.observed.inv_003_pct)}</td>
                <td className="px-3 py-1.5 text-right">{f1(s.observed.inv_any_pct)}</td>
                <td />
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">
          {best === PROPOSED
            ? 'FloatChat has the lowest temperature RMSE on this test set. '
            : `FloatChat is ${f3(fc.temp_rmse - rows[best].temp_rmse)} °C behind the most accurate model (${short[best]}) on temperature RMSE. `}
          Small inversions below 0.01 kg/m³ are common in raw observations ({f1(s.observed.inv_any_pct)}% of profiles), which is why the 0.03 kg/m³ threshold is the headline number.
        </p>
      </Card>

      {/* ----------------------------------------------------- derived + season */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <Card
          title="Ocean quantities computed from the forecasts"
          icon={<FlaskConical className="w-4 h-4 text-purple-600" />}
          subtitle="Errors in quantities oceanographers use. FloatChat matches the best models for D20 and heat content; mixed layer depth is its weak point."
        >
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50 text-slate-600">
                <tr>
                  {['Model', 'MLD RMSE (m)', 'MLD bias (m)', 'D20 RMSE (m)', 'OHC RMSE (GJ/m²)'].map((h, i) => (
                    <th key={h} className={`px-3 py-2 font-semibold ${i === 0 ? 'text-left' : 'text-right'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {names.map((n) => (
                  <tr key={n} className={n === PROPOSED ? 'bg-cyan-50/60 font-semibold' : ''}>
                    <td className="px-3 py-1.5 text-left font-sans text-slate-900 whitespace-nowrap">{short[n]}</td>
                    <td className="px-3 py-1.5 text-right">{f1(rows[n].mld_rmse)}</td>
                    <td className="px-3 py-1.5 text-right">{rows[n].mld_bias > 0 ? '+' : ''}{f1(rows[n].mld_bias)}</td>
                    <td className="px-3 py-1.5 text-right">{f1(rows[n].d20_rmse)}</td>
                    <td className="px-3 py-1.5 text-right">{f3(rows[n].ohc_rmse)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card
          title="Error by monsoon season"
          icon={<Layers className="w-4 h-4 text-amber-600" />}
          subtitle="Temperature RMSE (°C) per season. All models find the summer monsoon and the autumn transition hardest."
        >
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50 text-slate-600">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold">Model</th>
                  {seasons.map((se) => (
                    <th key={se} className="px-3 py-2 text-right font-semibold">
                      {se.replace(/\s*\(.*\)/, '')}
                      <div className="font-normal text-slate-400">{s.season_counts?.[se]?.toLocaleString()} fcst</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {names.map((n) => (
                  <tr key={n} className={n === PROPOSED ? 'bg-cyan-50/60 font-semibold' : ''}>
                    <td className="px-3 py-1.5 text-left font-sans text-slate-900 whitespace-nowrap">{short[n]}</td>
                    {seasons.map((se) => (
                      <td key={se} className="px-3 py-1.5 text-right">{f3(s.season_temp_rmse[n]?.[se])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* ------------------------------------------------------- MLD experiment */}
      {mld && (
        <Card
          title="Experiment: can the mixed-layer weakness be fixed?"
          icon={<FlaskConical className="w-4 h-4 text-amber-600" />}
          subtitle={`The stability penalty makes the network keep a little stratification everywhere, so it almost never forecasts a fully mixed surface layer (observed in ${f1(mld.observed_pct_mixed_5_20[split])}% of test profiles). Eight versions of the loss were trained (3 seeds each). Mean over seeds on this test set.`}
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
            <ScatterChart
              points={mld.variants.filter((v) => v[split]).map((v) => ({
                label: v.label,
                x: Math.max(v[split]!.inv_003_pct, 0.011),
                y: v[split]!.mld_rmse,
                highlight: v.id === 'baseline',
                color: v.id === 'no_physics' ? SERIES[1] : undefined,
                showLabel: ['baseline', 'surface_match', 'tol_0.03_surface', 'no_physics', 'tol_0.03'].includes(v.id),
                labelSide: v.id === 'no_physics' ? 'left' : 'right',
                tooltip: [v.label, `MLD RMSE ${f1(v[split]!.mld_rmse)} m`, `Inversions > 0.03: ${f2(v[split]!.inv_003_pct)}%`, `Temp RMSE ${f3(v[split]!.temp_rmse)} °C`],
              }))}
              xLog
              xDomain={[0.01, 20]}
              yDomain={[15, 27]}
              xLabel="Forecasts with an inversion > 0.03 kg/m³ (%, log scale)"
              yLabel="Mixed layer depth RMSE (m)"
            />
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead className="bg-slate-50 text-slate-600">
                  <tr>
                    {['Loss version', 'MLD RMSE (m)', 'Inversions > 0.03 (%)', 'Mixed top 20 m (%)', 'Temp RMSE (°C)'].map((h, i) => (
                      <th key={h} className={`px-3 py-2 font-semibold ${i === 0 ? 'text-left' : 'text-right'}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  {mld.variants.filter((v) => v[split]).map((v) => (
                    <tr key={v.id} className={v.id === 'baseline' ? 'bg-cyan-50/60 font-semibold' : ''}>
                      <td className="px-3 py-1.5 text-left font-sans text-slate-900">{v.label}</td>
                      <td className="px-3 py-1.5 text-right">{f1(v[split]!.mld_rmse)}{v[split]!.mld_rmse_std != null ? ` ± ${f1(v[split]!.mld_rmse_std)}` : ''}</td>
                      <td className="px-3 py-1.5 text-right">{f2(v[split]!.inv_003_pct)}</td>
                      <td className="px-3 py-1.5 text-right">{f1(v[split]!.pct_mixed_5_20)}</td>
                      <td className="px-3 py-1.5 text-right">{f3(v[split]!.temp_rmse)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-[11px] text-slate-600 mt-2 leading-relaxed">
                A surface-stratification term removes most of the mixed-layer error without hurting temperature accuracy,
                but small inversions become more frequent (still far fewer than without the physics loss). The served model keeps
                the current loss because physical stability is the main goal; the trade-off can be tuned.
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* ------------------------------------------------------------------ XAI */}
      {xai && (
        <Card
          title="Explanation methods compared"
          icon={<Brain className="w-4 h-4 text-indigo-600" />}
          subtitle={`${xai.target}, ${xai.n_windows} test forecasts. FloatChat uses Integrated Gradients; LIME and KernelSHAP were run on the same forecasts for comparison.`}
        >
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50 text-slate-600">
                <tr>
                  {['Method', 'Faithfulness', 'Same answer each run', 'Stable to small input changes', 'Additivity error', 'Time per explanation', 'Share to latest cycle (t-1)'].map((h, i) => (
                    <th key={h} className={`px-3 py-2 font-semibold ${i === 0 ? 'text-left' : 'text-right'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {Object.entries(xai.methods).map(([name, m]) => (
                  <tr key={name} className={name === 'Integrated Gradients' ? 'bg-cyan-50/60 font-semibold' : ''}>
                    <td className="px-3 py-1.5 text-left font-sans text-slate-900">{name}{name === 'Integrated Gradients' ? ' (used)' : ''}</td>
                    <td className="px-3 py-1.5 text-right">{f3(m.faithfulness)}</td>
                    <td className="px-3 py-1.5 text-right">{f3(m.repeatability)}</td>
                    <td className="px-3 py-1.5 text-right">{f3(m.robustness)}</td>
                    <td className="px-3 py-1.5 text-right">{f3(m.completeness_error)}</td>
                    <td className="px-3 py-1.5 text-right">{f1(m.runtime_ms)} ms</td>
                    <td className="px-3 py-1.5 text-right">{(m.share_latest_cycle * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-slate-500 mt-2">
            Higher is better for the first three columns; additivity error is how far the attributions are from summing to the change in the forecast (0 is exact).
          </p>
        </Card>
      )}

      {/* ------------------------------------------------- models and glossary */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <Card title="The models" icon={<Layers className="w-4 h-4 text-slate-600" />}>
          <ul className="divide-y divide-slate-100 text-xs">
            {data.models.map((m) => (
              <li key={m.name} className="py-1.5 flex gap-3">
                <span
                  className="mt-1 inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ background: m.proposed ? SERIES[0] : NEUTRAL }}
                />
                <span>
                  <span className={`font-semibold ${m.proposed ? 'text-slate-950' : 'text-slate-800'}`}>{m.name}</span>
                  <span className="text-slate-400"> · {m.family}</span>
                  <span className="block text-slate-600">{m.description}</span>
                </span>
              </li>
            ))}
          </ul>
        </Card>
        <Card title="What the terms mean" icon={<BookOpen className="w-4 h-4 text-slate-600" />}>
          <dl className="divide-y divide-slate-100 text-xs">
            {GLOSSARY.map(([t, d]) => (
              <div key={t} className="py-1.5">
                <dt className="font-semibold text-slate-800">{t}</dt>
                <dd className="text-slate-600">{d}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </div>
  );
};
