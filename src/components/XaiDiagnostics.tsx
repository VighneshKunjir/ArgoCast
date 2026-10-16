import React, { useMemo, useState } from 'react';
import { ForecastResult } from '../types.ts';
import { ShieldCheck, Sparkles, BarChart3, Grid, CheckCircle2, AlertTriangle, Sigma } from 'lucide-react';

interface XaiDiagnosticsProps {
  forecast: ForecastResult;
}

const DEPTHS = [5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000];

// Sequential single-hue ramp (light -> dark indigo) for attribution share.
const RAMP = ['#eef2ff', '#e0e7ff', '#c7d2fe', '#a5b4fc', '#818cf8', '#6366f1', '#4f46e5', '#4338ca', '#3730a3'];
const rampColor = (t: number) => RAMP[Math.min(RAMP.length - 1, Math.max(0, Math.floor(t * RAMP.length)))];

const pct = (v: number | undefined | null, d = 1) => (v == null ? '--' : `${(v * 100).toFixed(d)}%`);

export const XaiDiagnostics: React.FC<XaiDiagnosticsProps> = ({ forecast }) => {
  const { xai_attribution: xai, physical_diagnostics: phys } = forecast;
  const [cell, setCell] = useState<{ inD: number; outD: number; w: number } | null>(null);

  const matrix = useMemo(() => {
    const m: number[][] = DEPTHS.map(() => DEPTHS.map(() => 0));
    for (const e of xai.depth_attribution_matrix) {
      const i = DEPTHS.indexOf(e.input_depth);
      const j = DEPTHS.indexOf(e.output_depth);
      if (i >= 0 && j >= 0) m[i][j] = e.saliency_weight;
    }
    return m;
  }, [xai.depth_attribution_matrix]);
  const maxW = Math.max(...matrix.flat(), 1e-9);

  const target = xai.target ? `${xai.target.variable} at ${xai.target.depth_dbar} dbar` : 'temperature at 100 dbar';
  const matrixVar = xai.matrix_variable ?? 'temperature';
  const latest = xai.temporal_attribution.find((a) => a.cycle_offset === -1);
  const stable = phys.is_gravitationally_stable;
  const pairs = phys.level_pairs_checked ?? 15;
  const mldResolved = phys.mld_resolved ?? true;

  return (
    <div className="space-y-6">
      {/* ------------------------------------------------------------- banner */}
      <div className="bg-linear-to-r from-slate-900 to-indigo-950 text-white rounded-xl p-5 border border-slate-800 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="p-1 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-400/30">
                <Sparkles className="w-4 h-4" />
              </span>
              <h2 className="text-base font-bold tracking-tight">Explanation (Integrated Gradients) and TEOS-10 Physics</h2>
            </div>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Which past profiles and which depths drove this forecast, measured with Integrated Gradients, and whether
              the forecast profile is physically stable, checked with the TEOS-10 equation of state (gsw).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className={`rounded-lg p-3 text-center min-w-[120px] border ${stable ? 'bg-emerald-950/80 border-emerald-700/80' : 'bg-amber-950/80 border-amber-700/80'}`}>
              <div className={`text-[10px] uppercase font-bold tracking-wider ${stable ? 'text-emerald-400' : 'text-amber-400'}`}>Static stability</div>
              <div className="text-base font-bold text-white flex items-center justify-center gap-1 mt-0.5">
                {stable ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <AlertTriangle className="w-4 h-4 text-amber-400" />}
                {stable ? 'PASS' : `${phys.stability_violation_count} of ${pairs}`}
              </div>
              <div className={`text-[10px] ${stable ? 'text-emerald-300' : 'text-amber-300'}`}>
                {stable ? '∂σθ/∂z ≥ 0 at all levels' : 'level pairs with ∂σθ/∂z < 0'}
              </div>
            </div>

            <div className="bg-indigo-950/80 border border-indigo-700/80 rounded-lg p-3 text-center min-w-[120px]">
              <div className="text-[10px] uppercase font-bold text-indigo-300 tracking-wider">Mixed layer depth</div>
              <div className="text-base font-bold text-white mt-0.5">
                {mldResolved ? (
                  <>
                    {phys.mixed_layer_depth_m.toFixed(0)} <span className="text-xs font-normal text-slate-300">dbar</span>
                  </>
                ) : (
                  <span className="text-sm">&gt; 1000 dbar</span>
                )}
              </div>
              <div className="text-[10px] text-indigo-300">Δσθ = 0.03 kg/m³ from 5 dbar</div>
            </div>

            <div className="bg-slate-800/80 border border-slate-600/80 rounded-lg p-3 text-center min-w-[120px]">
              <div className="text-[10px] uppercase font-bold text-slate-300 tracking-wider">IG completeness</div>
              <div className="text-base font-bold text-white mt-0.5 font-mono">
                {xai.convergence_delta != null ? xai.convergence_delta.toExponential(1) : '--'}
              </div>
              <div className="text-[10px] text-slate-400">Σ attributions vs output change</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* -------------------------------------------------- temporal attribution */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-1">
            <BarChart3 className="w-4 h-4 text-cyan-600" />
            Which past profile matters
          </h3>
          <p className="text-xs text-slate-600 mb-4 leading-relaxed">
            Share of the Integrated Gradients attribution for the forecast <strong>{target}</strong> carried by each input
            profile (all 16 temperatures and 16 salinities of that profile). Reference input: the training-mean profile.
          </p>

          <div className="space-y-3">
            {xai.temporal_attribution.map((item) => {
              const w = (item.importance_score ?? 0) * 100;
              return (
                <div key={item.cycle_offset} className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-800 mb-1.5">
                    <span>{item.cycle_label}</span>
                    <span className="font-mono text-slate-900">{w.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                    <div className="bg-cyan-600 h-2.5 rounded-full transition-all duration-500" style={{ width: `${w}%` }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-4 p-3 bg-slate-100 rounded-lg text-xs text-slate-700 border border-slate-200 space-y-1.5 leading-relaxed">
            {latest && (
              <p>
                The most recent profile (cycle {forecast.input_sequence_cycles[2]}) carries{' '}
                <strong>{pct(latest.importance_score)}</strong> of the attribution; the two older profiles carry the rest.
              </p>
            )}
            {xai.top_input && (
              <p>
                Largest single input: <strong>{xai.top_input.variable} at {xai.top_input.depth_dbar} dbar</strong> of the t
                {xai.top_input.cycle_offset} profile{xai.top_input.cycle != null ? ` (cycle ${xai.top_input.cycle})` : ''}, with{' '}
                {pct(xai.top_input.share)} of the total.
              </p>
            )}
            {xai.cross_variable_share != null && (
              <p>
                Across all depths, salinity inputs supply {pct(xai.cross_variable_share, 0)} of the attribution for the{' '}
                {matrixVar} forecast, so the model uses both variables.
              </p>
            )}
          </div>
        </div>

        {/* ------------------------------------------------------- TEOS-10 physics */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-1">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            TEOS-10 stability of the forecast profile
          </h3>
          <p className="text-xs text-slate-600 mb-4 leading-relaxed">
            Potential density σθ is computed with gsw at the float position. A stable water column has density increasing
            with depth (∂σθ/∂z ≥ 0); a decrease means heavier water sits above lighter water.
          </p>

          <div className="grid grid-cols-2 gap-3 mb-4 text-xs">
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
              <span className="text-slate-500 block text-[11px]">σθ at 5 dbar</span>
              <span className="font-mono font-bold text-slate-900 text-sm">{phys.surface_potential_density?.toFixed(2) ?? '--'} kg/m³</span>
            </div>
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
              <span className="text-slate-500 block text-[11px]">σθ at 1000 dbar</span>
              <span className="font-mono font-bold text-slate-900 text-sm">{phys.deep_potential_density?.toFixed(2) ?? '--'} kg/m³</span>
            </div>
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
              <span className="text-slate-500 block text-[11px]">Steepest temperature gradient</span>
              <span className="font-mono font-bold text-slate-900 text-sm">{phys.max_thermocline_gradient?.toFixed(3) ?? '--'} °C/dbar</span>
              <span className="text-[10px] text-slate-500 block mt-0.5">around {phys.thermocline_depth_m} dbar</span>
            </div>
            <div className={`p-3 rounded-lg border ${stable ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
              <span className="text-slate-500 block text-[11px]">Density inversions</span>
              <span className={`font-mono font-bold text-sm ${stable ? 'text-emerald-700' : 'text-amber-700'}`}>
                {phys.stability_violation_count} of {pairs} level pairs
              </span>
              <span className="text-[10px] text-slate-600 block mt-0.5">
                {stable
                  ? 'No decrease of density with depth'
                  : `Largest drop ${phys.max_density_drop != null ? phys.max_density_drop.toFixed(3) : '--'} kg/m³; ${phys.inversions_above_0_03 ?? 0} above 0.03 kg/m³`}
              </span>
            </div>
          </div>

          <div className="border border-slate-200 rounded-lg overflow-hidden">
            <table className="min-w-full divide-y divide-slate-200 text-[11px]">
              <thead className="bg-slate-100 font-semibold text-slate-700">
                <tr>
                  <th className="px-3 py-1.5 text-left">Depth (dbar)</th>
                  <th className="px-3 py-1.5 text-left">σθ (kg/m³)</th>
                  <th className="px-3 py-1.5 text-left">N² of layer above (s⁻²)</th>
                  <th className="px-3 py-1.5 text-right">Layer above</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {phys.density_profile.map((dp, k) => {
                  const first = k === 0;
                  const ok = dp.buoyancy_frequency_n2 > 0;
                  return (
                    <tr key={dp.depth_dbar} className="hover:bg-slate-50">
                      <td className="px-3 py-1 text-slate-900">{dp.depth_dbar}</td>
                      <td className="px-3 py-1 text-slate-800 font-semibold">{dp.sigma_theta?.toFixed(3) ?? '--'}</td>
                      <td className="px-3 py-1 text-slate-600">{first ? '--' : dp.buoyancy_frequency_n2.toExponential(2)}</td>
                      <td className={`px-3 py-1 text-right font-bold ${first ? 'text-slate-400' : ok ? 'text-emerald-600' : 'text-amber-600'}`}>
                        {first ? '--' : ok ? 'stable' : 'neutral / unstable'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
            N² = (g/ρ₀)·∂σθ/∂z between the listed depth and the standard level just above it. The stability check above uses all 16 levels.
          </p>
        </div>
      </div>

      {/* ------------------------------------------------------- depth matrix */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-1">
          <Grid className="w-4 h-4 text-indigo-600" />
          Which input depths drive each forecast depth ({matrixVar})
        </h3>
        <p className="text-xs text-slate-600 mb-4 leading-relaxed">
          One Integrated Gradients run per forecast depth. Each column is a forecast depth and shows how its attribution
          from {matrixVar} inputs is shared across the 16 input depths (three past profiles summed); every column adds up to 100%.
          Click a cell for its value.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          <div className="lg:col-span-2 overflow-x-auto">
            <div className="min-w-[520px]">
              <div className="text-[11px] font-semibold text-slate-600 text-center mb-1 pl-12">Forecast depth (dbar) →</div>
              <div className="flex">
                <div className="flex flex-col justify-center mr-1">
                  <span className="text-[11px] font-semibold text-slate-600 [writing-mode:vertical-rl] rotate-180">Input depth (dbar)</span>
                </div>
                <div className="flex-1">
                  <div className="grid gap-px mb-px" style={{ gridTemplateColumns: `36px repeat(${DEPTHS.length}, minmax(0, 1fr))` }}>
                    <div />
                    {DEPTHS.map((d) => (
                      <div key={`c${d}`} className="text-[9px] font-mono text-slate-500 text-center">{d}</div>
                    ))}
                  </div>
                  {DEPTHS.map((inD, i) => (
                    <div key={`r${inD}`} className="grid gap-px mb-px" style={{ gridTemplateColumns: `36px repeat(${DEPTHS.length}, minmax(0, 1fr))` }}>
                      <div className="text-[9px] font-mono text-slate-500 text-right pr-1 self-center">{inD}</div>
                      {DEPTHS.map((outD, j) => {
                        const w = matrix[i][j];
                        const sel = cell?.inD === inD && cell?.outD === outD;
                        return (
                          <button
                            key={`${inD}-${outD}`}
                            type="button"
                            title={`${(w * 100).toFixed(1)}% of forecast ${matrixVar} at ${outD} dbar from input ${matrixVar} at ${inD} dbar`}
                            onClick={() => setCell({ inD, outD, w })}
                            className={`h-6 rounded-[2px] ${sel ? 'ring-2 ring-cyan-500 relative z-10' : ''}`}
                            style={{ background: rampColor(w / maxW) }}
                          />
                        );
                      })}
                    </div>
                  ))}
                  <div className="flex items-center gap-2 mt-3 pl-9 text-[10px] text-slate-500">
                    <span>0%</span>
                    <div className="h-2.5 w-40 rounded-sm" style={{ background: `linear-gradient(to right, ${RAMP.join(', ')})` }} />
                    <span>{(maxW * 100).toFixed(0)}% share</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            {cell && (
              <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3 text-xs text-cyan-900">
                <div className="font-bold mb-1">Input {cell.inD} dbar → forecast {cell.outD} dbar</div>
                <div>
                  <strong>{(cell.w * 100).toFixed(1)}%</strong> of the attribution of the forecast {matrixVar} at {cell.outD} dbar
                  comes from the input {matrixVar} at {cell.inD} dbar (three past profiles summed).
                </div>
              </div>
            )}

            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <Sigma className="w-3.5 h-3.5" /> Strongest input zone per forecast zone
            </h4>
            {xai.key_depth_influences.map((inf, i) => (
              <div key={i} className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-xs">
                <div className="font-bold text-slate-900 flex items-center justify-between mb-1">
                  <span>{inf.target_zone}</span>
                  <span className="text-slate-900 font-mono font-semibold">{inf.attribution_percentage.toFixed(1)}%</span>
                </div>
                <div className="text-[11px] text-slate-600 mb-1">
                  <strong className="text-slate-700">From:</strong> {inf.dominant_input_depth}
                </div>
                <div className="text-[11px] text-slate-500 leading-snug">{inf.scientific_driver}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
