import React, { useState } from 'react';
import { EvidenceCitation, ForecastResult } from '../types.ts';
import { Database, ShieldCheck, ExternalLink, ArrowRight, FileCode, CheckCircle, Eye, X } from 'lucide-react';

interface EvidenceLinkViewerProps {
  forecast: ForecastResult;
}

export const EvidenceLinkViewer: React.FC<EvidenceLinkViewerProps> = ({ forecast }) => {
  const [selectedCitation, setSelectedCitation] = useState<EvidenceCitation | null>(null);

  return (
    <div className="space-y-6">
      {/* Evidence Link Protocol Pipeline Card */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Database className="w-5 h-5 text-cyan-600" />
              Evidence-Link Protocol Architecture
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Preserves uncompromised data lineage from raw GDAC NetCDF archives to the final predicted profile.
            </p>
          </div>
          <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5" /> QC Strict Policy (Flags 1 & 2)
          </span>
        </div>

        {/* Provenance Pipeline Flow Steps */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs mb-2">
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="font-bold text-slate-800 flex items-center gap-1 mb-1">
              <span className="w-5 h-5 rounded-full bg-slate-200 flex items-center justify-center text-[10px]">1</span>
              Raw GDAC NetCDF
            </div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              Read-only immutable NetCDF mirror files preserved directly from official Argo assembly centres.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="font-bold text-slate-800 flex items-center gap-1 mb-1">
              <span className="w-5 h-5 rounded-full bg-slate-200 flex items-center justify-center text-[10px]">2</span>
              Point-Level QC Audit
            </div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              Strictly filters out QC flags 3 ("bad") and 4 ("definitely bad"). Only flags 1 & 2 admitted.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="font-bold text-slate-800 flex items-center gap-1 mb-1">
              <span className="w-5 h-5 rounded-full bg-slate-200 flex items-center justify-center text-[10px]">3</span>
              Vector Analogy Ranking
            </div>
            <p className="text-slate-500 text-[11px] leading-relaxed">
              Computes multi-dimensional profile cosine similarity + haversine spatial decay weight.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-cyan-50 border border-cyan-200">
            <div className="font-bold text-cyan-900 flex items-center gap-1 mb-1">
              <span className="w-5 h-5 rounded-full bg-cyan-200 text-cyan-900 flex items-center justify-center text-[10px]">4</span>
              Forecast Evidence Link
            </div>
            <p className="text-cyan-800 text-[11px] leading-relaxed">
              Citations of real QC-pass profiles that best match the forecast for cycle {forecast.predicted_cycle}, with their GDAC file paths.
            </p>
          </div>
        </div>
      </div>

      {/* Verified Evidence Cards */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
            Top Grounded Historical Evidence Citations ({forecast.evidence_citations.length})
          </h3>
          <span className="text-xs text-slate-500">
            Target Float: <strong>WMO {forecast.target_float_id}</strong> (Arabian Sea)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {forecast.evidence_citations.map((evidence, idx) => {
            const similarityPct = (evidence.similarity_score * 100).toFixed(1);
            return (
              <div
                key={evidence.citation_id}
                className="bg-white rounded-xl border border-slate-200 hover:border-cyan-400 p-4 shadow-xs transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-800">
                      Citation #{idx + 1}
                    </span>
                    <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> QC PASS (Flag {evidence.qc_flag})
                    </span>
                  </div>

                  <h4 className="text-base font-bold text-slate-900 mb-1">
                    Float WMO {evidence.wmo_id}
                  </h4>
                  <div className="text-xs text-slate-500 mb-2">
                    Cycle <strong>#{evidence.cycle_number}</strong> • Observed: {evidence.date}
                  </div>

                  {/* Metrics Badges */}
                  <div className="grid grid-cols-2 gap-2 my-3 text-xs">
                    <div className="bg-slate-50 p-2 rounded border border-slate-200">
                      <span className="text-slate-400 block text-[10px]">Analogy Score</span>
                      <span className="font-mono font-bold text-cyan-700">{similarityPct}%</span>
                    </div>
                    <div className="bg-slate-50 p-2 rounded border border-slate-200">
                      <span className="text-slate-400 block text-[10px]">Distance</span>
                      <span className="font-mono font-bold text-slate-800">{evidence.distance_km} km</span>
                    </div>
                  </div>

                  <p className="text-xs text-slate-600 mb-3 line-clamp-3">
                    {evidence.key_feature_relevance}
                  </p>
                </div>

                <div className="pt-3 border-t border-slate-100 space-y-2">
                  <div className="flex items-center gap-1.5 text-[11px] text-slate-500 font-mono truncate">
                    <FileCode className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    <span className="truncate" title={evidence.raw_netcdf_file}>
                      {evidence.raw_netcdf_file}
                    </span>
                  </div>

                  <button
                    onClick={() => setSelectedCitation(evidence)}
                    className="w-full py-1.5 px-3 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
                  >
                    <Eye className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Inspect Profile & Lineage</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Modal: Detailed Inspection of Cited Historical Profile */}
      {selectedCitation && (
        <div className="fixed inset-0 bg-slate-950/60 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-200 shadow-2xl p-6">
            <div className="flex items-center justify-between pb-4 border-b border-slate-200">
              <div>
                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                  <span>Evidence Inspection: Float {selectedCitation.wmo_id}</span>
                  <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-cyan-100 text-cyan-800">
                    Cycle #{selectedCitation.cycle_number}
                  </span>
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Observed at {selectedCitation.latitude != null ? selectedCitation.latitude.toFixed(3) : '--'}°N, {selectedCitation.longitude != null ? selectedCitation.longitude.toFixed(3) : '--'}°E on {selectedCitation.date}
                </p>
              </div>
              <button
                onClick={() => setSelectedCitation(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Provenance Lineage metadata */}
            <div className="my-4 p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-2">
              <div className="font-bold text-slate-800">Source Provenance Lineage:</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-slate-600 font-mono text-[11px]">
                <div>
                  <span className="text-slate-400 block">Raw NetCDF File:</span>
                  <strong>{selectedCitation.raw_netcdf_file}</strong>
                </div>
                <div>
                  <span className="text-slate-400 block">GDAC Archive Path:</span>
                  <span className="truncate block" title={selectedCitation.provenance_chain?.archive_gdac ?? selectedCitation.gdac_archive_path ?? ''}>
                    {selectedCitation.provenance_chain?.archive_gdac ?? selectedCitation.gdac_archive_path ?? '—'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">QC Status:</span>
                  <span className="text-emerald-700 font-bold">Flag {selectedCitation.qc_flag} (Delayed-Mode Certified)</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Spatial Offset:</span>
                  <span>{selectedCitation.distance_km} km from target float</span>
                </div>
              </div>
            </div>

            {/* Side-by-side vertical profile table */}
            <div className="border border-slate-200 rounded-xl overflow-hidden mb-4">
              <div className="bg-slate-100 px-3 py-2 text-xs font-bold text-slate-800 border-b border-slate-200">
                Depth-Resolved Observations (Evidence vs Forecasted Profile)
              </div>
              <div className="max-h-60 overflow-y-auto">
                <table className="min-w-full divide-y divide-slate-200 text-xs font-mono">
                  <thead className="bg-slate-50 text-slate-600 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left">Depth</th>
                      <th className="px-3 py-2 text-left">Evidence T (°C)</th>
                      <th className="px-3 py-2 text-left">Forecast T (°C)</th>
                      <th className="px-3 py-2 text-left">Evidence S (PSU)</th>
                      <th className="px-3 py-2 text-right">QC Flag</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {(selectedCitation.measurements ?? []).map((m, idx) => {
                      const forecastPt = forecast.profiles[idx];
                      return (
                        <tr key={m.depth_dbar} className="hover:bg-slate-50">
                          <td className="px-3 py-1.5 font-bold text-slate-900">{m.depth_dbar} dbar</td>
                          <td className="px-3 py-1.5 text-rose-700 font-semibold">{m.temperature != null ? m.temperature.toFixed(2) : '--'}</td>
                          <td className="px-3 py-1.5 text-slate-700">{forecastPt?.temperature_forecast != null ? forecastPt.temperature_forecast.toFixed(2) : '--'}</td>
                          <td className="px-3 py-1.5 text-blue-700 font-semibold">{m.salinity != null ? m.salinity.toFixed(2) : '--'}</td>
                          <td className="px-3 py-1.5 text-right text-emerald-600 font-bold">PASS ({m.qc_temperature})</td>
                        </tr>
                      );
                    })}
                    {(selectedCitation.measurements ?? []).length === 0 && (
                      <tr>
                        <td colSpan={5} className="px-3 py-4 text-center text-slate-500">
                          No depth-resolved levels stored for this citation.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={() => setSelectedCitation(null)}
                className="px-4 py-2 bg-slate-900 text-white text-xs font-semibold rounded-lg hover:bg-slate-800"
              >
                Close Inspection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
