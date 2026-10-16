/**
 * Small SVG chart primitives for the Benchmarks tab (no chart library).
 *
 * Conventions: thin marks (bars <= 18px, 4px rounded data end, 2px lines, r >= 4 dots
 * with a 2px surface ring), hairline solid gridlines, text in slate ink (never in the
 * series colour), a hover tooltip on every mark, and the proposed model highlighted
 * in blue while baselines stay neutral.
 */
import React, { useState } from 'react';

export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];
export const HIGHLIGHT = '#2a78d6';
export const NEUTRAL = '#a8b0bd';
const GRID = '#e5e7eb';
const INK = '#334155';
const MUTED = '#64748b';
const SURFACE = '#ffffff';

interface Tip {
  x: number;
  y: number;
  lines: string[];
}

function Tooltip({ tip, width }: { tip: Tip | null; width: number }) {
  if (!tip) return null;
  const left = Math.min(Math.max(tip.x, 8), width - 8);
  return (
    <div
      className="pointer-events-none absolute z-10 rounded-md border border-slate-200 bg-white/95 px-2.5 py-1.5 text-[11px] text-slate-700 shadow-md"
      style={{ left, top: tip.y, transform: 'translate(-50%, calc(-100% - 8px))', whiteSpace: 'nowrap' }}
    >
      {tip.lines.map((l, i) => (
        <div key={i} className={i === 0 ? 'font-semibold text-slate-900' : ''}>
          {l}
        </div>
      ))}
    </div>
  );
}

function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** Horizontal bar with square baseline end and 4px rounded data end. */
function barPath(x0: number, x1: number, y: number, h: number, r = 4): string {
  const w = Math.max(x1 - x0, 0);
  const rr = Math.min(r, w, h / 2);
  return `M ${x0} ${y} H ${x0 + w - rr} Q ${x0 + w} ${y} ${x0 + w} ${y + rr} V ${y + h - rr} Q ${x0 + w} ${y + h} ${x0 + w - rr} ${y + h} H ${x0} Z`;
}

// ---------------------------------------------------------------------------
export interface BarItem {
  label: string;
  value: number;
  highlight?: boolean;
  tooltip?: string[];
}

interface HBarProps {
  items: BarItem[];
  domain: [number, number];
  log?: boolean;
  format: (v: number) => string;
  axisLabel: string;
  refLine?: { value: number; label: string };
  height?: number;
}

/** Horizontal bars, one per model; values past the domain are clipped and labelled. */
export function HBarChart({ items, domain, log = false, format, axisLabel, refLine }: HBarProps) {
  const [tip, setTip] = useState<Tip | null>(null);
  const W = 560;
  const row = 24;
  const m = { top: 8, right: 64, bottom: 34, left: 150 };
  const H = m.top + m.bottom + items.length * row;
  const iw = W - m.left - m.right;
  const [lo, hi] = domain;
  const sx = (v: number) => {
    const c = Math.min(Math.max(v, lo), hi);
    if (log) return m.left + ((Math.log10(c) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * iw;
    return m.left + ((c - lo) / (hi - lo)) * iw;
  };
  const ticks = log
    ? [0.01, 0.1, 1, 10, 100].filter((t) => t >= lo && t <= hi)
    : niceTicks(lo, hi, 5);
  const barH = Math.min(16, row - 8);

  return (
    <div className="relative w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={axisLabel}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={m.top} y2={H - m.bottom} stroke={GRID} strokeWidth={1} />
            <text x={sx(t)} y={H - m.bottom + 14} textAnchor="middle" fontSize={10} fill={MUTED}>
              {log ? (t < 1 ? t.toString() : t.toFixed(0)) : t.toFixed(t < 1 ? 2 : 1)}
            </text>
          </g>
        ))}
        <text x={m.left + iw / 2} y={H - 4} textAnchor="middle" fontSize={10.5} fill={MUTED}>
          {axisLabel}
        </text>
        {items.map((it, i) => {
          const y = m.top + i * row + (row - barH) / 2;
          const zeroOnLog = log && it.value <= 0;
          const x1 = zeroOnLog ? m.left : sx(it.value);
          const clipped = it.value > hi;
          return (
            <g
              key={it.label}
              onMouseMove={(e) => {
                const svg = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setTip({
                  x: e.clientX - svg.left,
                  y: e.clientY - svg.top,
                  lines: it.tooltip ?? [it.label, format(it.value)],
                });
              }}
              onMouseLeave={() => setTip(null)}
            >
              <rect x={0} y={m.top + i * row} width={W} height={row} fill="transparent" />
              <text
                x={m.left - 8}
                y={y + barH / 2 + 3.5}
                textAnchor="end"
                fontSize={11}
                fill={it.highlight ? '#0f172a' : INK}
                fontWeight={it.highlight ? 700 : 400}
              >
                {it.label}
              </text>
              {!zeroOnLog && (
                <path d={barPath(m.left, x1, y, barH)} fill={it.highlight ? HIGHLIGHT : NEUTRAL} />
              )}
              <text
                x={(zeroOnLog ? m.left : x1) + 5}
                y={y + barH / 2 + 3.5}
                fontSize={10.5}
                fill={INK}
                fontWeight={it.highlight ? 700 : 400}
              >
                {zeroOnLog ? '0' : format(it.value)}
                {clipped ? ' →' : ''}
              </text>
            </g>
          );
        })}
        {refLine && refLine.value >= lo && refLine.value <= hi && (
          <g>
            <line
              x1={sx(refLine.value)}
              x2={sx(refLine.value)}
              y1={m.top - 2}
              y2={H - m.bottom}
              stroke="#0f172a"
              strokeWidth={1.25}
              strokeDasharray="4 3"
            />
            <text x={sx(refLine.value) + 4} y={m.top + 6} fontSize={10} fill={INK}>
              {refLine.label}
            </text>
          </g>
        )}
        <line x1={m.left} x2={m.left} y1={m.top} y2={H - m.bottom} stroke="#cbd5e1" strokeWidth={1} />
      </svg>
      <Tooltip tip={tip} width={W} />
    </div>
  );
}

// ---------------------------------------------------------------------------
export interface LineSeries {
  name: string;
  color: string;
  values: number[];
}

/** Error vs depth: depth on the vertical axis (surface at the top), one line per model. */
export function DepthLineChart({
  depths,
  series,
  axisLabel,
  format,
}: {
  depths: number[];
  series: LineSeries[];
  axisLabel: string;
  format: (v: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 520;
  const H = 360;
  const m = { top: 14, right: 150, bottom: 34, left: 48 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const all = series.flatMap((s) => s.values);
  const xmax = Math.max(...all) * 1.08;
  const sx = (v: number) => m.left + (v / xmax) * iw;
  const sy = (d: number) => m.top + (d / 1000) * ih;
  const xt = niceTicks(0, xmax, 5);
  const yt = [0, 200, 400, 600, 800, 1000];
  const hd = hover !== null ? depths[hover] : null;

  return (
    <div className="relative w-full">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto"
        role="img"
        aria-label={axisLabel}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const y = ((e.clientY - r.top) / r.height) * H;
          const d = ((y - m.top) / ih) * 1000;
          let best = 0;
          depths.forEach((dd, i) => {
            if (Math.abs(dd - d) < Math.abs(depths[best] - d)) best = i;
          });
          setHover(best);
        }}
        onMouseLeave={() => setHover(null)}
      >
        {xt.map((t) => (
          <g key={`x${t}`}>
            <line x1={sx(t)} x2={sx(t)} y1={m.top} y2={H - m.bottom} stroke={GRID} strokeWidth={1} />
            <text x={sx(t)} y={H - m.bottom + 14} textAnchor="middle" fontSize={10} fill={MUTED}>
              {t.toFixed(t < 1 ? 2 : 1)}
            </text>
          </g>
        ))}
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.left} x2={W - m.right} y1={sy(t)} y2={sy(t)} stroke={GRID} strokeWidth={1} />
            <text x={m.left - 6} y={sy(t) + 3.5} textAnchor="end" fontSize={10} fill={MUTED}>
              {t}
            </text>
          </g>
        ))}
        <text x={m.left + iw / 2} y={H - 4} textAnchor="middle" fontSize={10.5} fill={MUTED}>
          {axisLabel}
        </text>
        <text x={12} y={m.top + ih / 2} textAnchor="middle" fontSize={10.5} fill={MUTED} transform={`rotate(-90 12 ${m.top + ih / 2})`}>
          Pressure (dbar)
        </text>
        {hd !== null && <line x1={m.left} x2={W - m.right} y1={sy(hd)} y2={sy(hd)} stroke="#94a3b8" strokeWidth={1} />}
        {series.map((s) => (
          <g key={s.name}>
            <path
              d={s.values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${sx(v)} ${sy(depths[i])}`).join(' ')}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {hover !== null && (
              <circle cx={sx(s.values[hover])} cy={sy(depths[hover])} r={4} fill={s.color} stroke={SURFACE} strokeWidth={2} />
            )}
          </g>
        ))}
        {/* direct labels at the 1000 dbar end, nudged apart */}
        {(() => {
          const ends = series
            .map((s) => ({ name: s.name, color: s.color, x: sx(s.values[s.values.length - 1]) }))
            .sort((a, b) => a.x - b.x);
          return ends.map((e, i) => (
            <g key={`lab-${e.name}`}>
              <line x1={W - m.right + 6} x2={W - m.right + 16} y1={H - m.bottom - 8 - i * 14} y2={H - m.bottom - 8 - i * 14} stroke={e.color} strokeWidth={2} />
              <text x={W - m.right + 20} y={H - m.bottom - 4.5 - i * 14} fontSize={10} fill={INK}>
                {e.name}
              </text>
            </g>
          ));
        })()}
      </svg>
      {hover !== null && (
        <div className="pointer-events-none absolute right-2 top-2 rounded-md border border-slate-200 bg-white/95 px-2.5 py-1.5 text-[11px] shadow-md">
          <div className="font-semibold text-slate-900">{depths[hover]} dbar</div>
          {series.map((s) => (
            <div key={s.name} className="flex items-center gap-1.5 text-slate-700">
              <span className="inline-block h-0.5 w-3" style={{ background: s.color }} />
              {s.name}: {format(s.values[hover])}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
export interface ScatterPoint {
  label: string;
  x: number;
  y: number;
  color?: string;
  highlight?: boolean;
  showLabel?: boolean;
  labelSide?: 'left' | 'right';
  labelDy?: number;
  tooltip?: string[];
}

/** Scatter with optional log x axis (used for inversion rate, which spans 0.02-12%). */
export function ScatterChart({
  points,
  xLog = false,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  annotations = [],
}: {
  annotations?: { x: number; y: number; text: string }[];
  points: ScatterPoint[];
  xLog?: boolean;
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel: string;
  yLabel: string;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const W = 540;
  const H = 330;
  const m = { top: 14, right: 24, bottom: 38, left: 52 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const sx = (v: number) => {
    const c = Math.min(Math.max(v, xDomain[0]), xDomain[1]);
    return xLog
      ? m.left + ((Math.log10(c) - Math.log10(xDomain[0])) / (Math.log10(xDomain[1]) - Math.log10(xDomain[0]))) * iw
      : m.left + ((c - xDomain[0]) / (xDomain[1] - xDomain[0])) * iw;
  };
  const sy = (v: number) => m.top + (1 - (v - yDomain[0]) / (yDomain[1] - yDomain[0])) * ih;
  const xt = xLog ? [0.01, 0.1, 1, 10, 100].filter((t) => t >= xDomain[0] && t <= xDomain[1]) : niceTicks(xDomain[0], xDomain[1]);
  const yt = niceTicks(yDomain[0], yDomain[1], 5);

  return (
    <div className="relative w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={`${yLabel} vs ${xLabel}`}>
        {xt.map((t) => (
          <g key={`x${t}`}>
            <line x1={sx(t)} x2={sx(t)} y1={m.top} y2={H - m.bottom} stroke={GRID} strokeWidth={1} />
            <text x={sx(t)} y={H - m.bottom + 14} textAnchor="middle" fontSize={10} fill={MUTED}>
              {t < 1 ? t : t.toFixed(0)}
            </text>
          </g>
        ))}
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.left} x2={W - m.right} y1={sy(t)} y2={sy(t)} stroke={GRID} strokeWidth={1} />
            <text x={m.left - 6} y={sy(t) + 3.5} textAnchor="end" fontSize={10} fill={MUTED}>
              {t.toFixed(t < 1 ? 2 : t < 10 ? 1 : 0)}
            </text>
          </g>
        ))}
        <text x={m.left + iw / 2} y={H - 4} textAnchor="middle" fontSize={10.5} fill={MUTED}>
          {xLabel}
        </text>
        <text x={12} y={m.top + ih / 2} textAnchor="middle" fontSize={10.5} fill={MUTED} transform={`rotate(-90 12 ${m.top + ih / 2})`}>
          {yLabel}
        </text>
        {annotations.map((a) => (
          <text key={a.text} x={sx(a.x)} y={sy(a.y)} textAnchor="middle" fontSize={10.5} fill={MUTED} fontStyle="italic">
            {a.text}
          </text>
        ))}
        {points.map((p) => {
          const cx = sx(p.x);
          const cy = sy(p.y);
          const fill = p.color ?? (p.highlight ? HIGHLIGHT : NEUTRAL);
          return (
            <g
              key={p.label}
              onMouseMove={(e) => {
                const svg = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setTip({ x: e.clientX - svg.left, y: e.clientY - svg.top, lines: p.tooltip ?? [p.label] });
              }}
              onMouseLeave={() => setTip(null)}
            >
              <circle cx={cx} cy={cy} r={12} fill="transparent" />
              <circle cx={cx} cy={cy} r={p.highlight ? 6 : 4.5} fill={fill} stroke={SURFACE} strokeWidth={2} />
              {p.showLabel && (
                <text
                  x={p.labelSide === 'left' ? cx - 9 : cx + 9}
                  y={cy + 3.5 + (p.labelDy ?? 0)}
                  textAnchor={p.labelSide === 'left' ? 'end' : 'start'}
                  fontSize={10.5}
                  fill={p.highlight ? '#0f172a' : INK}
                  fontWeight={p.highlight ? 700 : 400}
                >
                  {p.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} width={W} />
    </div>
  );
}

// ---------------------------------------------------------------------------
/** Reliability diagram: nominal interval level vs observed coverage, raw and calibrated. */
export function ReliabilityChart({ nominal, raw, cal }: { nominal: number[]; raw: number[]; cal: number[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 360;
  const H = 300;
  const m = { top: 12, right: 14, bottom: 38, left: 46 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const sx = (v: number) => m.left + (v / 100) * iw;
  const sy = (v: number) => m.top + (1 - v / 100) * ih;
  const ticks = [0, 20, 40, 60, 80, 100];
  const line = (vals: number[]) => vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${sx(nominal[i])} ${sy(v)}`).join(' ');
  const series = [
    { name: 'Raw MC dropout', color: SERIES[1], vals: raw },
    { name: 'Calibrated', color: SERIES[0], vals: cal },
  ];
  return (
    <div className="relative w-full">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto"
        role="img"
        aria-label="Reliability diagram"
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = (((e.clientX - r.left) / r.width) * W - m.left) / iw * 100;
          let best = 0;
          nominal.forEach((n, i) => {
            if (Math.abs(n - x) < Math.abs(nominal[best] - x)) best = i;
          });
          setHover(best);
        }}
        onMouseLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={m.top} y2={H - m.bottom} stroke={GRID} strokeWidth={1} />
            <line x1={m.left} x2={W - m.right} y1={sy(t)} y2={sy(t)} stroke={GRID} strokeWidth={1} />
            <text x={sx(t)} y={H - m.bottom + 14} textAnchor="middle" fontSize={10} fill={MUTED}>
              {t}
            </text>
            <text x={m.left - 6} y={sy(t) + 3.5} textAnchor="end" fontSize={10} fill={MUTED}>
              {t}
            </text>
          </g>
        ))}
        <line x1={sx(0)} y1={sy(0)} x2={sx(100)} y2={sy(100)} stroke="#94a3b8" strokeWidth={1.25} strokeDasharray="4 3" />
        <text x={sx(62)} y={sy(70)} fontSize={10} fill={MUTED} transform={`rotate(-38 ${sx(62)} ${sy(70)})`}>
          perfect calibration
        </text>
        {series.map((s) => (
          <g key={s.name}>
            <path d={line(s.vals)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" />
            {s.vals.map((v, i) => (
              <circle key={i} cx={sx(nominal[i])} cy={sy(v)} r={hover === i ? 5 : 3.5} fill={s.color} stroke={SURFACE} strokeWidth={2} />
            ))}
          </g>
        ))}
        <text x={m.left + iw / 2} y={H - 4} textAnchor="middle" fontSize={10.5} fill={MUTED}>
          Nominal interval (%)
        </text>
        <text x={12} y={m.top + ih / 2} textAnchor="middle" fontSize={10.5} fill={MUTED} transform={`rotate(-90 12 ${m.top + ih / 2})`}>
          Observations inside (%)
        </text>
      </svg>
      <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-slate-600">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
      {hover !== null && (
        <div className="pointer-events-none absolute left-14 top-2 rounded-md border border-slate-200 bg-white/95 px-2.5 py-1.5 text-[11px] shadow-md">
          <div className="font-semibold text-slate-900">{nominal[hover]}% interval</div>
          <div className="text-slate-700">Raw: {raw[hover].toFixed(1)}% inside</div>
          <div className="text-slate-700">Calibrated: {cal[hover].toFixed(1)}% inside</div>
        </div>
      )}
    </div>
  );
}
