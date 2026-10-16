import { ForecastResult, ArgoProfile, ChatMessage, BenchmarkResults, ModelInfo } from '../types.ts';

const API_BASE = '/api';

export interface FloatSummary {
  wmo_id: string;
  name: string;
  baseLat: number;
  baseLon: number;
  cycles: number[];
  defaultCycle: number;
  firstDate?: string;
  lastDate?: string;
}

export interface ChatResponse {
  text: string;
  citations: string[];
  verified: boolean;
  forecast_context?: ForecastResult;
}

export const apiService = {
  async getHealth(): Promise<{ status: string; service: string }> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  },

  async getFloats(): Promise<FloatSummary[]> {
    const res = await fetch(`${API_BASE}/floats`);
    if (!res.ok) throw new Error(`Failed to load floats: ${res.statusText}`);
    return res.json();
  },

  async getProfiles(wmoId: string): Promise<ArgoProfile[]> {
    const res = await fetch(`${API_BASE}/profiles/${wmoId}`);
    if (!res.ok) throw new Error(`Failed to load profiles for ${wmoId}: ${res.statusText}`);
    return res.json();
  },

  async runForecast(wmoId: string, cycle: number): Promise<ForecastResult> {
    const res = await fetch(`${API_BASE}/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wmoId, cycle })
    });
    if (!res.ok) {
      const detail = await parseErrorDetail(res, res.statusText);
      throw new Error(`Forecast computation failed: ${detail}`);
    }
    return res.json();
  },

  async getBenchmark(): Promise<BenchmarkResults> {
    const res = await fetch(`${API_BASE}/benchmark`);
    if (!res.ok) {
      const detail = await parseErrorDetail(res, res.statusText);
      throw new Error(`Failed to load benchmark results: ${detail}`);
    }
    return res.json();
  },

  async getModelInfo(): Promise<ModelInfo> {
    const res = await fetch(`${API_BASE}/model`);
    if (!res.ok) throw new Error(`Failed to load model info: ${res.statusText}`);
    return res.json();
  },

  async sendChatMessage(query: string, wmoId: string, cycle: number): Promise<ChatResponse> {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, wmoId, cycle })
    });
    if (!res.ok) {
      const detail = await parseErrorDetail(res, res.statusText);
      throw new Error(`Chat generation failed: ${detail}`);
    }
    return res.json();
  }
};

async function parseErrorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json() as unknown;
    if (typeof body === 'object' && body !== null) {
      const record = body as Record<string, unknown>;
      if (typeof record.detail === 'string' && record.detail) return record.detail;
      const err = record.error as Record<string, unknown> | undefined;
      if (err && typeof err.message === 'string' && err.message) return err.message;
    }
  } catch {
    // Response body is not JSON - fall back to HTTP status text.
  }
  return fallback;
}

export type { ChatMessage };
