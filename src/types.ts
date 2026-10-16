export interface ArgoMeasurement {
  depth_dbar: number; // Pressure / depth in dbar
  temperature: number; // in °C
  salinity: number; // in PSU
  qc_temperature: number; // 1 = good, 2 = probably good
  qc_salinity: number;
}

export interface ArgoProfile {
  profile_id: string;
  wmo_id: string;
  float_name: string;
  platform_type: string;
  cycle_number: number;
  date: string;
  latitude: number;
  longitude: number;
  sea_region: string;
  data_mode: 'D' | 'R'; // Delayed-mode vs Real-time
  qc_status: 'QC_PASS_FLAG_1' | 'QC_PASS_FLAG_2' | 'QC_SUSPECT';
  raw_netcdf_source: string;
  gdac_archive_path: string;
  measurements: ArgoMeasurement[];
}

export interface UncertaintyBound {
  depth_dbar: number;
  mean_temp: number;
  std_temp: number;
  ci90_temp_lower: number;
  ci90_temp_upper: number;
  ci95_temp_lower: number;
  ci95_temp_upper: number;

  mean_sal: number;
  std_sal: number;
  ci90_sal_lower: number;
  ci90_sal_upper: number;
  ci95_sal_lower: number;
  ci95_sal_upper: number;
}

export interface EvidenceCitation {
  citation_id: string;
  wmo_id: string;
  cycle_number: number;
  date: string;
  latitude: number;
  longitude: number;
  distance_km: number;
  similarity_score: number; // 0 to 1
  cosine_profile_sim: number;
  spatial_temporal_weight: number;
  qc_flag: number;
  raw_netcdf_file: string;
  gdac_archive_path?: string;
  provenance_chain: {
    origin: string;
    archive_gdac: string;
    qc_step: string;
    standardized_grid: string;
  };
  key_feature_relevance: string;
  measurements: ArgoMeasurement[];
}

export interface PhysicalDiagnostics {
  is_gravitationally_stable: boolean;
  min_density_gradient: number; // d(sigma)/dz >= 0
  mixed_layer_depth_m: number;
  max_thermocline_gradient: number; // °C / dbar
  thermocline_depth_m: number;
  halocline_depth_m: number;
  surface_potential_density: number; // kg/m^3
  deep_potential_density: number; // kg/m^3
  stability_violation_count: number;
  density_profile: {
    depth_dbar: number;
    sigma_theta: number; // kg/m^3
    buoyancy_frequency_n2: number; // s^-2
  }[];
}

export interface XAIAttribution {
  temporal_attribution: {
    cycle_offset: number; // -3, -2, -1
    cycle_label: string;
    importance_score: number; // 0 to 1
    interpretation: string;
  }[];
  depth_attribution_matrix: {
    input_depth: number;
    output_depth: number;
    saliency_weight: number;
  }[];
  key_depth_influences: {
    target_zone: string;
    dominant_input_depth: string;
    attribution_percentage: number;
    scientific_driver: string;
  }[];
}

export interface ForecastResult {
  forecast_id: string;
  target_float_id: string;
  target_cycle: number;
  predicted_cycle: number;
  forecast_date: string;
  latitude: number;
  longitude: number;
  input_sequence_cycles: number[];
  profiles: {
    depth_dbar: number;
    temperature_forecast: number;
    salinity_forecast: number;
    persistence_temperature: number;
    persistence_salinity: number;
    ridge_temperature: number;
    ridge_salinity: number;
    observed_temperature: number | null;
    observed_salinity: number | null;
  }[];
  has_observation?: boolean;
  model_version?: string;
  input_window?: {
    dates: string[];
    target_date: string | null;
    gaps_days: number[];
    within_training_gap_range: boolean;
  };
  uncertainty_bounds: UncertaintyBound[];
  physical_diagnostics: PhysicalDiagnostics;
  xai_attribution: XAIAttribution;
  evidence_citations: EvidenceCitation[];
  metrics_comparison: {
    model: string;
    profile_rmse: number;
    profile_mae: number;
    thermocline_rmse: number;
    deep_rmse: number;
    salinity_rmse?: number;
    physical_violation_rate: number;
  }[];
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  content: string;
  forecast_context?: ForecastResult;
  grounding_verified?: boolean;
  cited_floats?: string[];
  suggested_queries?: string[];
}

// ---------------------------------------------------------------------------
// Benchmark study (GET /api/benchmark)
// ---------------------------------------------------------------------------
export interface BenchmarkModel {
  name: string;
  short: string;
  family: string;
  description: string;
  color: string;
  proposed: boolean;
}

export interface BenchmarkRow {
  temp_rmse: number;
  temp_rmse_ci: [number, number];
  sal_rmse: number;
  temp_mae: number;
  sal_mae: number;
  therm_rmse: number;
  deep_rmse: number;
  temp_r2: number;
  temp_bias: number;
  acc_T: number;
  p95_abs_error_T: number;
  skill_vs_persistence: number;
  skill_vs_climatology: number;
  inv_any_pct: number;
  inv_001_pct: number;
  inv_003_pct: number;
  mld_rmse: number;
  mld_bias: number;
  d20_rmse: number;
  ohc_rmse: number;
  crps_point_T: number;
  direction_acc_T: number | null;
  direction_f1_T: number | null;
  n_params: number;
  fit_seconds: number | null;
  seed_temp_rmse_mean: number | null;
  seed_temp_rmse_std: number | null;
  seed_inv_003_mean: number | null;
  seed_inv_003_std: number | null;
  seed_inv_any_mean: number | null;
  seed_inv_any_std: number | null;
  seed_mld_rmse_mean: number | null;
  diff_vs_floatchat_ci: [number, number] | null;
}

export interface BenchmarkUQ {
  coverage95_raw: number;
  coverage95_cal: number;
  coverage90_raw: number;
  coverage90_cal: number;
  scale_k: number;
  width95_cal: number;
  crps_raw: number;
  crps_cal: number;
  spread_skill_ratio: number;
  reliability_raw: number[];
  reliability_cal: number[];
}

export interface BenchmarkSplit {
  label: string;
  description: string;
  n_windows: { train: number; val: number; test: number };
  n_floats: { train: number; val: number; test: number };
  rows: Record<string, BenchmarkRow>;
  per_depth_temp_rmse: Record<string, number[]>;
  per_depth_sal_rmse: Record<string, number[]>;
  season_temp_rmse: Record<string, Record<string, number>>;
  season_counts: Record<string, number> | null;
  uq: Record<string, BenchmarkUQ>;
  reliability_nominal: number[];
  observed: { inv_any_pct: number; inv_001_pct: number; inv_003_pct: number; mld_mean_m: number };
  seeds: number[] | null;
}

export interface MldVariantResult {
  mld_rmse: number;
  mld_rmse_std: number | null;
  mld_bias: number;
  temp_rmse: number;
  inv_003_pct: number;
  pct_mixed_5_20: number;
}

export interface MldVariant {
  id: string;
  label: string;
  lam_stab?: number;
  tol?: number;
  lam_surf?: number;
  chronological?: MldVariantResult;
  unseen_floats?: MldVariantResult;
}

export interface XaiMethodResult {
  faithfulness: number;
  repeatability: number;
  robustness: number;
  completeness_error: number;
  runtime_ms: number;
  share_latest_cycle: number;
  temporal_weights: number[];
  deletion_auc: number;
}

export interface BenchmarkResults {
  generated: string;
  proposed_model: string;
  depths_dbar: number[];
  dataset: { n_profiles: number; n_floats: number; date_range: string[]; region: string; levels: string; task: string };
  models: BenchmarkModel[];
  splits: Record<'chronological' | 'unseen_floats', BenchmarkSplit>;
  mld_fix: {
    variants: MldVariant[];
    observed_pct_mixed_5_20: Record<string, number>;
    selected_on_validation: Record<string, string>;
    seeds: number[] | null;
  } | null;
  xai: {
    target: string;
    n_windows: number;
    split: string;
    methods: Record<string, XaiMethodResult>;
    depth_matrix_similarity: Record<string, number>;
  } | null;
  summary: Record<string, number>;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  architecture: string;
  split: string;
  test_target_dates: string[];
  test_rmse_temp: number;
  test_rmse_sal: number;
  test_rmse_temp_persistence: number;
  test_rmse_temp_ridge: number;
  [key: string]: unknown;
}
