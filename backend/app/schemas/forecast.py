"""Pydantic schemas for forecast-related API models."""

from datetime import date
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    service: str = "FloatChat-XRAG-Forecasting-Engine"
    region: str = "Arabian Sea / Northern Indian Ocean"
    version: str = "1.2.0-IEEE-Access-Spec"


class FloatSummary(BaseModel):
    """Summary of an Argo float."""
    wmo_id: str
    name: str
    baseLat: float
    baseLon: float
    cycles: List[int]
    defaultCycle: int
    firstDate: Optional[str] = None
    lastDate: Optional[str] = None


class ArgoMeasurement(BaseModel):
    """Single depth measurement from an Argo profile."""
    depth_dbar: int
    temperature: float
    salinity: float
    qc_temperature: int
    qc_salinity: int


class ArgoProfile(BaseModel):
    """Historical Argo profile with measurements."""
    profile_id: str
    wmo_id: str
    float_name: str
    platform_type: str
    cycle_number: int
    date: date
    latitude: float
    longitude: float
    sea_region: str
    data_mode: Literal["D", "R"]
    qc_status: str
    raw_netcdf_source: str
    gdac_archive_path: str
    measurements: List[ArgoMeasurement]


class ForecastRequest(BaseModel):
    """Request to generate a forecast."""
    wmoId: str = Field(..., description="WMO ID of the Argo float")
    cycle: int = Field(..., description="Cycle number to forecast from")


class ProfileData(BaseModel):
    """Single depth level in a forecast profile."""
    depth_dbar: int
    temperature_forecast: float
    salinity_forecast: float
    persistence_temperature: float
    persistence_salinity: float
    ridge_temperature: float
    ridge_salinity: float
    observed_temperature: Optional[float] = None
    observed_salinity: Optional[float] = None


class UncertaintyBound(BaseModel):
    """Uncertainty bounds at a single depth level."""
    depth_dbar: int
    mean_temp: float
    std_temp: float
    ci90_temp_lower: float
    ci90_temp_upper: float
    ci95_temp_lower: float
    ci95_temp_upper: float
    mean_sal: float
    std_sal: float
    ci90_sal_lower: float
    ci90_sal_upper: float
    ci95_sal_lower: float
    ci95_sal_upper: float


class PhysicalDiagnostics(BaseModel):
    """Physical diagnostics for a forecast."""
    is_gravitationally_stable: bool
    min_density_gradient: float
    mixed_layer_depth_m: float
    max_thermocline_gradient: float
    thermocline_depth_m: float
    halocline_depth_m: float
    surface_potential_density: float
    deep_potential_density: float
    stability_violation_count: int
    density_profile: List[dict]


class TemporalAttribution(BaseModel):
    """Temporal attribution for XAI."""
    cycle_offset: int
    cycle_label: str
    importance_score: float
    interpretation: str


class DepthAttribution(BaseModel):
    """Cross-depth saliency entry."""
    input_depth: int
    output_depth: int
    saliency_weight: float


class KeyDepthInfluence(BaseModel):
    """Key depth influence interpretation."""
    target_zone: str
    dominant_input_depth: str
    attribution_percentage: float
    scientific_driver: str


class XAIAttribution(BaseModel):
    """XAI attribution results."""
    temporal_attribution: List[TemporalAttribution]
    depth_attribution_matrix: List[dict]
    key_depth_influences: List[dict]


class EvidenceCitation(BaseModel):
    """Evidence citation from NetCDF archives."""
    citation_id: str
    wmo_id: str
    cycle_number: int
    date: str
    latitude: float
    longitude: float
    distance_km: float
    similarity_score: float
    cosine_profile_sim: float
    spatial_temporal_weight: float
    qc_flag: int
    raw_netcdf_file: str
    provenance_chain: dict
    key_feature_relevance: str
    measurements: List[ArgoMeasurement]


class MetricsComparison(BaseModel):
    """Model comparison metrics."""
    model: str
    profile_rmse: float
    profile_mae: float
    thermocline_rmse: float
    deep_rmse: float
    salinity_rmse: Optional[float] = None
    physical_violation_rate: float


class ForecastResult(BaseModel):
    """Complete forecast result with all components."""
    forecast_id: str
    target_float_id: str
    target_cycle: int
    predicted_cycle: int
    forecast_date: str
    latitude: float
    longitude: float
    input_sequence_cycles: List[int]
    profiles: List[ProfileData]
    uncertainty_bounds: List[UncertaintyBound]
    physical_diagnostics: PhysicalDiagnostics
    xai_attribution: XAIAttribution
    evidence_citations: List[EvidenceCitation]
    metrics_comparison: List[MetricsComparison]


class ForecastRequest(BaseModel):
    """Request to generate a forecast."""
    wmoId: str = Field(..., description="WMO ID of the Argo float")
    cycle: int = Field(..., description="Cycle number to forecast from")