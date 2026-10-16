"""Forecast endpoint: model inference, uncertainty, TEOS-10 checks, XAI and evidence links."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.evidence import get_evidence_links
from app.core.physics import STANDARD_DEPTHS, validate_profile_physics
from app.core.uq import run_mc_dropout_inference, uncertainty_bounds_to_dict
from app.db.session import SessionLocal
from app.ml.model import PhysicsInformedBiLSTM
from app.ml.xai import compute_xai_attribution
from app.models.schema import ArgoFloat, ArgoProfile, ForecastLog, ProfileLevel

router = APIRouter()

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
N_LEV = 16
# The model was trained on windows whose consecutive profiles are 3-20 days apart.
TRAINING_GAP_DAYS = (3.0, 20.0)

# Loaded once and reused by every request.
model: Optional[PhysicsInformedBiLSTM] = None
preprocessor: Optional[dict] = None
ridge: Optional[dict] = None
metadata: Dict = {}


def load_model_and_preprocessor() -> bool:
    """Load the serving model, its scalers, the Ridge baseline and metadata (once)."""
    global model, preprocessor, ridge, metadata
    if model is not None and preprocessor is not None:
        return True
    model_path = ARTIFACTS_DIR / "model_weights.pt"
    prep_path = ARTIFACTS_DIR / "preprocessor.joblib"
    if not model_path.exists() or not prep_path.exists():
        print(f"Model artifacts missing in {ARTIFACTS_DIR}")
        return False
    try:
        preprocessor = joblib.load(prep_path)
        net = PhysicsInformedBiLSTM()
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval()
        model = net
        ridge_path = ARTIFACTS_DIR / "ridge_baseline.joblib"
        ridge = joblib.load(ridge_path) if ridge_path.exists() else None
        meta_path = ARTIFACTS_DIR / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return True
    except Exception as exc:  # pragma: no cover - logged for operators
        print(f"Error loading model: {exc}")
        model = preprocessor = None
        return False


def model_version() -> str:
    return metadata.get("model_version", "unknown")


@router.get("/model", response_model=dict)
async def get_model_info():
    """Provenance and held-out scores of the model that serves forecasts."""
    if not load_model_and_preprocessor():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return metadata


def _levels_by_profile(db, profile_ids: List[str]) -> Dict[str, List[ProfileLevel]]:
    rows = (
        db.query(ProfileLevel)
        .filter(ProfileLevel.profile_id.in_(profile_ids))
        .order_by(ProfileLevel.profile_id, ProfileLevel.depth_dbar.asc())
        .all()
    )
    out: Dict[str, List[ProfileLevel]] = {pid: [] for pid in profile_ids}
    for lev in rows:
        out[lev.profile_id].append(lev)
    return out


def _rmse(pred, obs, idx=None) -> float:
    pred, obs = np.asarray(pred), np.asarray(obs)
    if idx is not None:
        pred, obs = pred[idx], obs[idx]
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def _mae(pred, obs) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(obs))))


@router.post("/forecast", response_model=dict)
async def run_forecast(request: dict, background_tasks: BackgroundTasks):
    """
    Forecast the next profile of a float.

    Request body: { "wmoId": "2902127", "cycle": 120 }
    The selected cycle is the base cycle (t-1): it and the two profiles before it are the
    model input, and the forecast is for the float's next profile. When that next
    profile is already in the database it is used to score this one forecast;
    for a float's latest cycle the forecast is a true look ahead with no observation yet.
    """
    wmo_id = request.get("wmoId")
    cycle = request.get("cycle")
    if not wmo_id or cycle is None:
        raise HTTPException(status_code=400, detail="wmoId and cycle are required")
    wmo_id, cycle = str(wmo_id), int(cycle)
    if not load_model_and_preprocessor():
        raise HTTPException(status_code=503, detail="Model not loaded")

    db = SessionLocal()
    try:
        float_obj = db.query(ArgoFloat).filter(ArgoFloat.wmo_id == wmo_id).first()
        if not float_obj:
            raise HTTPException(status_code=404, detail=f"Float {wmo_id} not found")
        profiles = (
            db.query(ArgoProfile)
            .filter(ArgoProfile.wmo_id == wmo_id)
            .order_by(ArgoProfile.cycle_number.asc())
            .all()
        )
        cycles = [p.cycle_number for p in profiles]
        if cycle not in cycles:
            raise HTTPException(status_code=404, detail=f"Cycle {cycle} not found for float {wmo_id}")
        idx = cycles.index(cycle)
        if idx < 2:
            earliest = cycles[2] if len(cycles) > 2 else None
            hint = (f" Select cycle {earliest} or later for this float." if earliest is not None
                    else " This float has fewer than 3 recorded cycles.")
            raise HTTPException(
                status_code=400,
                detail=(f"Not enough history for a forecast: cycle {cycle} is position {idx + 1} of {len(cycles)} "
                        f"for float {wmo_id}, and the model needs the base cycle plus the two before it "
                        f"(t-3, t-2, t-1)." + hint),
            )

        inputs = profiles[idx - 2:idx + 1]  # t-3, t-2, t-1 (= selected base cycle)
        target = profiles[idx + 1] if idx + 1 < len(profiles) else None
        window = inputs + ([target] if target is not None else [])
        levels = _levels_by_profile(db, [p.profile_id for p in window])
        if any(len(levels[p.profile_id]) != N_LEV for p in window):
            raise HTTPException(status_code=422, detail="A profile in the input window is not on the 16-level grid")
        temps = np.array([[lv.temperature for lv in levels[p.profile_id]] for p in window])
        sals = np.array([[lv.salinity for lv in levels[p.profile_id]] for p in window])
        input_cycles = [p.cycle_number for p in inputs]
        has_obs = target is not None
        obs_temps, obs_sals = (temps[3], sals[3]) if has_obs else (None, None)
        pers_temps, pers_sals = temps[2], sals[2]
        gaps = [float((window[k + 1].date - window[k].date).days) for k in range(len(window) - 1)]
        within_training = all(TRAINING_GAP_DAYS[0] <= g <= TRAINING_GAP_DAYS[1] for g in gaps)
        predicted_cycle = target.cycle_number if has_obs else cycle + 1
        base = inputs[-1]

        # ---- model input (1, 3, 32), standardised per depth with the training scalers
        ts, ss = preprocessor["temp_scaler"], preprocessor["sal_scaler"]
        x = np.concatenate([ts.transform(temps[:3]), ss.transform(sals[:3])], axis=1).astype(np.float32)
        input_tensor = torch.from_numpy(x[None])
        with torch.no_grad():
            t_scaled, s_scaled = model(input_tensor)
        temp_pred = ts.inverse_transform(t_scaled.numpy())[0]
        sal_pred = ss.inverse_transform(s_scaled.numpy())[0]

        # ---- Ridge baseline (same input, trained on the same split)
        if ridge is not None:
            r = ridge["model"].predict(x.reshape(1, -1))
            ridge_temps = ts.inverse_transform(r[:, :N_LEV])[0]
            ridge_sals = ss.inverse_transform(r[:, N_LEV:])[0]
        else:
            ridge_temps, ridge_sals = pers_temps, pers_sals

        # ---- calibrated MC-dropout uncertainty
        bounds, _ = run_mc_dropout_inference(model, input_tensor, ts, ss)
        uncertainty_bounds = uncertainty_bounds_to_dict(bounds)

        profiles_result = [{
            "depth_dbar": int(d),
            "temperature_forecast": float(temp_pred[i]),
            "salinity_forecast": float(sal_pred[i]),
            "persistence_temperature": float(pers_temps[i]),
            "persistence_salinity": float(pers_sals[i]),
            "ridge_temperature": float(ridge_temps[i]),
            "ridge_salinity": float(ridge_sals[i]),
            "observed_temperature": float(obs_temps[i]) if has_obs else None,
            "observed_salinity": float(obs_sals[i]) if has_obs else None,
        } for i, d in enumerate(STANDARD_DEPTHS)]

        # ---- evidence links: QC-pass historical profiles most similar to the forecast,
        # searched around the base-cycle position
        lat, lon = float(base.latitude), float(base.longitude)
        evidence = get_evidence_links(wmo_id, cycle, db, query_vector=np.concatenate([temp_pred, sal_pred]),
                                      query_lat=lat, query_lon=lon)
        cited = _levels_by_profile(db, [e["profile_id"] for e in evidence]) if evidence else {}
        evidence_citations = []
        for e in evidence:
            qc_flag = 1 if e["qc_status"] == "QC_PASS_FLAG_1" else 2
            evidence_citations.append({
                "citation_id": f"EVID_CITE_{e['wmo_id']}_C{e['cycle']}",
                "wmo_id": e["wmo_id"],
                "cycle_number": e["cycle"],
                "date": e["date"],
                "latitude": e.get("latitude", 0.0),
                "longitude": e.get("longitude", 0.0),
                "distance_km": e["distance_km"],
                "similarity_score": e["cosine_similarity"],
                "cosine_profile_sim": e["cosine_similarity"],
                "spatial_temporal_weight": e["composite_score"],
                "qc_flag": qc_flag,
                "raw_netcdf_file": e["raw_netcdf_source"],
                "gdac_archive_path": e["gdac_archive_path"],
                "provenance_chain": {
                    "origin": f"Argo GDAC profile {e['raw_netcdf_source']}",
                    "archive_gdac": e["gdac_archive_path"],
                    "qc_step": (f"QC flag {qc_flag} (delayed mode)" if qc_flag == 1
                                else f"QC flag {qc_flag} (adjusted real-time)"),
                    "standardized_grid": "16-level canonical grid [5..1000 dbar]",
                },
                "key_feature_relevance": f"Analogous profile at {e['distance_km']:.1f} km",
                "measurements": [{
                    "depth_dbar": lv.depth_dbar, "temperature": lv.temperature, "salinity": lv.salinity,
                    "qc_temperature": qc_flag, "qc_salinity": qc_flag,
                } for lv in cited.get(e["profile_id"], [])],
            })

        # ---- TEOS-10 diagnostics of the forecast
        phys = validate_profile_physics(temp_pred, sal_pred, STANDARD_DEPTHS, lat, lon)
        sal_grad = np.abs(np.diff(sal_pred) / np.diff(STANDARD_DEPTHS))
        density_profile = []
        for d in (5, 50, 100, 200, 400, 700, 1000):
            k = int(np.where(STANDARD_DEPTHS == d)[0][0])
            density_profile.append({
                "depth_dbar": int(d),
                "sigma_theta": round(float(phys["potential_density"][k]), 3),
                "buoyancy_frequency_n2": float(phys["brunt_vaisala_N2"][k - 1]) if k > 0 else 0.0,
            })
        mld = phys["mld_dbar"]
        physical_diagnostics = {
            "is_gravitationally_stable": bool(phys["is_gravitationally_stable"]),
            "min_density_gradient": float(np.min(phys["stability"])),
            "mixed_layer_depth_m": float(mld) if not np.isnan(mld) else float(STANDARD_DEPTHS[-1]),
            "max_thermocline_gradient": float(phys["max_thermocline_gradient"]),
            "thermocline_depth_m": int(STANDARD_DEPTHS[phys["thermocline_depth_idx"]]),
            "halocline_depth_m": int(STANDARD_DEPTHS[int(np.argmax(sal_grad)) + 1]),
            "surface_potential_density": round(float(phys["potential_density"][0]), 2),
            "deep_potential_density": round(float(phys["potential_density"][-1]), 2),
            "stability_violation_count": int(phys["stability_violations"]),
            "density_profile": density_profile,
        }

        # ---- Integrated Gradients attribution (temperature at 100 dbar)
        xai_result = compute_xai_attribution(
            model, input_tensor, target_variable="temp", target_depth_idx=4,
            input_cycles=input_cycles, temp_profile=temp_pred, sal_profile=sal_pred,
            mld_dbar=physical_diagnostics["mixed_layer_depth_m"],
            thermocline_depth_dbar=physical_diagnostics["thermocline_depth_m"],
        )
        xai_dict = {k: xai_result[k] for k in
                    ("temporal_attribution", "depth_attribution_matrix", "key_depth_influences", "convergence_delta")}

        # ---- score this single forecast against the observed target profile
        therm = np.where((STANDARD_DEPTHS >= 50) & (STANDARD_DEPTHS <= 200))[0]
        deep = np.where((STANDARD_DEPTHS >= 500) & (STANDARD_DEPTHS <= 1000))[0]

        def row(name, t, s):
            p = validate_profile_physics(np.asarray(t), np.asarray(s), STANDARD_DEPTHS, lat, lon)
            return {
                "model": name,
                "profile_rmse": round(_rmse(t, obs_temps), 3),
                "profile_mae": round(_mae(t, obs_temps), 3),
                "thermocline_rmse": round(_rmse(t, obs_temps, therm), 3),
                "deep_rmse": round(_rmse(t, obs_temps, deep), 3),
                "salinity_rmse": round(_rmse(s, obs_sals), 4),
                "physical_violation_rate": round(p["stability_violations"] / (N_LEV - 1) * 100, 1),
            }

        metrics_comparison = [
            row("Persistence (t-1)", pers_temps, pers_sals),
            row("Ridge regression", ridge_temps, ridge_sals),
            row("FloatChat PI-BiLSTM", temp_pred, sal_pred),
        ] if has_obs else []

        response = {
            "forecast_id": f"FC_{wmo_id}_C{cycle}_{int(datetime.now().timestamp() * 1000)}",
            "target_float_id": wmo_id,
            "target_cycle": cycle,
            "predicted_cycle": predicted_cycle,
            "forecast_date": str(target.date) if has_obs else str(base.date),
            "has_observation": has_obs,
            "latitude": lat,
            "longitude": lon,
            "input_sequence_cycles": input_cycles,
            "input_window": {
                "dates": [str(p.date) for p in inputs],
                "target_date": str(target.date) if has_obs else None,
                "gaps_days": gaps,
                "within_training_gap_range": within_training,
            },
            "model_version": model_version(),
            "profiles": profiles_result,
            "uncertainty_bounds": uncertainty_bounds,
            "physical_diagnostics": physical_diagnostics,
            "xai_attribution": xai_dict,
            "evidence_citations": evidence_citations,
            "metrics_comparison": metrics_comparison,
        }
        background_tasks.add_task(persist_forecast_log, response)
        return response
    finally:
        db.close()


def persist_forecast_log(result: dict):
    """Background task: store the forecast in forecast_logs."""
    db = SessionLocal()
    try:
        db.add(ForecastLog(
            forecast_id=result["forecast_id"],
            target_wmo_id=result["target_float_id"],
            target_cycle=result["target_cycle"],
            predicted_cycle=result["predicted_cycle"],
            model_version=result["model_version"],
            surface_temp_forecast=result["profiles"][0]["temperature_forecast"],
            mixed_layer_depth_m=result["physical_diagnostics"]["mixed_layer_depth_m"],
            is_gravitationally_stable=result["physical_diagnostics"]["is_gravitationally_stable"],
            full_result_json=json.dumps({k: result[k] for k in (
                "profiles", "uncertainty_bounds", "physical_diagnostics", "xai_attribution",
                "evidence_citations", "metrics_comparison")}),
        ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
