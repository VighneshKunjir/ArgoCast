"""Floats and profiles endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.schema import ArgoFloat, ArgoProfile, ProfileLevel
from app.schemas.forecast import FloatSummary, ArgoProfile as ArgoProfileSchema, ArgoMeasurement

router = APIRouter()


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


@router.get("/floats", response_model=List[FloatSummary])
async def get_floats():
    """
    Get list of all available Argo floats with their cycles.
    
    Returns catalog of active floats in the Arabian Sea with base coordinates and cycle lists.
    """
    db = get_db()
    try:
        floats = db.query(ArgoFloat).order_by(ArgoFloat.wmo_id.asc()).all()
        cycles_by_float, dates_by_float = {}, {}
        for wmo, cyc, day in db.query(ArgoProfile.wmo_id, ArgoProfile.cycle_number, ArgoProfile.date):
            cycles_by_float.setdefault(wmo, []).append(cyc)
            dates_by_float.setdefault(wmo, []).append(day)

        result = []
        for f in floats:
            cycles = sorted(cycles_by_float.get(f.wmo_id, []))
            # Default base cycle: the second-to-last one, so the default forecast targets the
            # float's latest profile and can be checked against it. The latest cycle itself
            # gives a true look-ahead forecast.
            default_cycle = cycles[-2] if len(cycles) >= 4 else (cycles[-1] if cycles else 0)
            result.append(FloatSummary(
                wmo_id=f.wmo_id,
                name=f.name,
                baseLat=f.base_lat,
                baseLon=f.base_lon,
                cycles=cycles,
                defaultCycle=default_cycle,
                firstDate=str(min(dates_by_float[f.wmo_id])) if f.wmo_id in dates_by_float else None,
                lastDate=str(max(dates_by_float[f.wmo_id])) if f.wmo_id in dates_by_float else None,
            ))
        return result
    finally:
        db.close()


@router.get("/profiles/{wmoId}", response_model=list)
async def get_profiles(wmoId: str):
    """
    Get all historical profiles for a given float ID.
    
    Returns all historical vertical profile cycles for a float, sorted by cycle number.
    """
    db = get_db()
    try:
        profiles = db.query(ArgoProfile).filter(
            ArgoProfile.wmo_id == wmoId
        ).order_by(ArgoProfile.cycle_number.asc()).all()
        
        if not profiles:
            raise HTTPException(status_code=404, detail=f"Float {wmoId} not found")
        
        levels_by_profile = {}
        for lv in (
            db.query(ProfileLevel)
            .join(ArgoProfile, ArgoProfile.profile_id == ProfileLevel.profile_id)
            .filter(ArgoProfile.wmo_id == wmoId)
            .order_by(ProfileLevel.profile_id, ProfileLevel.depth_dbar.asc())
        ):
            levels_by_profile.setdefault(lv.profile_id, []).append(lv)

        result = []
        for p in profiles:
            qc = 1 if p.qc_status == "QC_PASS_FLAG_1" else 2
            measurements = [
                {
                    "depth_dbar": l.depth_dbar,
                    "temperature": l.temperature,
                    "salinity": l.salinity,
                    "qc_temperature": qc,
                    "qc_salinity": qc,
                }
                for l in levels_by_profile.get(p.profile_id, [])
            ]
            result.append({
                "profile_id": p.profile_id,
                "wmo_id": p.wmo_id,
                "float_name": f"Argo float {p.wmo_id}",
                "platform_type": "Argo profiling float",
                "cycle_number": p.cycle_number,
                "date": p.date.isoformat() if p.date else "",
                "latitude": p.latitude,
                "longitude": p.longitude,
                "sea_region": "Arabian Sea (Northern Indian Ocean)",
                "data_mode": "D" if qc == 1 else "A",
                "qc_status": p.qc_status,
                "raw_netcdf_source": p.raw_netcdf_source,
                "gdac_archive_path": p.gdac_archive_path,
                "measurements": measurements
            })

        return result
    finally:
        db.close()