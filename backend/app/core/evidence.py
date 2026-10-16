"""
Evidence-Link Cosine Provenance Matcher

Implements composite similarity scoring combining:
- Vector cosine similarity (75%) on T/S profiles
- Haversine spatial proximity (25%) on lat/lon

Matches query profiles against historical database for GDAC NetCDF citations.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from math import radians, sin, cos, sqrt, atan2

from app.models.schema import ArgoProfile, ProfileLevel


# Standard depths for vector comparison
STANDARD_DEPTHS = np.array([5, 20, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)

# Composite weights
COSINE_WEIGHT = 0.75
SPATIAL_WEIGHT = 0.25

# Earth radius in km
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points using Haversine formula.
    
    Returns distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return EARTH_RADIUS_KM * c


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns value in [0, 1], where 1 = identical direction.
    """
    vec1 = np.asarray(vec1, dtype=float)
    vec2 = np.asarray(vec2, dtype=float)
    
    # Handle NaN values
    valid = ~(np.isnan(vec1) | np.isnan(vec2))
    if not np.any(valid):
        return 0.0
    
    v1 = vec1[valid]
    v2 = vec2[valid]
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(v1, v2) / (norm1 * norm2))


def build_profile_vector(profile_id: str, db: Session) -> Tuple[np.ndarray, Dict]:
    """
    Build concatenated T+S vector for a profile from database.
    
    Returns (vector_32, metadata_dict) or (None, {}) if not found.
    """
    levels = db.query(ProfileLevel).filter(
        ProfileLevel.profile_id == profile_id
    ).order_by(ProfileLevel.depth_dbar).all()
    
    if not levels or len(levels) != 16:
        return None, {}
    
    temps = np.array([l.temperature for l in levels], dtype=float)
    sals = np.array([l.salinity for l in levels], dtype=float)
    
    # Concatenate T + S = 32 features
    vector = np.concatenate([temps, sals])
    
    # Get profile metadata
    profile = db.query(ArgoProfile).filter(
        ArgoProfile.profile_id == profile_id
    ).first()
    
    meta = {}
    if profile:
        meta = {
            'wmo_id': profile.wmo_id,
            'cycle': profile.cycle_number,
            'date': str(profile.date),
            'latitude': profile.latitude,
            'longitude': profile.longitude,
            'raw_netcdf_source': profile.raw_netcdf_source,
            'gdac_archive_path': profile.gdac_archive_path,
            'qc_status': profile.qc_status
        }
    
    return vector, meta


def spatial_similarity(lat1: float, lon1: float, lat2: float, lon2: float,
                       max_dist_km: float = 500.0) -> float:
    """
    Compute spatial proximity score from Haversine distance.
    
    Returns similarity in [0, 1], where 1 = same location.
    Exponential decay with distance.
    """
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    
    if dist >= max_dist_km:
        return 0.0
    
    # Exponential decay: similarity = exp(-dist / scale)
    # scale = max_dist / 3 gives ~5% similarity at max_dist
    scale = max_dist_km / 3.0
    return float(np.exp(-dist / scale))


def composite_similarity(query_vec: np.ndarray, query_lat: float, query_lon: float,
                          cand_vec: np.ndarray, cand_lat: float, cand_lon: float,
                          max_dist_km: float = 500.0) -> float:
    """
    Compute composite similarity score.
    
    Score = 0.75 * cosine_similarity + 0.25 * spatial_similarity
    
    Returns value in [0, 1].
    """
    cos_sim = cosine_similarity(query_vec, cand_vec)
    spat_sim = spatial_similarity(query_lat, query_lon, cand_lat, cand_lon, max_dist_km)
    
    return COSINE_WEIGHT * cos_sim + SPATIAL_WEIGHT * spat_sim


_INDEX_CACHE: Dict[str, object] = {"key": None}


def _candidate_index(db: Session) -> Dict[str, np.ndarray]:
    """All QC-pass profiles as one (n, 32) T+S matrix plus metadata, cached in memory.

    Built with two queries and reused until the number of profiles or levels changes,
    so a search over tens of thousands of profiles costs a few milliseconds instead of
    two database round trips per candidate.
    """
    from sqlalchemy import func

    key = (db.query(func.count(ArgoProfile.profile_id)).scalar(),
           db.query(func.count(ProfileLevel.id)).scalar())
    if _INDEX_CACHE.get("key") == key:
        return _INDEX_CACHE["data"]  # type: ignore[return-value]

    profiles = (
        db.query(ArgoProfile)
        .filter(ArgoProfile.qc_status.in_(["QC_PASS_FLAG_1", "QC_PASS_FLAG_2"]))
        .all()
    )
    row = {p.profile_id: i for i, p in enumerate(profiles)}
    depth_col = {int(d): k for k, d in enumerate(STANDARD_DEPTHS)}
    mat = np.full((len(profiles), 2 * len(STANDARD_DEPTHS)), np.nan)
    for pid, depth, temp, sal in db.query(
        ProfileLevel.profile_id, ProfileLevel.depth_dbar, ProfileLevel.temperature, ProfileLevel.salinity
    ):
        i = row.get(pid)
        k = depth_col.get(int(depth))
        if i is not None and k is not None:
            mat[i, k] = temp
            mat[i, len(STANDARD_DEPTHS) + k] = sal
    complete = ~np.isnan(mat).any(axis=1)
    keep = np.where(complete)[0]
    data = {
        "vectors": mat[keep],
        "norms": np.linalg.norm(mat[keep], axis=1),
        "lat": np.array([profiles[i].latitude for i in keep], dtype=float),
        "lon": np.array([profiles[i].longitude for i in keep], dtype=float),
        "wmo": np.array([profiles[i].wmo_id for i in keep]),
        "profiles": [profiles[i] for i in keep],
    }
    _INDEX_CACHE.update(key=key, data=data)
    return data


def _haversine_many(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def find_analogues(wmo_id: str, cycle: int, db: Session,
                   n_results: int = 3,
                   max_distance_km: float = 500.0,
                   min_cosine: float = 0.5,
                   exclude_same_float: bool = True,
                   query_vector: Optional[np.ndarray] = None,
                   query_lat: Optional[float] = None,
                   query_lon: Optional[float] = None) -> List[Dict]:
    """
    Find the top N historical profiles matching the query profile.

    Score = 0.75 * cosine(T+S vectors) + 0.25 * exp(-3 d / d_max), with d the Haversine
    distance and d_max = max_distance_km (spatial term is 0 beyond d_max). Only QC flag
    1/2 profiles are candidates; profiles of the query float are excluded by default.

    Returns:
        List of dicts with match metadata and similarity scores, best first.
    """
    if query_vector is not None:
        # Search for analogues of a given T+S vector (e.g. a forecast) at a given position.
        query_vec = np.asarray(query_vector, dtype=float)
        q_lat, q_lon = float(query_lat or 0.0), float(query_lon or 0.0)
    else:
        query_profile_id = f"ARGO_{wmo_id}_CYC{cycle:03d}"
        query_vec, query_meta = build_profile_vector(query_profile_id, db)
        if query_vec is None:
            return []
        q_lat = float(query_meta.get('latitude', 0))
        q_lon = float(query_meta.get('longitude', 0))

    idx = _candidate_index(db)
    if len(idx["vectors"]) == 0:
        return []
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return []
    cos = idx["vectors"] @ query_vec / (idx["norms"] * q_norm)
    dist = _haversine_many(q_lat, q_lon, idx["lat"], idx["lon"])
    spat = np.where(dist < max_distance_km, np.exp(-dist / (max_distance_km / 3.0)), 0.0)
    score = COSINE_WEIGHT * cos + SPATIAL_WEIGHT * spat
    ok = cos >= min_cosine
    if exclude_same_float:
        ok &= idx["wmo"] != str(wmo_id)
    cand = np.where(ok)[0]
    if cand.size == 0:
        return []
    # best profile per float, so the citations come from n_results different floats
    top, seen = [], set()
    for i in cand[np.argsort(-score[cand], kind="stable")]:
        if idx["wmo"][i] in seen:
            continue
        seen.add(idx["wmo"][i])
        top.append(i)
        if len(top) == n_results:
            break

    results = []
    for i in top:
        p = idx["profiles"][i]
        results.append({
            'profile_id': p.profile_id,
            'wmo_id': p.wmo_id,
            'cycle': p.cycle_number,
            'date': str(p.date),
            'latitude': p.latitude,
            'longitude': p.longitude,
            'distance_km': float(dist[i]),
            'cosine_similarity': float(cos[i]),
            'spatial_similarity': float(spat[i]),
            'composite_score': float(score[i]),
            'raw_netcdf_source': p.raw_netcdf_source,
            'gdac_archive_path': p.gdac_archive_path,
            'qc_status': p.qc_status,
        })
    return results


def format_evidence_citation(match: Dict) -> Dict:
    """
    Format a match into the EvidenceCitation schema for API response.
    """
    return {
        'profile_id': match['profile_id'],
        'wmo_id': match['wmo_id'],
        'cycle': match['cycle'],
        'date': match['date'],
        'latitude': match.get('latitude', 0.0),
        'longitude': match.get('longitude', 0.0),
        'distance_km': round(match['distance_km'], 1),
        'cosine_similarity': round(match['cosine_similarity'], 4),
        'spatial_similarity': round(match['spatial_similarity'], 4),
        'composite_score': round(match['composite_score'], 4),
        'raw_netcdf_source': match['raw_netcdf_source'],
        'gdac_archive_path': match['gdac_archive_path'],
        'qc_status': match['qc_status']
    }


def get_evidence_links(wmo_id: str, cycle: int, db: Session,
                       n_results: int = 3,
                       query_vector: Optional[np.ndarray] = None,
                       query_lat: Optional[float] = None,
                       query_lon: Optional[float] = None) -> List[Dict]:
    """
    Main entry point: formatted evidence citations.

    Without query_vector the stored profile (wmo_id, cycle) is the query; with it, the
    given T+S vector (for example a forecast) at (query_lat, query_lon) is the query.
    """
    matches = find_analogues(wmo_id, cycle, db, n_results=n_results, query_vector=query_vector,
                             query_lat=query_lat, query_lon=query_lon)
    return [format_evidence_citation(m) for m in matches]


if __name__ == "__main__":
    # Quick test with mock data
    print("Testing Evidence-Link matcher...")
    
    # Test cosine similarity
    v1 = np.array([1.0, 2.0, 3.0, 4.0])
    v2 = np.array([1.0, 2.0, 3.0, 4.0])
    v3 = np.array([4.0, 3.0, 2.0, 1.0])
    
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-6
    assert cosine_similarity(v1, v3) < 1.0
    
    # Test haversine
    dist = haversine_distance(20.0, 65.0, 20.5, 65.5)
    print(f"Haversine test: {dist:.1f} km")
    
    print("Evidence module basic tests passed")