"""
FSA Food Hygiene Ratings ingest.

Usage:
    python -m src.ingest.fsa_ingest            # full fetch → data/raw/fsa_full.parquet
    python -m src.ingest.fsa_ingest --daily    # upsert authorities published since yesterday
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.ratings.food.gov.uk"
HEADERS = {"x-api-version": "2", "accept": "application/json"}
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_PATH = BASE_DIR / "data" / "raw" / "fsa_full.parquet"
PAGE_SIZE = 5000

FIELDS = [
    "FHRSID", "BusinessName", "BusinessTypeID", "BusinessType",
    "AddressLine1", "AddressLine2", "PostCode",
    "Latitude", "Longitude", "RatingValue", "RatingDate",
    "scores_Hygiene", "scores_Structure", "scores_ConfidenceInManagement",
    "LocalAuthorityName", "LocalAuthorityCode",
]


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_authorities(session: requests.Session) -> list[dict]:
    url = f"{BASE_URL}/Authorities/basic?pageSize=500"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["authorities"]


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_date(val):
    if not val:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(val[:19], fmt)
        except (ValueError, TypeError):
            continue
    return None


def parse_establishment(est: dict) -> dict:
    scores = est.get("scores") or {}
    return {
        "FHRSID": est.get("FHRSID"),
        "BusinessName": est.get("BusinessName"),
        "BusinessTypeID": _safe_int(est.get("BusinessTypeID")),
        "BusinessType": est.get("BusinessType"),
        "AddressLine1": est.get("AddressLine1"),
        "AddressLine2": est.get("AddressLine2"),
        "PostCode": est.get("PostCode"),
        "Latitude": _safe_float(est.get("geocode", {}).get("latitude") if est.get("geocode") else est.get("Latitude")),
        "Longitude": _safe_float(est.get("geocode", {}).get("longitude") if est.get("geocode") else est.get("Longitude")),
        "RatingValue": str(est.get("RatingValue") or ""),
        "RatingDate": _safe_date(est.get("RatingDate")),
        "scores_Hygiene": _safe_float(scores.get("Hygiene")),
        "scores_Structure": _safe_float(scores.get("Structure")),
        "scores_ConfidenceInManagement": _safe_float(scores.get("ConfidenceInManagement")),
        "LocalAuthorityName": est.get("LocalAuthorityName"),
        "LocalAuthorityCode": est.get("LocalAuthorityCode"),
    }


def fetch_establishments(session: requests.Session, authority_id: int) -> list[dict]:
    records = []
    page = 1
    while True:
        url = (
            f"{BASE_URL}/Establishments"
            f"?localAuthorityId={authority_id}&pageSize={PAGE_SIZE}&pageNumber={page}"
        )
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        establishments = data.get("establishments") or []
        records.extend(parse_establishment(e) for e in establishments)

        meta = data.get("meta", {})
        total_pages = meta.get("totalPages", 1)
        if page >= total_pages:
            break
        page += 1
    return records


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=FIELDS)
    # Ensure RatingDate is datetime
    if "RatingDate" in df.columns:
        df["RatingDate"] = pd.to_datetime(df["RatingDate"], errors="coerce")
    return df


def upsert_parquet(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Upsert new_df into existing parquet on (BusinessName, PostCode) key."""
    if not path.exists():
        return new_df
    existing = pd.read_parquet(path)
    key = ["BusinessName", "PostCode"]
    # Drop old rows that match any key in new_df
    mask = existing.set_index(key).index.isin(new_df.set_index(key).index)
    existing = existing[~mask]
    combined = pd.concat([existing, new_df], ignore_index=True)
    return combined


def run(daily: bool = False) -> None:
    session = get_session()

    print("Fetching authority list...")
    authorities = fetch_authorities(session)
    total = len(authorities)
    print(f"Found {total} authorities.")

    if daily:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        authorities = [
            a for a in authorities
            if a.get("lastPublished") and
            datetime.fromisoformat(a["lastPublished"].replace("Z", "+00:00")) >= yesterday
        ]
        print(f"Daily mode: {len(authorities)} authorities updated since yesterday.")

    all_records: list[dict] = []

    for idx, auth in enumerate(authorities, 1):
        auth_id = auth.get("LocalAuthorityId") or auth.get("localAuthorityId")
        name = auth.get("Name") or auth.get("name") or str(auth_id)
        print(f"[{idx}/{len(authorities)}] {name} ...", end=" ", flush=True)

        try:
            records = fetch_establishments(session, auth_id)
        except requests.HTTPError as exc:
            print(f"ERROR {exc}")
            continue

        print(f"{len(records)} establishments")
        all_records.extend(records)

    new_df = build_dataframe(all_records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if daily:
        final_df = upsert_parquet(new_df, OUT_PATH)
    else:
        final_df = new_df

    final_df.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"\nSaved {len(final_df):,} records -> {OUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FSA food hygiene ingest")
    parser.add_argument("--daily", action="store_true", help="Only fetch recently updated authorities")
    args = parser.parse_args()
    run(daily=args.daily)


if __name__ == "__main__":
    main()
