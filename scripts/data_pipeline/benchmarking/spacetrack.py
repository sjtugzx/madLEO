"""Minimal Space-Track clients for TLE, metadata, and public ephemeris downloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from downloaders.download_utils import atomic_output_file, cleanup_temp_file


LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
TLE_QUERY_TEMPLATE = (
    "https://www.space-track.org/basicspacedata/query/class/gp_history/"
    "NORAD_CAT_ID/{norad}/EPOCH/{start}--{end}/orderby/EPOCH asc/format/tle"
)
SATCAT_QUERY_TEMPLATE = (
    "https://www.space-track.org/basicspacedata/query/class/satcat/"
    "NORAD_CAT_ID/{norad}/format/json"
)
PUBLIC_DIRS_URL = "https://www.space-track.org/publicfiles/query/class/dirs"
PUBLIC_DETAILS_URL = "https://www.space-track.org/publicfiles/query/class/loadpublicdata"
PUBLIC_DOWNLOAD_URL = "https://www.space-track.org/publicfiles/query/class/download?name={link}"
RETRY_STRATEGY = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)


class SpaceTrackClient:
    """Small authenticated client for benchmark download scripts."""

    def __init__(self, identity: str, password: str):
        self.identity = identity
        self.password = password
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @classmethod
    def from_env(cls) -> "SpaceTrackClient":
        identity = os.environ.get("SPACETRACK_ID")
        password = os.environ.get("SPACETRACK_PASSWORD")
        if not identity or not password:
            raise RuntimeError("SPACETRACK_ID and SPACETRACK_PASSWORD must be set in the environment")
        return cls(identity=identity, password=password)

    def login(self) -> None:
        response = self.session.post(LOGIN_URL, data={"identity": self.identity, "password": self.password}, timeout=60)
        response.raise_for_status()

    def download_tle_history(self, norad: str, start: str, end: str, output_dir: Path) -> Path:
        """Download TLE history in raw line-pair format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        query = TLE_QUERY_TEMPLATE.format(norad=norad, start=start, end=end)
        response = self.session.get(query, timeout=120)
        response.raise_for_status()
        output_path = output_dir / f"{norad}_{start}_{end}.tle"
        with atomic_output_file(output_path, "w") as (handle, _):
            handle.write(response.text)
        return output_path

    def get_satcat_metadata(self, norad: str) -> Dict:
        """Fetch SATCAT metadata for one NORAD object."""
        query = SATCAT_QUERY_TEMPLATE.format(norad=norad)
        response = self.session.get(query, timeout=60)
        response.raise_for_status()
        rows = response.json()
        return rows[0] if rows else {}

    def list_public_ephemeris(self, name_contains: Iterable[str] | None = None) -> List[Dict]:
        """List public ephemeris-like files from Space-Track public files."""
        self.session.get(PUBLIC_DIRS_URL, timeout=60).raise_for_status()
        response = self.session.get(PUBLIC_DETAILS_URL, timeout=60)
        response.raise_for_status()
        rows = [row for row in response.json() if row.get("type") == "Ephemeris"]
        if name_contains:
            terms = [term.lower() for term in name_contains if term]
            rows = [row for row in rows if _row_matches_terms(row, terms)]
        return rows

    def download_public_files(self, file_rows: Iterable[Dict], output_dir: Path, limit: int | None = None) -> List[Path]:
        """Download selected public files to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[Path] = []
        for index, row in enumerate(file_rows):
            if limit is not None and index >= limit:
                break
            filename = row["name"].replace(":", "_")
            response = self.session.get(PUBLIC_DOWNLOAD_URL.format(link=row["link"]), timeout=120)
            response.raise_for_status()
            output_path = output_dir / filename
            with atomic_output_file(output_path, "wb") as (handle, _):
                handle.write(response.content)
            downloaded.append(output_path)
        return downloaded


def _row_matches_terms(row: Dict, terms: Iterable[str]) -> bool:
    """Case-insensitive substring matching for public file rows."""
    haystack = " ".join(str(row.get(field, "")) for field in ["name", "link", "description", "folder"]).lower()
    return all(term in haystack for term in terms)
