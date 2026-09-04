"""Copernicus Data Space Ecosystem downloader for Sentinel-3 auxiliary orbit files."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from lxml import etree

from downloaders.download_utils import atomic_output_file, cleanup_temp_file, retry_with_backoff
from downloaders.net_guard import (
    assert_allowed_url_args,
    assert_path_within,
    guarded_request,
    sanitize_remote_filename,
)


def _hardened_xml_parser() -> etree.XMLParser:
    """Return a hardened parser for provider XML (no entities, no network)."""
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        dtd_validation=False,
        load_dtd=False,
    )


TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL_TEMPLATE = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LEO-Orbit-Dataset/1.0)",
}
PAGE_SIZE = 1000


def _curl_json(args: List[str]) -> Dict:
    """Run curl and parse a JSON response."""
    assert_allowed_url_args(args)
    result = retry_with_backoff(
        lambda: subprocess.run(
            ["curl", "-sS", *args],
            check=True,
            capture_output=True,
            text=True,
        ),
        attempts=6,
        base_delay_sec=2.0,
        max_delay_sec=20.0,
        label="CDSE curl json",
        verbose=False,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Temporary failure: CDSE curl response was not JSON: {exc}") from exc


def _access_token_from_payload(payload: Dict) -> str:
    """Extract a CDSE access token with an explicit retryable error on bad JSON."""
    token = payload.get("access_token")
    if not token:
        keys = ",".join(sorted(str(key) for key in payload.keys())[:5])
        raise RuntimeError(f"Temporary failure: CDSE token response missing access_token; keys={keys}")
    return str(token)


def _curl_download(args: List[str], output_path: str) -> None:
    """Run curl and stream a response to disk."""
    assert_allowed_url_args(args)
    retry_with_backoff(
        lambda: subprocess.run(
            ["curl", "-sS", "-L", *args, "-o", output_path],
            check=True,
            capture_output=True,
            text=False,
        ),
        attempts=6,
        base_delay_sec=2.0,
        max_delay_sec=20.0,
        label=f"CDSE curl download {output_path}",
        verbose=False,
    )


def _looks_like_xml(path: str) -> bool:
    """Return True when the downloaded file parses as XML (hardened parser)."""
    try:
        etree.parse(path, parser=_hardened_xml_parser())
    except Exception:
        return False
    return True


def get_cdse_token(username: str, password: str) -> str:
    """Fetch a CDSE access token."""

    def _request_token() -> str:
        response = guarded_request(
            "POST",
            TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": username,
                "password": password,
            },
            headers=HEADERS,
            timeout=60,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Temporary failure: CDSE token response was not JSON: {exc}") from exc
        return _access_token_from_payload(payload)

    try:
        return retry_with_backoff(
            _request_token,
            attempts=6,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
            label="CDSE token request",
            verbose=False,
        )
    except Exception as request_exc:
        payload = _curl_json(
            [
                "-X",
                "POST",
                TOKEN_URL,
                "-H",
                "Content-Type: application/x-www-form-urlencoded",
                "--data-urlencode",
                "client_id=cdse-public",
                "--data-urlencode",
                "grant_type=password",
                "--data-urlencode",
                f"username={username}",
                "--data-urlencode",
                f"password={password}",
            ]
        )
        try:
            return _access_token_from_payload(payload)
        except RuntimeError as curl_exc:
            raise RuntimeError(f"CDSE token request failed: {request_exc}; curl fallback failed: {curl_exc}") from curl_exc


def list_cdse_sentinel3_orbits(
    satellite: str,
    start_date: datetime,
    end_date: datetime,
    product_type: str = "AUX_POEORB",
    access_token: str | None = None,
    verbose: bool = True,
) -> List[Dict]:
    """List Sentinel-3 orbit products from CDSE OData."""
    if access_token is None:
        username = os.environ.get("CDSE_USERNAME")
        password = os.environ.get("CDSE_PASSWORD")
        if not username or not password:
            raise RuntimeError("CDSE_USERNAME and CDSE_PASSWORD must be set for Sentinel-3 orbit downloads")
        access_token = get_cdse_token(username, password)

    satellite = satellite.upper()
    filter_start = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    filter_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z")
    base_params = {
        "$filter": (
            "Collection/Name eq 'SENTINEL-3' and "
            f"startswith(Name,'{satellite}_') and "
            "Attributes/OData.CSC.StringAttribute/any("
            "att:att/Name eq 'productType' and "
            f"att/OData.CSC.StringAttribute/Value eq '{product_type}') and "
            f"ContentDate/Start ge {filter_start} and "
            f"ContentDate/Start lt {filter_end}"
        ),
        "$orderby": "ContentDate/Start asc",
        "$top": str(PAGE_SIZE),
    }
    headers = {
        **HEADERS,
        "Authorization": f"Bearer {access_token}",
    }

    if verbose:
        print(f"Querying CDSE for {satellite} {product_type} products...")
    rows: List[Dict] = []
    skip = 0
    while True:
        params = {**base_params, "$skip": str(skip)}
        try:
            response = retry_with_backoff(
                lambda: guarded_request("GET", ODATA_PRODUCTS_URL, params=params, headers=headers, timeout=120),
                attempts=6,
                base_delay_sec=2.0,
                max_delay_sec=20.0,
                label=f"CDSE listing page skip={skip}",
                verbose=False,
            )
            response.raise_for_status()
            page_rows = response.json().get("value", [])
        except requests.RequestException:
            query_args = []
            for key, value in params.items():
                query_args.extend(["--data-urlencode", f"{key}={value}"])
            payload = _curl_json(
                [
                    "-G",
                    ODATA_PRODUCTS_URL,
                    "-H",
                    f"Authorization: Bearer {access_token}",
                    *query_args,
                ]
            )
            page_rows = payload.get("value", [])

        rows.extend(page_rows)
        if len(page_rows) < PAGE_SIZE:
            break
        skip += PAGE_SIZE

    products = []
    for row in rows:
        products.append(
            {
                "id": row.get("Id"),
                "filename": row.get("Name"),
                "url": DOWNLOAD_URL_TEMPLATE.format(product_id=row.get("Id")),
                "s3_path": row.get("S3Path"),
                "content_start": row.get("ContentDate", {}).get("Start"),
                "content_end": row.get("ContentDate", {}).get("End"),
                "publication_date": row.get("PublicationDate"),
                "modification_date": row.get("ModificationDate"),
                "product_type": product_type,
                "satellite": satellite.upper(),
            }
        )

    if verbose:
        print(f"  Found {len(products)} products")

    return products


def download_cdse_product(product_info: Dict, output_dir: str, access_token: str, verbose: bool = True) -> Optional[str]:
    """Download a single CDSE product."""
    os.makedirs(output_dir, exist_ok=True)
    filename = sanitize_remote_filename(str(product_info["filename"]))
    local_path = os.path.join(output_dir, filename)
    assert_path_within(local_path, output_dir)

    if os.path.exists(local_path):
        if _looks_like_xml(local_path):
            if verbose:
                print(f"  Skipping {filename} (already exists)")
            return local_path
        if verbose:
            print(f"  Existing {filename} is invalid, re-downloading")

    headers = {
        **HEADERS,
        "Authorization": f"Bearer {access_token}",
    }
    if verbose:
        print(f"  Downloading {filename}...")

    try:
        response = retry_with_backoff(
            lambda: guarded_request("GET", product_info["url"], headers=headers, timeout=300, stream=True),
            attempts=6,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
            label=f"CDSE product {filename}",
            verbose=False,
        )
        response.raise_for_status()
        with atomic_output_file(local_path, "wb") as (handle, _):
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
    except requests.RequestException:
        _curl_download(
            [
                "-H",
                f"Authorization: Bearer {access_token}",
                product_info["url"],
            ],
            output_path=local_path,
        )
    except Exception:
        cleanup_temp_file(local_path)
        raise

    if not _looks_like_xml(local_path):
        try:
            os.remove(local_path)
        except OSError:
            pass
        raise RuntimeError(f"Downloaded CDSE product is not valid XML for {filename}")

    return local_path


def download_cdse_sentinel3_orbits(
    satellite: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: str,
    product_type: str = "AUX_POEORB",
    max_files: int | None = None,
    verbose: bool = True,
) -> List[str]:
    """Download Sentinel-3 orbit products from CDSE."""
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("CDSE_USERNAME and CDSE_PASSWORD must be set for Sentinel-3 orbit downloads")

    token = get_cdse_token(username, password)
    products = list_cdse_sentinel3_orbits(
        satellite=satellite,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
        access_token=token,
        verbose=verbose,
    )

    if max_files is not None:
        products = products[:max_files]

    downloaded = []
    for product in products:
        local_path = download_cdse_product(product, output_dir=output_dir, access_token=token, verbose=verbose)
        if local_path:
            downloaded.append(local_path)

    if verbose:
        print(f"Downloaded {len(downloaded)} files")

    return downloaded
