"""Process Jason-3 PO.DAAC OGDR GPS granules into benchmark state vectors.

Output velocity columns are ECEF (Earth-fixed) velocities obtained by
FINITE DIFFERENCING the geodetic position series in time -- they are
rotating-frame velocities, not inertial ones, and need the omega x r
correction before any inertial-frame use (REVIEW_FINDINGS 6.2.2 / 6.3.4).

All decoding metadata is read from the product itself (CF 1.7,
docs/standards/cf-conventions-1.7.pdf; Jason-3 Products Handbook,
docs/standards item 8): the time reference epoch comes from the ``units``
attribute ("seconds since <datetime>"), ``calendar`` is asserted Gregorian,
``tai_utc_difference`` is cross-validated against the ``time_tai``
companion variable, and the geodetic -> ECEF conversion uses the declared
``ellipsoid_axis`` / ``ellipsoid_flattening`` root attributes.  Jason-1/2
GDR files (no ``data_01`` group) are explicitly rejected with
:class:`UnsupportedProductError` (REVIEW_FINDINGS 6.3.4: implement a GDR
parser or reject with a clear counted error -- never a silent skip).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
import pandas as pd


class UnsupportedProductError(ValueError):
    """netCDF orbit product this parser explicitly does not support."""


# CF 1.7 time coordinate units: "<unit> since <reference date and time>".
_TIME_UNITS_PATTERN = re.compile(r"^\s*seconds\s+since\s+(.+?)\s*$")
_GREGORIAN_CALENDARS = {"gregorian", "standard", "proleptic_gregorian"}
_UNITS_EPOCH_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)
# TAI/UTC cross-check tolerance (s): the stored tai_utc_difference and the
# measured time_tai - time offset agree to float rounding (<1e-6 s).
_TAI_UTC_TOLERANCE_S = 1e-3


def _decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _first_attr_float(attrs: dict, key: str) -> float | None:
    value = attrs.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0]) if len(array) else None


def _read_dataset(handle: h5py.File, candidates: Iterable[str]) -> np.ndarray:
    for key in candidates:
        if key in handle:
            return _decode_dataset(handle[key])
    raise KeyError(f"None of the dataset paths exist: {list(candidates)}")


def _decode_dataset(dataset: h5py.Dataset) -> np.ndarray:
    """Decode netCDF-style scale_factor/add_offset datasets from HDF5."""
    values = dataset[()]
    values = np.asarray(values, dtype=float)

    fill_value = dataset.attrs.get("_FillValue")
    if fill_value is not None:
        fill_array = np.asarray(fill_value, dtype=float).reshape(-1)
        if len(fill_array) > 0:
            values[values == fill_array[0]] = np.nan

    scale_factor = dataset.attrs.get("scale_factor")
    if scale_factor is not None:
        scale_array = np.asarray(scale_factor, dtype=float).reshape(-1)
        if len(scale_array) > 0:
            values = values * scale_array[0]

    add_offset = dataset.attrs.get("add_offset")
    if add_offset is not None:
        offset_array = np.asarray(add_offset, dtype=float).reshape(-1)
        if len(offset_array) > 0:
            values = values + offset_array[0]

    return values


def _time_dataset(handle: h5py.File) -> h5py.Dataset:
    for key in ("/data_01/time", "data_01/time"):
        if key in handle:
            return handle[key]
    raise UnsupportedProductError(
        "jason_gdr_without_data_01_group: no data_01/time variable; "
        "Jason-1/2 GDR granules (root-level time/lat/lon) are explicitly "
        "not supported by the Jason-3 OGDR parser (REVIEW_FINDINGS 6.3.4)"
    )


def _time_reference_epoch(time_dataset: h5py.Dataset) -> datetime:
    """Reference epoch declared by the time variable's ``units`` attribute.

    CF 1.7 defines the time coordinate's ``units`` as
    "<unit> since <reference date and time>"; the reference epoch must come
    from this declaration, never from a hard-coded constant
    (REVIEW_FINDINGS 6.2.1 rule 2).  A missing/unparsable ``units`` or a
    non-Gregorian ``calendar`` blocks the parse.
    """
    units = time_dataset.attrs.get("units")
    if units is None:
        raise UnsupportedProductError("time_units_attribute_missing")
    match = _TIME_UNITS_PATTERN.match(_decode_attr(units))
    if match is None:
        raise UnsupportedProductError(
            f"time_units_not_seconds_since_reference:{_decode_attr(units)!r}"
        )
    reference_text = match.group(1)
    for fmt in _UNITS_EPOCH_FORMATS:
        try:
            epoch = datetime.strptime(reference_text, fmt)
            break
        except ValueError:
            continue
    else:
        raise UnsupportedProductError(
            f"time_units_reference_unparsable:{reference_text!r}"
        )
    calendar = time_dataset.attrs.get("calendar")
    if calendar is not None and _decode_attr(calendar) not in _GREGORIAN_CALENDARS:
        raise UnsupportedProductError(
            f"time_calendar_not_gregorian:{_decode_attr(calendar)!r}"
        )
    return epoch


def _crosscheck_tai_utc(handle: h5py.File, time: np.ndarray, time_dataset: h5py.Dataset) -> None:
    """Validate ``tai_utc_difference`` against the ``time_tai`` variable.

    The products declare the TAI-UTC offset at the first measurement
    (Jason-3 handbook); the companion ``time_tai`` variable measures the
    same offset for every row.  A disagreement means the metadata is
    internally inconsistent and the parse blocks (REVIEW_FINDINGS 6.3.4
    "TAI/UTC relation cross-validated with time_tai").
    """
    tai_attr = _first_attr_float(time_dataset.attrs, "tai_utc_difference")
    if tai_attr is None:
        return
    for key in ("/data_01/time_tai", "data_01/time_tai"):
        if key not in handle:
            continue
        dataset = handle[key]
        values = np.asarray(dataset[()], dtype=float)
        if len(values) != len(time):
            return
        fill = _first_attr_float(dataset.attrs, "_FillValue")
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(values) & np.isfinite(time)
        if fill is not None:
            valid &= values != fill
        if not valid.any():
            return
        offset = float(np.median((values - time)[valid]))
        if abs(offset - tai_attr) > _TAI_UTC_TOLERANCE_S:
            raise ValueError(
                "tai_utc_crosscheck_failed: time_tai - time = "
                f"{offset!r} s but tai_utc_difference = {tai_attr!r} s"
            )
        return


def _declared_ellipsoid(handle: h5py.File) -> tuple[float, float]:
    """Ellipsoid declared by the product's root attributes (m, flattening)."""
    axis = _first_attr_float(handle.attrs, "ellipsoid_axis")
    flattening = _first_attr_float(handle.attrs, "ellipsoid_flattening")
    if axis is None or flattening is None:
        raise UnsupportedProductError(
            "ellipsoid_not_declared: geodetic -> ECEF conversion requires "
            "the ellipsoid_axis/ellipsoid_flattening attributes "
            "(REVIEW_FINDINGS 6.2.3: conversions from declarations only)"
        )
    return axis, flattening


def _geodetic_to_ecef(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    height_m: np.ndarray,
    semimajor_axis_m: float,
    flattening: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form geodetic -> ECEF on the DECLARED ellipsoid (m).

    Standard ellipsoidal datum relations (IERS Conventions 2010 TN36
    ch.1, docs/standards/iers-tn36-chapter1-definitions.pdf):

        N = a / sqrt(1 - e^2 sin^2 lat),  e^2 = 2f - f^2
        x = (N + h) cos(lat) cos(lon)
        y = (N + h) cos(lat) sin(lon)
        z = (N (1 - e^2) + h) sin(lat)

    astropy's ``EarthLocation.from_geodetic`` only accepts the three named
    ellipsoids (no custom a/f in the pinned version), so the declared
    ellipsoid (Jason-3 OGDR: a = 6378137 m; Jason-1 GDR: TOPEX a =
    6378136.3 m) is applied here directly.
    """
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    height = np.asarray(height_m, dtype=float)
    a = float(semimajor_axis_m)
    f = float(flattening)
    eccentricity_sq = 2.0 * f - f * f
    sin_lat = np.sin(lat)
    prime_vertical = a / np.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
    x = (prime_vertical + height) * np.cos(lat) * np.cos(lon)
    y = (prime_vertical + height) * np.cos(lat) * np.sin(lon)
    z = (prime_vertical * (1.0 - eccentricity_sq) + height) * sin_lat
    return x, y, z


def _load_required_ogdr_arrays(handle: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the minimum required Jason-3 OGDR arrays."""
    if not any(key in handle for key in ("/data_01", "data_01")):
        raise UnsupportedProductError(
            "jason_gdr_without_data_01_group: Jason-1/2 GDR granules "
            "(root-level time/lat/lon, no data_01 group) are explicitly "
            "not supported by the Jason-3 OGDR parser (REVIEW_FINDINGS 6.3.4)"
        )
    time = _read_dataset(handle, ["/data_01/time", "data_01/time"]).astype(float)
    latitude = _read_dataset(handle, ["/data_01/latitude", "data_01/latitude"]).astype(float)
    longitude = _read_dataset(handle, ["/data_01/longitude", "data_01/longitude"]).astype(float)
    altitude = _read_dataset(handle, ["/data_01/gps_altitude", "data_01/gps_altitude"]).astype(float)
    return time, latitude, longitude, altitude


def validate_jason3_ogdr_file(filepath: str) -> str | None:
    """Return a one-line validation error for a broken Jason-3 granule, else None."""
    try:
        with h5py.File(filepath, "r") as handle:
            _load_required_ogdr_arrays(handle)
        return None
    except Exception as exc:
        return str(exc).splitlines()[0]


def process_jason3_ogdr_file(filepath: str, sat_id: str = "jason-3") -> pd.DataFrame:
    """Convert one Jason-3 OGDR GPS netCDF granule into a state-vector DataFrame.

    The velocity columns are obtained by FINITE DIFFERENCING the ECEF
    position series: they are Earth-fixed rotating-frame velocities, not
    inertial velocities (see module docstring; REVIEW_FINDINGS 6.3.4).
    """
    with h5py.File(filepath, "r") as handle:
        time, latitude, longitude, altitude = _load_required_ogdr_arrays(handle)
        time_dataset = _time_dataset(handle)
        reference_epoch = _time_reference_epoch(time_dataset)
        _crosscheck_tai_utc(handle, time, time_dataset)
        semimajor_axis_m, flattening = _declared_ellipsoid(handle)

    epochs = pd.to_datetime([reference_epoch + timedelta(seconds=float(value)) for value in time])
    x, y, z = _geodetic_to_ecef(
        longitude, latitude, altitude, semimajor_axis_m=semimajor_axis_m, flattening=flattening
    )

    dt_seconds = np.gradient(time)
    dt_seconds[dt_seconds == 0] = np.nan
    vx = np.gradient(x) / dt_seconds
    vy = np.gradient(y) / dt_seconds
    vz = np.gradient(z) / dt_seconds

    df = pd.DataFrame(
        {
            "sat_id": sat_id,
            "epoch": epochs,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
        }
    )
    return df.dropna(subset=["epoch", "x", "y", "z", "vx", "vy", "vz"]).reset_index(drop=True)


def process_jason3_ogdr_directory(input_dir: str, sat_id: str = "jason-3") -> pd.DataFrame:
    """Process all Jason-3 OGDR granules in a directory."""
    frames: List[pd.DataFrame] = []
    for path in sorted(Path(input_dir).glob("*.nc*")):
        try:
            frames.append(process_jason3_ogdr_file(str(path), sat_id=sat_id))
        except Exception as exc:
            print(f"Error processing {path}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("epoch").reset_index(drop=True)
