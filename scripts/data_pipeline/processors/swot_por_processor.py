"""Process SWOT CNES POR/MOE netCDF orbit products into state vectors.

SWOT precise/medium orbit ephemeris files (``SWOT_POR_*.nc``) store ECEF
(ITRF) position and velocity vectors against a UTC time base declared by
the ``time`` variable's ``units`` attribute ("seconds since <datetime>",
CF 1.7 -- read from the file, never hard-coded).  The output DataFrame
uses the same unified layout as the other processors: sat_id, epoch,
x..z (m), vx..vz (m/s, still in the rotating ECEF frame).

Decoding is CF-aware (docs/standards/cf-conventions-1.7.pdf,
REVIEW_FINDINGS 6.3.5): ``_FillValue``/``scale_factor``/``add_offset`` are
all honored; the ``orbit_qual`` fill/sentinel value 127 maps to NaN with a
counted ledger entry on ``df.attrs``; the global ``reference_frame``
attribute is surfaced on ``df.attrs`` (frame declared by the product, not
assumed; REVIEW_FINDINGS 6.2.2 rule 1).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd

# CF 1.7 time-units decoder shared with the Jason-3 OGDR parser -- the
# reference epoch of a netCDF time variable is read from its ``units``
# attribute, never hard-coded (REVIEW_FINDINGS 6.2.1 rule 2; single
# implementation, lives in jason3_ogdr_processor).
from processors.jason3_ogdr_processor import _time_reference_epoch


# orbit_qual fill/sentinel (the variable's _FillValue): missing quality
# flag, never a measurement (REVIEW_FINDINGS 6.3.5).
SWOT_ORBIT_QUAL_FILL = 127


def _decode_array(dataset: h5py.Dataset) -> np.ndarray:
    """Decode a dataset honoring _FillValue, scale_factor and add_offset."""
    values = np.asarray(dataset[()], dtype=float)
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


def _decode_attr(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return None if value is None else str(value)


def process_swot_por_file(filepath: str, sat_id: str = "swot") -> pd.DataFrame:
    """Convert one SWOT POR/MOE netCDF orbit file into a state-vector DataFrame."""
    with h5py.File(filepath, "r") as handle:
        time = _decode_array(handle["time"])
        reference_epoch = _time_reference_epoch(handle["time"])
        position = _decode_array(handle["position"])
        velocity = _decode_array(handle["velocity"])
        reference_frame = _decode_attr(handle.attrs.get("reference_frame"))

        # orbit_qual: decode fill/sentinel to NaN and COUNT the occurrences
        # (counted flag, rows kept; REVIEW_FINDINGS 6.3.5).  127 is the
        # product's missing-quality sentinel (declared _FillValue on real
        # files); mask it even when the attribute is absent.
        quality = None
        orbit_qual_sentinel_rows = 0
        if "orbit_qual" in handle:
            quality_dataset = handle["orbit_qual"]
            raw_quality = np.asarray(quality_dataset[()])
            orbit_qual_sentinel_rows = int(np.count_nonzero(raw_quality == SWOT_ORBIT_QUAL_FILL))
            quality = _decode_array(quality_dataset)
            quality[quality == SWOT_ORBIT_QUAL_FILL] = np.nan

    epochs = pd.to_datetime(
        [reference_epoch + timedelta(seconds=float(value)) for value in time]
    )
    df = pd.DataFrame(
        {
            "sat_id": sat_id,
            "epoch": epochs,
            "x": position[:, 0],
            "y": position[:, 1],
            "z": position[:, 2],
            "vx": velocity[:, 0],
            "vy": velocity[:, 1],
            "vz": velocity[:, 2],
        }
    )
    if quality is not None:
        df["orbit_qual"] = quality
    # Product-declared reference frame + quality-sentinel ledger (best-effort
    # metadata channel; attrs do not survive all DataFrame operations).
    df.attrs["reference_frame"] = reference_frame
    df.attrs["orbit_qual_sentinel_rows"] = orbit_qual_sentinel_rows
    return df.dropna(subset=["epoch", "x", "y", "z"]).reset_index(drop=True)


def process_swot_por_directory(input_dir: str, sat_id: str = "swot") -> pd.DataFrame:
    """Process all SWOT POR netCDF files in a directory."""
    frames: List[pd.DataFrame] = []
    for path in sorted(Path(input_dir).glob("SWOT_POR_*.nc")):
        try:
            frames.append(process_swot_por_file(str(path), sat_id=sat_id))
        except Exception as exc:
            print(f"Error processing {path}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("epoch").reset_index(drop=True)
