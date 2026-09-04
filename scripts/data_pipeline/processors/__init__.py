"""
Processors module for LEO orbit data

This module provides parsers for various orbit data formats:
- POD (Precision Orbit Determination): SP3, EOF formats
- SLR (Satellite Laser Ranging): CRD, NPT formats
- Orbital-mechanics utilities: RTN frame rotation, Keplerian elements

Frame transforms (ITRF/TEME/GCRS) were removed from
``coordinate_transform`` (WP8 / REVIEW_FINDINGS 6.2.2: single
implementation via astropy coordinates; see that module's docstring).
"""

from .pod_processor import (
    sp3_to_dataframe,
    eof_to_dataframe,
    process_pod_file,
    process_pod_directory,
    SP3Header,
    EOFMetadata,
)

from .slr_processor import (
    slr_to_dataframe,
    process_slr_file,
    calculate_residual,
    merge_slr_with_orbit,
    CRDHeader,
    NormalPoint,
)

from .coordinate_transform import (
    eci_to_rtn,
    rtn_to_eci,
    compute_orbital_elements,
    state_vector_from_elements,
)

__all__ = [
    # POD processing
    'sp3_to_dataframe',
    'eof_to_dataframe',
    'process_pod_file',
    'process_pod_directory',
    'SP3Header',
    'EOFMetadata',
    # SLR processing
    'slr_to_dataframe',
    'process_slr_file',
    'calculate_residual',
    'merge_slr_with_orbit',
    'CRDHeader',
    'NormalPoint',
    # Orbital-mechanics utilities
    'eci_to_rtn',
    'rtn_to_eci',
    'compute_orbital_elements',
    'state_vector_from_elements',
]
