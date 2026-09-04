"""
Coordinate Transformation Module (orbital-mechanics utilities).

Retained utilities (consumed by ``processors.physical_qc``):

1. ECI -> RTN (Radial-Transverse-Normal) frame rotation and its inverse
2. Keplerian orbital elements <-> Cartesian state vector

Frame conversions (ITRF <-> TEME <-> GCRS) were REMOVED (WP8,
REVIEW_FINDINGS 6.2.2 rule 2: coordinate/velocity transforms have exactly
one implementation -- astropy frame differentials).  The removed wrappers
(``itrf_to_teme``/``teme_to_eci``/``itrf_to_gcrs``/``transform_dataframe``)
had no callers and duplicated the astropy path already used by
``analyzers.state_estimation.teme_to_itrs``; the unused ``OMEGA_EARTH``
constant was cleared with them.  For any frame transform, use
``astropy.coordinates`` frames directly, e.g.::

    from astropy.coordinates import ITRS, TEME, CartesianRepresentation

For the first-order ECEF-velocity correction
``v_inertial = v_ecef + omega x r`` use
``processors.physical_qc.ecef_velocity_to_inertial`` (the single shared
implementation; ``EARTH_ROTATION_RAD_PER_S`` is defined once there).
"""

from __future__ import annotations

import numpy as np


def eci_to_rtn(r_eci: np.ndarray, v_eci: np.ndarray,
               delta_v_eci: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """
    Transform ECI vectors to RTN (Radial-Transverse-Normal) frame

    RTN frame definition:
    - R (Radial): Along position vector (outward from Earth center)
    - T (Transverse/Along-track): In orbital plane, perpendicular to R
    - N (Normal/Cross-track): Perpendicular to orbital plane (r × v direction)

    Args:
        r_eci: Position vector [x, y, z] in ECI (meters)
        v_eci: Velocity vector [vx, vy, vz] in ECI (m/s)
        delta_v_eci: Optional velocity change vector in ECI (m/s)

    Returns:
        Tuple of (rotation_matrix, r_rtn, v_rtn, [delta_v_rtn if provided])
    """
    r_eci = np.array(r_eci)
    v_eci = np.array(v_eci)

    # Unit vectors for RTN frame
    r_hat = r_eci / np.linalg.norm(r_eci)  # Radial
    h = np.cross(r_eci, v_eci)  # Angular momentum
    n_hat = h / np.linalg.norm(h)  # Normal
    t_hat = np.cross(n_hat, r_hat)  # Transverse

    # Rotation matrix from ECI to RTN
    R_eci_to_rtn = np.array([r_hat, t_hat, n_hat])

    # Transform position and velocity to RTN
    r_rtn = R_eci_to_rtn @ r_eci
    v_rtn = R_eci_to_rtn @ v_eci

    if delta_v_eci is not None:
        delta_v_eci = np.array(delta_v_eci)
        delta_v_rtn = R_eci_to_rtn @ delta_v_eci
        return R_eci_to_rtn, r_rtn, v_rtn, delta_v_rtn

    return R_eci_to_rtn, r_rtn, v_rtn


def rtn_to_eci(r_eci: np.ndarray, v_eci: np.ndarray,
               delta_v_rtn: np.ndarray) -> np.ndarray:
    """
    Transform RTN vector back to ECI frame

    Args:
        r_eci: Reference position in ECI (meters)
        v_eci: Reference velocity in ECI (m/s)
        delta_v_rtn: Vector in RTN frame to transform (m/s)

    Returns:
        Vector in ECI frame
    """
    R_eci_to_rtn, _, _ = eci_to_rtn(r_eci, v_eci)
    # Inverse of rotation matrix is its transpose
    R_rtn_to_eci = R_eci_to_rtn.T
    return R_rtn_to_eci @ delta_v_rtn


def compute_orbital_elements(r: np.ndarray, v: np.ndarray,
                             mu: float = 3.986004418e14) -> dict:
    """
    Compute Keplerian orbital elements from state vector

    Args:
        r: Position vector [x, y, z] in meters (ECI frame)
        v: Velocity vector [vx, vy, vz] in m/s (ECI frame)
        mu: Gravitational parameter (default: Earth, m³/s²)

    Returns:
        Dictionary with orbital elements:
        - a: Semi-major axis (m)
        - e: Eccentricity
        - i: Inclination (rad)
        - omega: Argument of perigee (rad)
        - Omega: RAAN (rad)
        - nu: True anomaly (rad)
        - M: Mean anomaly (rad)
        - n: Mean motion (rad/s)
        - period: Orbital period (s)
    """
    r = np.array(r)
    v = np.array(v)

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Specific angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Node vector
    k = np.array([0, 0, 1])
    n_vec = np.cross(k, h)
    n_mag = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # Specific orbital energy
    energy = v_mag**2 / 2 - mu / r_mag

    # Semi-major axis
    if abs(e - 1.0) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = float('inf')  # Parabolic

    # Inclination
    i = np.arccos(np.clip(h[2] / h_mag, -1, 1))

    # RAAN (Right Ascension of Ascending Node)
    if n_mag > 1e-10:
        Omega = np.arccos(np.clip(n_vec[0] / n_mag, -1, 1))
        if n_vec[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0

    # Argument of perigee
    if n_mag > 1e-10 and e > 1e-10:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0

    # True anomaly
    if e > 1e-10:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0

    # Mean anomaly (for elliptical orbits)
    if e < 1.0 and a > 0:
        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
        M = E - e * np.sin(E)
        M = M % (2 * np.pi)
    else:
        M = 0.0

    # Mean motion and period
    if a > 0:
        n = np.sqrt(mu / a**3)
        period = 2 * np.pi / n
    else:
        n = 0.0
        period = float('inf')

    return {
        'a': a,
        'e': e,
        'i': i,
        'omega': omega,
        'Omega': Omega,
        'nu': nu,
        'M': M,
        'n': n,
        'period': period,
        'h_mag': h_mag,
        'energy': energy
    }


def state_vector_from_elements(a: float, e: float, i: float,
                               omega: float, Omega: float, nu: float,
                               mu: float = 3.986004418e14) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute state vector from Keplerian orbital elements

    Args:
        a: Semi-major axis (m)
        e: Eccentricity
        i: Inclination (rad)
        omega: Argument of perigee (rad)
        Omega: RAAN (rad)
        nu: True anomaly (rad)
        mu: Gravitational parameter (default: Earth, m³/s²)

    Returns:
        Tuple of (r, v) position and velocity vectors in ECI frame
    """
    # Semi-latus rectum
    p = a * (1 - e**2)

    # Position and velocity in perifocal frame
    r_pqw = p / (1 + e * np.cos(nu))
    r_perifocal = np.array([
        r_pqw * np.cos(nu),
        r_pqw * np.sin(nu),
        0
    ])

    v_perifocal = np.sqrt(mu / p) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0
    ])

    # Rotation matrices
    R3_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])

    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])

    R3_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])

    # Combined rotation: perifocal to ECI
    R = R3_Omega @ R1_i @ R3_omega

    r_eci = R @ r_perifocal
    v_eci = R @ v_perifocal

    return r_eci, v_eci
