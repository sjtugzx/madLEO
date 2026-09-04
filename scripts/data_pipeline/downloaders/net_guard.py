"""Network egress guard for the MAD-LEO downloader modules.

Every outbound request issued from ``scripts/data_pipeline/downloaders/`` must
flow through this module:

- HTTP(S): call :func:`guarded_request`, which validates the URL and then
  dispatches through a ``requests.Session().request(...)`` call.
- curl subprocesses: guard the URL at the call site with
  :func:`assert_allowed_url` (or scan the full argument list with
  :func:`assert_allowed_url_args`).
- every local path built from a remote-supplied filename must pass
  :func:`sanitize_remote_filename`; download targets should additionally be
  checked with :func:`assert_path_within`.

Validation policy:

- Scheme: ``https`` and ``ftps`` are allowed by default. Plaintext ``http``/
  ``ftp`` fail closed unless ``MADLEO_ALLOW_INSECURE=1`` is set in the
  environment (a warning is printed when the escape hatch is used); all other
  schemes (``file``, ``gopher``, ...) are always rejected.
- Host allowlist: the URL host must equal an entry in
  ``ALLOWED_HOST_SUFFIXES`` or be a subdomain of one (matching on domain
  boundaries only, so ``evil-cddis.nasa.gov.attacker.com`` is rejected even
  though it contains ``cddis.nasa.gov``).
- Resolved-IP boundary: after the hostname gate, the host is resolved via
  ``socket.getaddrinfo`` and every returned address must be global-scope
  public internet space. Loopback, private, link-local (including the cloud
  metadata address 169.254.169.254), unique-local IPv6, multicast, reserved,
  and unspecified addresses are rejected, as is any DNS resolution failure.
  This blocks DNS-rebinding/misresolved-allowlisted-host SSRF.

Example::

    from downloaders.net_guard import guarded_request, sanitize_remote_filename

    filename = sanitize_remote_filename(file_info["filename"])
    response = guarded_request("GET", file_info["url"], timeout=60)
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import requests

# Opt-in escape hatch for historical plaintext HTTP endpoints. Plaintext FTP
# is no longer implemented in Python (EDC now speaks explicit FTPS); plaintext
# transports may still be reached through guarded curl subprocesses with this
# flag set.
INSECURE_ENV_FLAG = "MADLEO_ALLOW_INSECURE"

# Hosts (exact or domain-boundary suffixes) the downloaders may contact.
# Collected from every URL constant in scripts/data_pipeline/downloaders/
# plus the provider manifests under configs/. Never add public suffixes
# (".com", "nasa.gov", ...) here.
ALLOWED_HOST_SUFFIXES = frozenset(
    {
        # IDS/DORIS maneuver histories (download_ids_maneuver_histories.py)
        "ids-doris.org",
        # CDDIS SLR/DORIS archives and Earthdata CMR/SSO redirects
        # (ilrs_downloader.py, podaac_downloader.py,
        # download_reference_orbit_samples.py, download_reference_orbit_bulk.py)
        "cddis.nasa.gov",
        "gdc.cddis.eosdis.nasa.gov",
        "cmr.earthdata.nasa.gov",
        "urs.earthdata.nasa.gov",
        # GFZ ISDC GNSS orbits (gfz_downloader.py) and historical GFZ hosts
        "isdc-data.gfz.de",
        "isdc.gfz-potsdam.de",
        "laser.gfz-potsdam.de",
        # Copernicus Data Space Ecosystem (cdse_downloader.py)
        "identity.dataspace.copernicus.eu",
        "catalogue.dataspace.copernicus.eu",
        "download.dataspace.copernicus.eu",
        # Sentinel auxiliary orbits (sentinel_downloader.py)
        "s1qc.asf.alaska.edu",
        "s1-orbits.s3.us-west-2.amazonaws.com",
        # PO.DAAC / Earthdata cloud granule archives (podaac_downloader.py,
        # download_reference_orbit_samples.py). The suffix entry also covers
        # archive.swot.podaac.earthdata.nasa.gov.
        "podaac.earthdata.nasa.gov",
        "podaac.jpl.nasa.gov",
        "podaac-tools.jpl.nasa.gov",
        "podaac-opendap.jpl.nasa.gov",
        # TLE sources used by the acquisition clients (benchmarking.spacetrack
        # and the celestrak fallbacks)
        "space-track.org",
        "celestrak.org",
        # EDC mirror, now reached via explicit FTPS (ilrs_downloader.py)
        "edc.dgfi.tum.de",
    }
)

_SECURE_SCHEMES = frozenset({"https", "ftps"})
_INSECURE_SCHEMES = frozenset({"http", "ftp"})
_URL_PREFIX_RE = re.compile(r"^(https?|ftps?)://", re.IGNORECASE)


def _insecure_transport_allowed() -> bool:
    """Return True when the caller explicitly opted into plaintext transports."""
    return os.environ.get(INSECURE_ENV_FLAG, "").strip() == "1"


def _host_is_allowed(host: str) -> bool:
    """Match host against the allowlist on exact or domain-boundary suffix."""
    normalized = host.lower().rstrip(".")
    for entry in ALLOWED_HOST_SUFFIXES:
        if normalized == entry or normalized.endswith("." + entry):
            return True
    return False


def _resolve_host_addresses(host: str) -> list[str]:
    """Resolve a hostname to its unique address strings (no network policy here)."""
    infos = socket.getaddrinfo(host, None)
    addresses: list[str] = []
    for info in infos:
        raw_address = info[4][0]
        address = raw_address.split("%", 1)[0]  # drop IPv6 scope ids
        if address not in addresses:
            addresses.append(address)
    return addresses


def _assert_global_addresses(host: str) -> None:
    """Reject the host unless every resolved address is global-scope public space."""
    try:
        addresses = _resolve_host_addresses(host)
    except OSError as exc:
        raise ValueError(f"could not resolve allowlisted host {host!r}: {exc}") from exc
    if not addresses:
        raise ValueError(f"allowlisted host {host!r} resolved to no addresses")
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError(
                f"allowlisted host {host!r} resolved to a non-IP address {address!r}"
            ) from exc
        if (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_reserved
            or parsed.is_unspecified
        ):
            raise ValueError(
                f"allowlisted host {host!r} resolves to {parsed}, which is not "
                f"global-scope public internet space (blocked as SSRF)"
            )


def assert_allowed_url(url: str) -> str:
    """Validate one outbound URL against scheme, host allowlist, and resolved IPs.

    Gate order: scheme policy, hostname allowlist, plaintext escape-hatch
    check, then resolved-IP boundary validation. Returns the (stripped) URL on
    success so call sites can write ``url = assert_allowed_url(url)``. Raises
    ``ValueError`` on any policy violation.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")
    candidate = url.strip()
    parts = urlsplit(candidate)
    scheme = (parts.scheme or "").lower()
    if scheme not in _SECURE_SCHEMES and scheme not in _INSECURE_SCHEMES:
        raise ValueError(
            f"blocked URL scheme {scheme!r} for {candidate!r}: only https/ftps "
            f"are allowed (http/ftp require {INSECURE_ENV_FLAG}=1)"
        )
    host = parts.hostname
    if not host:
        raise ValueError(f"URL has no host: {candidate!r}")
    if not _host_is_allowed(host):
        raise ValueError(
            f"host {host!r} is not in the MAD-LEO downloader allowlist "
            f"({candidate!r})"
        )
    if scheme in _INSECURE_SCHEMES and not _insecure_transport_allowed():
        raise ValueError(
            f"insecure scheme {scheme!r} blocked by default for {candidate!r}: "
            f"set {INSECURE_ENV_FLAG}=1 to allow plaintext endpoints"
        )
    _assert_global_addresses(host)
    if scheme in _INSECURE_SCHEMES:
        print(
            f"WARNING: {INSECURE_ENV_FLAG}=1 permits plaintext {scheme} transfer "
            f"to allowlisted host {host!r}",
            file=sys.stderr,
        )
    return candidate


def guarded_request(method: str, url: str, session: requests.Session | None = None, **kwargs):
    """Validate a URL and dispatch the HTTP request through a Session.

    This is the single sanctioned HTTP entry point for the downloaders: the
    URL is run through :func:`assert_allowed_url` first, then the request is
    sent via ``requests.Session().request(method=..., url=..., **kwargs)``.
    Pass ``session=`` to reuse a caller-configured session (e.g. with retry
    adapters); otherwise a fresh session is created and closed after the
    call (streamed responses remain consumable afterwards).
    """
    validated = assert_allowed_url(url)
    active = session if session is not None else requests.Session()
    own_session = session is None
    try:
        return active.request(method=method, url=validated, **kwargs)
    finally:
        if own_session:
            active.close()


def assert_allowed_url_args(args: Iterable[str]) -> None:
    """Guard every URL-looking argument handed to a subprocess (e.g. curl)."""
    for arg in args:
        if isinstance(arg, str) and _URL_PREFIX_RE.match(arg.strip()):
            assert_allowed_url(arg.strip())


def assert_path_within(path: str | Path, root: str | Path) -> Path:
    """Require a local write target to resolve inside the given root directory.

    Both paths are resolved with ``os.path.realpath`` first, so ``..``
    segments and symlink escapes out of the root are rejected. Returns the
    resolved target path.
    """
    resolved = Path(os.path.realpath(str(path)))
    resolved_root = Path(os.path.realpath(str(root)))
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(
            f"local write target {str(path)!r} resolves to {str(resolved)!r}, "
            f"which is outside the allowed root {str(resolved_root)!r}"
        )
    return resolved


def sanitize_remote_filename(name: str) -> str:
    """Return a safe local basename for a remote-supplied filename.

    Path segments are stripped to the final basename (``a/b/c.nc`` becomes
    ``c.nc``); absolute paths, backslashes, null bytes, and empty/dot-only
    basenames are rejected with ``ValueError``.
    """
    if not isinstance(name, str):
        raise ValueError(f"remote filename must be a string, got {type(name)!r}")
    candidate = name.strip()
    if not candidate:
        raise ValueError("remote filename is empty")
    if "\x00" in candidate:
        raise ValueError(f"remote filename contains a null byte: {name!r}")
    if "\\" in candidate:
        raise ValueError(f"remote filename contains a backslash: {name!r}")
    if candidate.startswith("/"):
        raise ValueError(f"remote filename is an absolute path: {name!r}")
    basename = candidate.split("/")[-1]
    if basename in {"", ".", ".."}:
        raise ValueError(f"remote filename reduces to an unsafe basename: {name!r}")
    return basename
