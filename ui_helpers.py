# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fundación CENER
"""Pure helpers shared by the Streamlit interface and its tests."""

from hashlib import sha256
from math import isfinite
from pathlib import Path
import re
from urllib.parse import urlsplit


def format_value(value: float | None, decimals: int = 2, suffix: str = "") -> str:
    """Format an optional finite metric without misleading zero values."""
    if value is None or not isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def payback_display(value: float | None) -> str:
    """Distinguish an unreached payback from a numerical result."""
    if value is None or not isfinite(value):
        return "Not reached"
    return format_value(value, 1, " years")


def file_fingerprint(*paths: Path) -> str:
    """Invalidate cached results when the source or input files change."""
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def valid_repository_url(url: str) -> bool:
    """Accept an explicitly configured HTTPS GitHub repository URL only."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return (parts.scheme == "https" and parts.netloc.lower() == "github.com"
            and re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", parts.path) is not None
            and not parts.query and not parts.fragment)
