# app/utils/photo_metadata.py
from __future__ import annotations

import os
import pathlib
from io import BytesIO
from typing import Any, Dict, Optional, Union
from tempfile import NamedTemporaryFile

import requests
from PIL import Image, ExifTags
from exiftool import ExifToolHelper

# Optional HEIC/HEIF support
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ───────────────────────── helpers ──────────────────────────
def _download(url: str) -> bytes:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content

def _convert_gps_str(raw: str | None) -> Optional[float]:
    """Parse '37 deg 46' 29.99" N' → 37.774999."""
    if not raw:
        return None
    import re
    m = re.match(r"(\d+)\D+(\d+)\D+([\d.]+)\D+([NSEW])", raw)
    if not m:
        return None
    deg, minute, sec, ref = m.groups()
    dec = float(deg) + float(minute) / 60 + float(sec) / 3600
    return -dec if ref in "SW" else dec

# ───────────────────────── public API ──────────────────────────
def extract_photo_metadata(path_or_url: str) -> Dict[str, Optional[Union[str, float]]]:
    """
    Extract EXIF metadata via ExifTool.  Called inside `asyncio.to_thread()`
    so the blocking I/O is safe within the async code-base.
    """
    is_remote = path_or_url.startswith(("http://", "https://"))
    tmp_path: str | os.PathLike

    if is_remote:
        data = _download(path_or_url)
        suffix = pathlib.Path(path_or_url).suffix or ".img"
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
    else:
        tmp_path = path_or_url

    try:
        with ExifToolHelper() as et:
            meta_list = et.get_metadata(str(tmp_path))
        meta = meta_list[0] if meta_list else {}
    finally:
        if is_remote:
            os.unlink(tmp_path)  # always clean up

    lat = _convert_gps_str(meta.get("EXIF:GPSLatitude"))
    lon = _convert_gps_str(meta.get("EXIF:GPSLongitude"))

    return {
        "datetime_original": meta.get("EXIF:DateTimeOriginal"),
        "gps_latitude": lat,
        "gps_longitude": lon,
        "camera_make": meta.get("EXIF:Make"),
        "camera_model": meta.get("EXIF:Model"),
    }