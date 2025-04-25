# app/utils/photo_metadata.py
from __future__ import annotations

import pathlib
from io import BytesIO
from typing import Any, Dict, Optional, Union

import requests
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
from exiftool import ExifToolHelper
import os
from tempfile import NamedTemporaryFile

# ─────────────────────────── HEIC/HEIF support ─────────────────────────────
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except ImportError:
    # Library not installed → JPEG/PNG still work; silently ignore
    pass


# ────────────────────────────── Helpers ────────────────────────────────────
def _load_image(path_or_url: str) -> Image.Image:
    """Open a Pillow Image from a local path **or** a remote URL."""
    if path_or_url.startswith(("http://", "https://")):
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    return Image.open(pathlib.Path(path_or_url).expanduser())


def _exif_dict(img: Image.Image) -> dict[int, Any]:
    """Return raw EXIF tags as a plain dict, regardless of Pillow version."""
    try:
        return dict(img.getexif())  # Pillow ≥ 7
    except AttributeError:
        # Older Pillow
        return dict(getattr(img, "_getexif", lambda: {})() or {})


def _convert_gps(coord, ref: Optional[Union[str, bytes]]) -> Optional[float]:
    """Convert EXIF GPS tuples into signed decimal degrees."""
    if not coord or not ref:
        return None

    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")

    def _val(rat) -> float:
        try:
            return rat.num / rat.den  # IFDRational
        except AttributeError:
            num, den = rat if isinstance(rat, tuple) else (rat, 1)
            return num / den

    deg, minute, sec = map(_val, coord)
    dec = deg + minute / 60 + sec / 3600
    return -dec if ref in {"S", "W"} else dec


# ─────────────────────────── Public API ─────────────────────────────────────
def extract_photo_metadata(path_or_url: str) -> Dict[str, Optional[Union[str, float]]]:
    """
    Extract EXIF metadata using ExifTool (robust for JPEG, HEIC, RAW, video, etc).
    Returns a dict with keys: datetime_original, gps_latitude, gps_longitude, camera_make, camera_model.
    """
    # Download if URL
    if path_or_url.startswith(("http://", "https://")):
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        with NamedTemporaryFile(delete=False, suffix=Path(path_or_url).suffix) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
    else:
        tmp_path = path_or_url

    with ExifToolHelper() as et:
        meta_list = et.get_metadata(str(tmp_path))
        meta = meta_list[0] if meta_list else {}

    # Clean up temp file if we downloaded
    if path_or_url.startswith(("http://", "https://")):
        os.unlink(tmp_path)

    return {
        "datetime_original": meta.get("EXIF:DateTimeOriginal"),
        "gps_latitude": meta.get("EXIF:GPSLatitude"),
        "gps_longitude": meta.get("EXIF:GPSLongitude"),
        "camera_make": meta.get("EXIF:Make"),
        "camera_model": meta.get("EXIF:Model"),
    }