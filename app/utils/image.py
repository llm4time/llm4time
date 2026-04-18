import io
from pathlib import Path
from typing import Union

import requests
from PIL import Image


def load_image(
    src: Union[str, bytes, None],
    size: int | None = None,
    as_bytes: bool = False,
    fallback_path: str | None = None,
) -> Image.Image | bytes | None:
  """
  Load an image (bytes, URL, or local path), crop to a centered square,
  resize with LANCZOS to `size` × `size`, and return as PIL.Image or PNG bytes.
  """

  blob: bytes | None = None

  # 1) Load bytes
  if isinstance(src, bytes):
    blob = src

  elif isinstance(src, str):
    try:
      if src.startswith(("http://", "https://")):
        resp = requests.get(src, timeout=10, stream=True)
        resp.raise_for_status()
        blob = resp.content
      else:
        p = Path(src)
        if p.is_file():
          blob = p.read_bytes()
    except Exception:
      blob = None

  # 2) Fallback
  if not blob and fallback_path:
    try:
      p = Path(fallback_path)
      blob = p.read_bytes() if p.is_file() else None
    except Exception:
      return None

  if not blob:
    return None

  # 3) Decode + verify image
  try:
    bio = io.BytesIO(blob)
    img = Image.open(bio)
    img.verify()

    bio.seek(0)
    img = Image.open(bio)

    if img.mode not in ("RGB", "RGBA"):
      img = img.convert("RGBA")
  except Exception:
    return None

  # 4) Crop to centered square
  w, h = img.size
  min_dim = min(w, h)
  left = (w - min_dim) // 2
  top = (h - min_dim) // 2
  img = img.crop((left, top, left + min_dim, top + min_dim))

  # 5) Resize
  if size is None:
    size = min_dim
  img = img.resize((size, size), Image.Resampling.LANCZOS)

  # 6) Return bytes or Image
  if as_bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()

  return img
