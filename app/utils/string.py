import pandas as pd
import unicodedata
import re


def normalize(string: str) -> str:
  """Normalize a string so that it is compatible with URLs and filenames."""
  string = unicodedata.normalize('NFKD', string)
  string = string.encode('ASCII', 'ignore').decode('utf-8')
  string = re.sub(r'[^a-zA-Z0-9]', '_', string)
  string = re.sub(r'_+', '_', string)
  return string.strip('_').lower()


def freq_to_description(freq: str | None) -> str:
  """Convert frequency code to human-readable description."""
  if not freq:
    return "N/A"
  freq_options = {
      "ms": "Milliseconds",
      "s": "Seconds",
      "min": "Minutes",
      "h": "Hours",
      "D": "Days",
      "M": "Months",
      "Y": "Years"
  }
  return freq_options.get(freq, freq)
