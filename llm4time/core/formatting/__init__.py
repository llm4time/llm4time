from ._formats import *
from ._parsers import *
from ._encoders import *
from ._decoders import *

from ..data import TimeSeries, TSFormat


def from_str(string: str, format: TSFormat) -> TimeSeries:
  formats_map = {
      TSFormat.ARRAY: from_array,
      TSFormat.CONTEXT: from_context,
      TSFormat.CSV: from_csv,
      TSFormat.CUSTOM: from_custom,
      TSFormat.JSON: from_json,
      TSFormat.MARKDOWN: from_markdown,
      TSFormat.PLAIN: from_plain,
      TSFormat.SYMBOL: from_symbol,
      TSFormat.TSV: from_tsv,
  }
  if format not in formats_map:
    raise ValueError(f"Unknown format: {format}.")
  return decode_textual(formats_map[format](string))
