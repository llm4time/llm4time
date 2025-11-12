import plotly.express as px
import colorsys
import math


def adjust_lightness(hex_color: str, lightness: float) -> str:
  hex_color = hex_color.lstrip("#")
  r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
  h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
  l = min(max(lightness, 0), 1)
  r, g, b = colorsys.hls_to_rgb(h, l, s)
  return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def get_lightness_map(n: int, lightness: float) -> list[float]:
  if n <= 0:
    return []
  elif n == 1:
    return [lightness]
  step = (lightness - 0.3) / (n - 1)
  values = [lightness - i * step for i in range(n)]
  return [min(1.0, math.ceil(v * 100) / 100) for v in values]


def get_color(i: int, lightness: float = 0.7) -> str:
  color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
  return adjust_lightness(color, lightness)
