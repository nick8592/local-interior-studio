"""Interactive instance selector canvas for Gradio.

Renders a self-contained HTML5 Canvas overlay that lets users hover
and multi-select SAM-segmented instances of a room photo. Designed to
be embedded via ``gr.HTML(value=render_instance_selector_html(...))``.
"""

from __future__ import annotations

import colorsys
import html
import json
import re
from typing import Any

import numpy as np

__all__ = [
    "encode_mask_rle",
    "instance_color_hex",
    "render_instance_selector_html",
    "render_selected_objects_html",
]

_CANVAS_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def encode_mask_rle(mask: np.ndarray) -> list[int]:
    """Run-length encode a binary mask into ``[start, length, ...]`` pairs.

    The mask is flattened in C (row-major) order. Each pair describes a
    contiguous run of True pixels: ``start`` is the index of the first
    True pixel in the flat array, ``length`` is the count of consecutive
    True pixels following it. Runs are emitted in ascending start order.

    Args:
        mask: 2D boolean numpy array of shape ``(H, W)``.

    Returns:
        Flat list ``[start_1, length_1, start_2, length_2, ...]``.
        Returns an empty list if the mask contains no True pixels.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(
            f"encode_mask_rle expects a 2D mask, got shape {arr.shape}"
        )
    flat = arr.astype(bool, copy=False).ravel(order="C")
    rle: list[int] = []
    n = flat.size
    i = 0
    while i < n:
        if not flat[i]:
            i += 1
            continue
        start = i
        i += 1
        while i < n and flat[i]:
            i += 1
        rle.append(int(start))
        rle.append(int(i - start))
    return rle


def _instance_color(idx: int, total: int) -> tuple[int, int, int]:
    """Evenly-spaced bright color for instance ``idx`` out of ``total``."""
    count = max(int(total), 1)
    hue = (idx % count) / count
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.85)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def instance_color_hex(idx: int, total: int) -> str:
    """Return the deterministic hex color string for instance ``idx``.

    Mirrors the color used by ``render_instance_selector_html`` so the
    server-side selected-objects list matches the canvas overlay swatches.
    """
    return _to_hex(_instance_color(idx, total))


def _safe_json(data: Any) -> str:
    """JSON-encode ``data`` with characters escaped for embedding in HTML."""
    return (
        json.dumps(data, separators=(",", ":"), ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_instance_selector_html(
    image_data_url: str,
    instances: list[dict],
    canvas_id: str = "instance-canvas",
) -> str:
    """Build the self-contained HTML for the interactive selector.

    Args:
        image_data_url: Source image as a base64 ``data:`` URL.
        instances: List of instance dicts. Each must contain:
            ``id`` (int), ``mask_rle`` (list[int]), ``bbox``
            (list[float] ``[x, y, w, h]``), ``area`` (int),
            ``score`` (float), ``height`` (int), ``width`` (int).
        canvas_id: Unique DOM id for the canvas (and the
            ``{canvas_id}-selection`` hidden input that carries the
            current selection as JSON).

    Returns:
        HTML string suitable for ``gr.HTML(value=...)``.
    """
    if not isinstance(canvas_id, str) or not _CANVAS_ID_RE.match(canvas_id):
        raise ValueError(
            f"canvas_id must be a valid HTML id (got {canvas_id!r})"
        )

    container_id = f"{canvas_id}-container"
    tooltip_id = f"{canvas_id}-tooltip"
    selection_id = f"{canvas_id}-selection"

    if instances:
        img_w = int(instances[0]["width"])
        img_h = int(instances[0]["height"])
    else:
        img_w = 0
        img_h = 0
    total_pixels = max(img_w * img_h, 1)

    inst_js: list[dict] = []
    total = len(instances)
    for idx, inst in enumerate(instances):
        area = int(inst.get("area", 0))
        r, g, b = _instance_color(idx, total)
        rgb = (r, g, b)
        bbox = [float(v) for v in inst["bbox"]]
        inst_js.append({
            "id": int(inst["id"]),
            "rle": [int(v) for v in inst["mask_rle"]],
            "bbox": bbox,
            "area": area,
            "score": float(inst.get("score", 0.0)),
            "color": "rgb({},{},{})".format(r, g, b),
            "colorHex": _to_hex(rgb),
            "areaPct": (area / total_pixels * 100.0) if total_pixels else 0.0,
        })

    instances_json = _safe_json(inst_js)
    image_json = _safe_json(image_data_url)

    iframe_srcdoc = _HTML_TEMPLATE.format(
        CONTAINER_ID=html.escape(container_id, quote=True),
        TOOLTIP_ID=html.escape(tooltip_id, quote=True),
        SELECTION_ID=html.escape(selection_id, quote=True),
        CANVAS_ID=html.escape(canvas_id, quote=True),
        INSTANCES_JSON=instances_json,
        IMAGE_JSON=image_json,
    )

    return '<iframe srcdoc="{srcdoc}" style="width:100%;border:none;min-height:200px;display:block" onload="this.style.height=this.contentDocument.body.scrollHeight+\'px\'"></iframe>'.format(
        srcdoc=html.escape(iframe_srcdoc, quote=True),
    )


_HTML_TEMPLATE = """<div id="{CONTAINER_ID}" class="isc-root">
<style>
#{CONTAINER_ID} {{
  position: relative;
  display: block;
  width: 100%;
  max-width: 100%;
  margin: 0 auto;
  box-sizing: border-box;
}}
#{CONTAINER_ID} canvas {{
  display: block;
  width: 100% !important;
  height: auto !important;
  max-width: 100%;
  cursor: crosshair;
  border-radius: 6px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.10);
  background: #111;
}}
#{TOOLTIP_ID} {{
  position: fixed;
  display: none;
  background: rgba(20, 20, 25, 0.94);
  color: #fff;
  padding: 6px 10px;
  border-radius: 4px;
  font: 13px/1.3 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  pointer-events: none;
  z-index: 10000;
  white-space: nowrap;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.30);
}}
#{SELECTION_ID} {{
  display: none;
}}
</style>
<canvas id="{CANVAS_ID}"></canvas>
<div id="{TOOLTIP_ID}"></div>
<input id="{SELECTION_ID}" type="text" value="[]" />
<script>
(function () {{
  var canvas = document.getElementById('{CANVAS_ID}');
  var tooltip = document.getElementById('{TOOLTIP_ID}');
  var input = document.getElementById('{SELECTION_ID}');
  if (!canvas || !tooltip || !input) return;
  var ctx = canvas.getContext('2d');
  var INSTANCES = {INSTANCES_JSON};
  var IMG_SRC = {IMAGE_JSON};

  var img = new Image();
  var imgW = 0, imgH = 0;
  var hitTest = null;
  var bitmaps = [];
  var offCanvases = [];
  var outlineCanvases = [];
  var selected = Object.create(null);
  var hoveredIdx = -1;
  var ready = false;

  function decodeRle(rle) {{
    var total = imgW * imgH;
    var out = new Uint8Array(total);
    for (var i = 0; i < rle.length; i += 2) {{
      var start = rle[i];
      var len = rle[i + 1];
      var end = start + len;
      if (end > total) end = total;
      for (var j = start; j < end; j++) out[j] = 1;
    }}
    return out;
  }}

  function buildHitTest() {{
    hitTest = new Int32Array(imgW * imgH);
    hitTest.fill(-1);
    for (var i = 0; i < INSTANCES.length; i++) {{
      var bm = decodeRle(INSTANCES[i].rle);
      bitmaps.push(bm);
      for (var p = 0; p < bm.length; p++) {{
        if (bm[p]) hitTest[p] = i;
      }}
    }}
  }}

  function buildOffscreens() {{
    for (var i = 0; i < INSTANCES.length; i++) {{
      var inst = INSTANCES[i];
      var bm = bitmaps[i];
      var rgb = inst.color.match(/\\d+/g);
      var cr = +rgb[0], cg = +rgb[1], cb = +rgb[2];

      // Fill canvas (for semi-transparent region overlay)
      var off = document.createElement('canvas');
      off.width = imgW;
      off.height = imgH;
      var offCtx = off.getContext('2d');
      var imgData = offCtx.createImageData(imgW, imgH);
      var d = imgData.data;
      for (var p = 0; p < bm.length; p++) {{
        if (bm[p]) {{
          var o = p * 4;
          d[o] = cr;
          d[o + 1] = cg;
          d[o + 2] = cb;
          d[o + 3] = 255;
        }}
      }}
      offCtx.putImageData(imgData, 0, 0);
      offCanvases.push(off);

      // Outline canvas (1px edge of the mask region — panoptic outline)
      var out = document.createElement('canvas');
      out.width = imgW;
      out.height = imgH;
      var outCtx = out.getContext('2d');
      var outData = outCtx.createImageData(imgW, imgH);
      var od = outData.data;
      for (var y = 0; y < imgH; y++) {{
        for (var x = 0; x < imgW; x++) {{
          var idx = y * imgW + x;
          if (!bm[idx]) continue;
          var isEdge = false;
          if (x === 0 || x === imgW - 1 || y === 0 || y === imgH - 1) {{
            isEdge = true;
          }} else {{
            if (!bm[idx - 1] || !bm[idx + 1] || !bm[idx - imgW] || !bm[idx + imgW]) isEdge = true;
          }}
          if (isEdge) {{
            var o = idx * 4;
            od[o] = cr;
            od[o + 1] = cg;
            od[o + 2] = cb;
            od[o + 3] = 255;
          }}
        }}
      }}
      outCtx.putImageData(outData, 0, 0);
      outlineCanvases.push(out);
    }}
  }}

  function hasSelection() {{
    for (var k in selected) {{ if (selected[k]) return true; }}
    return false;
  }}

  function selectedIds() {{
    var ids = [];
    for (var k in selected) {{
      if (selected[k]) ids.push(+k);
    }}
    ids.sort(function (a, b) {{ return a - b; }});
    return ids;
  }}

  function syncInput() {{
    var v = JSON.stringify(selectedIds());
    input.value = v;
    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
    try {{ localStorage.setItem('{CANVAS_ID}-selection', v); }} catch(e) {{}}
    try {{ window.parent.postMessage({{ source: '{CANVAS_ID}', selection: v }}, '*'); }} catch(e) {{}}
  }}

  function render() {{
    if (!ready) return;
    ctx.clearRect(0, 0, imgW, imgH);
    var dim = hasSelection();
    ctx.globalAlpha = dim ? 0.65 : 1.0;
    ctx.drawImage(img, 0, 0);
    ctx.globalAlpha = 1.0;

    for (var i = 0; i < INSTANCES.length; i++) {{
      var inst = INSTANCES[i];
      var isSel = !!selected[inst.id];
      var isHov = hoveredIdx === i;

      var fillA, lw;
      if (isSel) {{
        fillA = isHov ? 0.72 : 0.60;
        lw = isHov ? 4 : 3;
      }} else if (isHov) {{
        fillA = 0.40;
        lw = 4;
      }} else if (dim) {{
        fillA = 0.12;
        lw = 1.5;
      }} else {{
        fillA = 0.20;
        lw = 2;
      }}

      ctx.globalAlpha = fillA;
      ctx.drawImage(offCanvases[i], 0, 0);
      ctx.globalAlpha = 1.0;

      ctx.globalAlpha = isSel ? 1.0 : (isHov ? 0.9 : (dim ? 0.5 : 0.8));
      ctx.lineWidth = 1;
      ctx.drawImage(outlineCanvases[i], 0, 0);
      ctx.globalAlpha = 1.0;
    }}
  }}

  function pointFromEvent(e) {{
    var rect = canvas.getBoundingClientRect();
    var sx = imgW / rect.width;
    var sy = imgH / rect.height;
    var x = Math.floor((e.clientX - rect.left) * sx);
    var y = Math.floor((e.clientY - rect.top) * sy);
    return [x, y];
  }}

  canvas.addEventListener('mousemove', function (e) {{
    if (!ready) return;
    var p = pointFromEvent(e);
    var x = p[0], y = p[1];
    if (x < 0 || x >= imgW || y < 0 || y >= imgH) {{
      if (hoveredIdx !== -1) {{
        hoveredIdx = -1;
        tooltip.style.display = 'none';
        render();
      }}
      return;
    }}
    var idx = hitTest[y * imgW + x];
    if (idx !== hoveredIdx) {{
      hoveredIdx = idx;
      render();
    }}
    if (idx >= 0) {{
      var inst = INSTANCES[idx];
      tooltip.style.display = 'block';
      tooltip.textContent = 'Object ' + (inst.id + 1) + ' — ' +
        inst.areaPct.toFixed(1) + '% of image';
    }} else {{
      tooltip.style.display = 'none';
    }}
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY + 14) + 'px';
  }});

  canvas.addEventListener('mouseleave', function () {{
    if (hoveredIdx !== -1) {{
      hoveredIdx = -1;
      tooltip.style.display = 'none';
      render();
    }}
  }});

  canvas.addEventListener('click', function (e) {{
    if (!ready) return;
    var p = pointFromEvent(e);
    var x = p[0], y = p[1];
    if (x < 0 || x >= imgW || y < 0 || y >= imgH) return;
    var idx = hitTest[y * imgW + x];
    if (idx < 0) return;
    var id = INSTANCES[idx].id;
    if (selected[id]) delete selected[id];
    else selected[id] = 1;
    syncInput();
    render();
  }});

  window.getSelectedIds = function () {{
    return JSON.stringify(selectedIds());
  }};

  img.onload = function () {{
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    if (!imgW || !imgH) return;
    canvas.width = imgW;
    canvas.height = imgH;
    buildHitTest();
    buildOffscreens();
    ready = true;
    render();
  }};
  img.onerror = function () {{
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, canvas.width || 200, canvas.height || 60);
    ctx.fillStyle = '#c44';
    ctx.font = '14px sans-serif';
    ctx.fillText('Failed to load image', 12, 30);
  }};
  img.src = IMG_SRC;
}})();
</script>
</div>
"""


_EMPTY_PLACEHOLDER_HTML = (
    "<div style='padding:10px 12px;color:#888;font-style:italic;"
    "font:13px/1.4 -apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif'>"
    "Click objects on the canvas above to select them.</div>"
)

_NO_OBJECTS_PLACEHOLDER_HTML = (
    "<div style='padding:10px 12px;color:#888;font-style:italic;"
    "font:13px/1.4 -apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif'>"
    "Run Auto-Segment to detect objects.</div>"
)


def _area_pct_for(inst: dict) -> float:
    """Best-effort percentage of image area covered by an instance dict."""
    if "areaPct" in inst and inst["areaPct"] is not None:
        try:
            return float(inst["areaPct"])
        except (TypeError, ValueError):
            pass
    area = int(inst.get("area", 0))
    height = int(inst.get("height", 0))
    width = int(inst.get("width", 0))
    total = max(height * width, 1)
    return area / total * 100.0 if total else 0.0


def render_selected_objects_html(
    instances: list[dict],
    selected_ids: list[int] | None,
) -> str:
    """Build an HTML block listing the currently selected instances.

    Args:
        instances: List of instance dicts. Each must contain ``id`` (int) and
            ``colorHex`` (str). ``areaPct`` (float) is preferred; otherwise
            ``area`` + ``height`` * ``width`` will be used to compute the
            percentage.
        selected_ids: List of instance ids currently selected on the canvas.
            Unknown ids are ignored. An empty list renders an empty-state
            placeholder.

    Returns:
        An HTML fragment suitable for ``gr.HTML(value=...)``. Uses inline
        styles only — Gradio's HTML renderer strips ``<style>`` blocks in
        values that share a sandbox with the surrounding page.
    """
    if not instances:
        return _NO_OBJECTS_PLACEHOLDER_HTML

    selected_set = {int(i) for i in (selected_ids or [])}
    selected = [inst for inst in instances if int(inst.get("id", -1)) in selected_set]
    selected.sort(key=lambda inst: int(inst["id"]))

    if not selected:
        return _EMPTY_PLACEHOLDER_HTML

    header = (
        f"<div style='padding:4px 8px 6px 8px;color:#bbb;"
        f"font:12px/1.3 -apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif'>"
        f"{len(selected)} object{'s' if len(selected) != 1 else ''} selected</div>"
    )

    rows: list[str] = []
    for inst in selected:
        iid = int(inst["id"])
        color = html.escape(str(inst.get("colorHex", "#888888")), quote=True)
        area_pct = _area_pct_for(inst)
        label = f"Object {iid + 1}"
        rows.append(
            "<div style='display:flex;align-items:center;gap:10px;"
            "padding:7px 10px;margin:3px 0;border-radius:5px;"
            "background:#1f1f23;color:#eee;"
            "font:13px/1.3 -apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif'>"
            f"<span style='display:inline-block;width:18px;height:18px;"
            f"border-radius:4px;background:{color};"
            f"border:1px solid rgba(255,255,255,0.25);flex-shrink:0'></span>"
            f"<span style='flex:1'>{label}</span>"
            f"<span style='color:#aaa;font-variant-numeric:tabular-nums'>{area_pct:.1f}%</span>"
            "</div>"
        )

    return (
        "<div style='margin:2px 0'>"
        + header
        + "".join(rows)
        + "</div>"
    )
