// Minimal SVG charts: no external dependencies so the dashboard works offline.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tooltip = () => document.getElementById("tooltip");

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceTicks(min, max, count) {
    if (!isFinite(min) || !isFinite(max)) return [];
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const step0 = span / Math.max(count, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const ticks = [];
    for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) ticks.push(+t.toPrecision(12));
    return ticks;
  }

  function fmt(v) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return v.toPrecision(2);
  }

  function ema(points, alpha) {
    if (!alpha) return points;
    let s = null;
    return points.map(([x, y]) => {
      s = s === null ? y : alpha * s + (1 - alpha) * y;
      return [x, s];
    });
  }

  // opts: {series:[{name,color,points:[[x,y]],band:[[x,lo,hi]],dashed,scatter,raw}],
  //        hLines:[{y,label,color}], yFormat, xFormat, yMin, yMax, height}
  function lineChart(container, opts) {
    container.innerHTML = "";
    const series = (opts.series || []).map((s) => ({
      ...s, points: (s.points || []).filter(([x, y]) => isFinite(x) && isFinite(y)),
    })).filter((s) => s.points.length || (s.band && s.band.length));
    const hLines = (opts.hLines || []).filter((h) => isFinite(h.y));
    if (!series.length) {
      container.innerHTML = `<div class="empty">${opts.emptyText || "暂无数据"}</div>`;
      return;
    }
    const W = container.clientWidth || 460, H = opts.height || 240;
    const m = { l: 48, r: 12, t: 10, b: 26 };
    const xs = [], ys = [];
    for (const s of series) {
      for (const [x, y] of s.points) { xs.push(x); ys.push(y); }
      for (const [x, lo, hi] of s.band || []) { xs.push(x); if (isFinite(lo)) ys.push(lo); if (isFinite(hi)) ys.push(hi); }
      for (const [x, y] of s.raw || []) { if (isFinite(y)) ys.push(y); }
    }
    for (const h of hLines) ys.push(h.y);
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    let yMin = opts.yMin ?? Math.min(...ys), yMax = opts.yMax ?? Math.max(...ys);
    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    if (yMin === yMax) { yMin -= Math.abs(yMin) * 0.1 || 1; yMax += Math.abs(yMax) * 0.1 || 1; }
    const pad = (yMax - yMin) * 0.06;
    if (opts.yMin === undefined) yMin -= pad;
    if (opts.yMax === undefined) yMax += pad;
    const X = (x) => m.l + ((x - xMin) / (xMax - xMin)) * (W - m.l - m.r);
    const Y = (y) => H - m.b - ((y - yMin) / (yMax - yMin)) * (H - m.t - m.b);
    const yf = opts.yFormat || fmt, xf = opts.xFormat || fmt;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" }, container);
    const axis = el("g", { class: "axis" }, svg);
    for (const t of niceTicks(yMin, yMax, 5)) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid-line" }, axis);
      el("text", { x: m.l - 6, y: Y(t) + 3, "text-anchor": "end" }, axis).textContent = yf(t);
    }
    for (const t of niceTicks(xMin, xMax, Math.max(2, Math.floor(W / 90)))) {
      el("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, axis).textContent = xf(t);
    }
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, axis);

    for (const h of hLines) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(h.y), y2: Y(h.y), stroke: h.color || "#8b98b3",
        "stroke-dasharray": "5 4", "stroke-width": 1.2 }, svg);
    }
    for (const s of series) {
      if (s.band && s.band.length) {
        const b = s.band.filter(([x, lo, hi]) => isFinite(x) && isFinite(lo) && isFinite(hi));
        if (b.length) {
          const d = b.map(([x, , hi], i) => `${i ? "L" : "M"}${X(x)},${Y(hi)}`).join("") +
            b.slice().reverse().map(([x, lo]) => `L${X(x)},${Y(lo)}`).join("") + "Z";
          el("path", { d, fill: s.color, opacity: 0.13 }, svg);
        }
      }
      if (s.raw && s.raw.length) {
        const d = s.raw.filter(([x, y]) => isFinite(x) && isFinite(y)).map(([x, y], i) => `${i ? "L" : "M"}${X(x)},${Y(y)}`).join("");
        el("path", { d, fill: "none", stroke: s.color, "stroke-width": 1, opacity: 0.22 }, svg);
      }
      if (s.scatter) {
        for (const [x, y, c] of s.points) el("circle", { cx: X(x), cy: Y(y), r: 2.4, fill: c || s.color, opacity: 0.75 }, svg);
      } else if (s.points.length) {
        const d = s.points.map(([x, y], i) => `${i ? "L" : "M"}${X(x)},${Y(y)}`).join("");
        el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.8,
          "stroke-dasharray": s.dashed ? "5 4" : "none" }, svg);
        if (s.points.length <= 40) for (const [x, y] of s.points) el("circle", { cx: X(x), cy: Y(y), r: 2.6, fill: s.color }, svg);
      }
    }

    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "#8b98b3", "stroke-width": 1, opacity: 0 }, svg);
    const overlay = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    overlay.addEventListener("mousemove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const px = ((ev.clientX - rect.left) / rect.width) * W;
      const xv = xMin + ((px - m.l) / (W - m.l - m.r)) * (xMax - xMin);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("opacity", 0.5);
      const rows = [];
      for (const s of series) {
        if (s.scatter || !s.points.length) continue;
        let best = s.points[0];
        for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        rows.push(`<div class="t-row"><span><span class="sw" style="background:${s.color}"></span>${s.name}</span><b>${yf(best[1])}</b></div>`);
      }
      for (const h of hLines) rows.push(`<div class="t-row"><span>${h.label}</span><b>${yf(h.y)}</b></div>`);
      const tip = tooltip();
      tip.innerHTML = `<div class="muted">步数 ≈ ${xf(xv)}</div>${rows.join("")}`;
      tip.style.display = "block";
      tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 300) + "px";
      tip.style.top = ev.clientY + 14 + "px";
    });
    overlay.addEventListener("mouseleave", () => { tooltip().style.display = "none"; cross.setAttribute("opacity", 0); });

    const legend = document.createElement("div");
    legend.className = "legend";
    for (const s of series) {
      if (s.hideLegend) continue;
      const span = document.createElement("span");
      span.style.setProperty("--c", s.color);
      if (s.dashed) span.className = "dash";
      span.textContent = s.name;
      legend.appendChild(span);
    }
    for (const h of hLines) {
      const span = document.createElement("span");
      span.className = "dash";
      span.style.setProperty("--c", h.color || "#8b98b3");
      span.textContent = h.label;
      legend.appendChild(span);
    }
    container.appendChild(legend);
  }

  // items: [{label, value, color?}] — horizontal bars around zero.
  function barChart(container, items, opts = {}) {
    container.innerHTML = "";
    items = items.filter((i) => isFinite(i.value));
    if (!items.length) { container.innerHTML = `<div class="empty">${opts.emptyText || "暂无数据"}</div>`; return; }
    const W = container.clientWidth || 460, rowH = 24, m = { l: opts.labelWidth || 230, r: 60, t: 6, b: 6 };
    const H = m.t + m.b + rowH * items.length;
    const vmax = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);
    const allPos = items.every((i) => i.value >= 0), allNeg = items.every((i) => i.value <= 0);
    const lo = allPos ? 0 : -vmax, hi = allNeg ? 0 : vmax;
    const X = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, style: `height:${H}px` }, container);
    el("line", { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: "#26324a" }, svg);
    items.forEach((it, i) => {
      const y = m.t + i * rowH;
      const x0 = X(Math.min(0, it.value)), x1 = X(Math.max(0, it.value));
      el("rect", { x: x0, y: y + 4, width: Math.max(x1 - x0, 1), height: rowH - 8, rx: 3,
        fill: it.color || (it.value >= 0 ? "#3ecf8e" : "#ff6b6b"), opacity: 0.85 }, svg);
      const lab = el("text", { x: m.l - 8, y: y + rowH / 2 + 4, "text-anchor": "end", fill: "#e3e9f5", "font-size": 11 }, svg);
      lab.textContent = it.label;
      const val = el("text", { x: it.value >= 0 ? x1 + 4 : x0 - 4, y: y + rowH / 2 + 4,
        "text-anchor": it.value >= 0 ? "start" : "end", fill: "#8b98b3", "font-size": 11 }, svg);
      val.textContent = (opts.format || fmt)(it.value);
    });
  }

  window.Charts = { lineChart, barChart, fmt, ema };
})();
