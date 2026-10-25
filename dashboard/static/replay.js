// Canvas replay of one recorded evaluation episode (competitor/replay.py).
// World frame: x = east km, y = north km around the sector centre.
(function () {
  const C = {
    bg: "#0b1019", sector: "#3ecf8e", obstacle: "#3a4660", obstacleEdge: "#5b6b8c",
    route: "rgba(139,152,179,0.25)", goal: "#3ecf8e", agent: "#5b9dff", ghost: "rgba(200,208,224,0.55)",
    intruder: "#e3e9f5", conflict: "#ff6b6b", restricted: "#b38cff", outside: "#f5b84a", text: "#e3e9f5", muted: "#8b98b3",
  };
  const BASE_RATE = 60; // simulated seconds per real second at 1×

  function lerp(a, b, f) { return a + (b - a) * f; }
  function lerpAngle(a, b, f) { const d = ((b - a + 540) % 360) - 180; return a + d * f; }

  function locate(frames, t) {
    let lo = 0, hi = frames.length - 1;
    if (t <= frames[0].t) return [0, 0];
    if (t >= frames[hi].t) return [hi, 0];
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (frames[mid].t <= t) lo = mid; else hi = mid; }
    const span = frames[hi].t - frames[lo].t;
    return [lo, span > 0 ? (t - frames[lo].t) / span : 0];
  }

  function interpAgent(frames, t) {
    const [k, f] = locate(frames, t);
    const a = frames[k].agent, b = (frames[k + 1] || frames[k]).agent;
    return { k, x: lerp(a[0], b[0], f), y: lerp(a[1], b[1], f), hdg: lerpAngle(a[2], b[2], f), spd: lerp(a[3], b[3], f) };
  }

  function statusColor(fr) {
    if (fr.conflict_with && fr.conflict_with.length) return C.conflict;
    if (fr.in_restricted) return C.restricted;
    if (fr.outside) return C.outside;
    return null;
  }

  function mount(root, ep, opts = {}) {
    const frames = ep.frames, ghost = ep.neutral ? ep.neutral.frames : null, st = ep.static;
    const tEnd = Math.max(frames[frames.length - 1].t, ghost ? ghost[ghost.length - 1].t : 0);
    const cum = []; frames.reduce((acc, fr, i) => (cum[i] = acc + (fr.reward || 0)), 0);
    const routeById = st.routes || {};
    const state = { t: 0, playing: false, speed: 1, last: null, raf: null,
      show: { ghost: !!ghost, routes: true, circles: true, trail: true } };

    root.innerHTML = `
      <div class="replay-wrap">
        <div class="replay-stage">
          <canvas class="replay-canvas"></canvas>
          <div class="replay-controls">
            <button class="btn" data-act="start" title="回到开头">⏮</button>
            <button class="btn" data-act="back" title="后退一步">◀</button>
            <button class="btn primary" data-act="play" style="min-width:64px">▶ 播放</button>
            <button class="btn" data-act="fwd" title="前进一步">▶</button>
            <button class="btn" data-act="end" title="跳到结尾">⏭</button>
            <select class="btn" data-act="speed">
              ${[0.25, 0.5, 1, 2, 4, 8].map((s) => `<option value="${s}" ${s === 1 ? "selected" : ""}>${s}×</option>`).join("")}
            </select>
            <input type="range" class="replay-slider" min="0" max="${tEnd}" step="1" value="0">
            <span class="mono replay-clock">0 s</span>
          </div>
          <canvas class="replay-timeline" height="${ghost ? 54 : 30}"></canvas>
          <div class="legend small">
            <span style="--c:${C.agent}">训练模型</span>${ghost ? `<span class="dash" style="--c:#c8d0e0">中性 [0,0] 幽灵轨迹</span>` : ""}
            <span style="--c:${C.conflict}">冲突 (&lt;5 NM)</span><span style="--c:${C.restricted}">限制区内</span><span style="--c:${C.outside}">扇区外</span>
          </div>
        </div>
        <div class="replay-side">
          <div class="panel"><h3>当前状态</h3><div class="replay-hud"></div></div>
          <div class="panel"><h3>显示</h3>
            ${[["ghost", "中性幽灵轨迹", !ghost], ["routes", "入侵机航线"], ["circles", "5 NM 间隔圈"], ["trail", "轨迹"]].map(([k, l, dis]) =>
              `<label class="toggle"><input type="checkbox" data-show="${k}" ${state.show[k] ? "checked" : ""} ${dis ? "disabled" : ""}> ${l}</label>`).join("")}
          </div>
          <div class="panel small muted">空格 播放/暂停 · ←/→ 单步 · 点击下方时间轴跳转。时间是仿真秒；1× = 每秒 ${BASE_RATE} 仿真秒。</div>
        </div>
      </div>`;

    const canvas = root.querySelector(".replay-canvas");
    const timeline = root.querySelector(".replay-timeline");
    const slider = root.querySelector(".replay-slider");
    const clock = root.querySelector(".replay-clock");
    const hud = root.querySelector(".replay-hud");
    const playBtn = root.querySelector('[data-act="play"]');

    // World bounds: sector + obstacles + goal + both tracks.
    const pts = [...st.sector, ...st.obstacles.flat(), st.goal, ...frames.map((f) => f.agent), ...(ghost || []).map((f) => f.agent)];
    let minX = Math.min(...pts.map((p) => p[0])), maxX = Math.max(...pts.map((p) => p[0]));
    let minY = Math.min(...pts.map((p) => p[1])), maxY = Math.max(...pts.map((p) => p[1]));
    const padW = (maxX - minX) * 0.06 + 5, padH = (maxY - minY) * 0.06 + 5;
    minX -= padW; maxX += padW; minY -= padH; maxY += padH;

    let W = 0, H = 0, scale = 1, ox = 0, oy = 0, dpr = 1;
    function resize() {
      dpr = window.devicePixelRatio || 1;
      W = canvas.clientWidth; H = Math.max(320, Math.min(W * 0.7, window.innerHeight - 330));
      canvas.style.height = H + "px";
      canvas.width = W * dpr; canvas.height = H * dpr;
      scale = Math.min(W / (maxX - minX), H / (maxY - minY));
      ox = (W - (maxX - minX) * scale) / 2; oy = (H - (maxY - minY) * scale) / 2;
      timeline.width = timeline.clientWidth * dpr; timeline.height = (ghost ? 54 : 30) * dpr;
      timeline.style.height = (ghost ? 54 : 30) + "px";
      drawTimeline(); draw();
    }
    const X = (x) => ox + (x - minX) * scale;
    const Y = (y) => H - oy - (y - minY) * scale;

    function poly(ctx, ring) { ctx.beginPath(); ring.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.closePath(); }

    function plane(ctx, x, y, hdg, size, fill, stroke) {
      const r = (hdg * Math.PI) / 180, s = Math.sin(r), c = Math.cos(r);
      const tip = [X(x) + s * size, Y(y) - c * size];
      const l = [X(x) + (-s * 0.6 - c * 0.55) * size, Y(y) - (-c * 0.6 + s * 0.55) * size];
      const rr = [X(x) + (-s * 0.6 + c * 0.55) * size, Y(y) - (-c * 0.6 - s * 0.55) * size];
      const tail = [X(x) - s * 0.3 * size, Y(y) + c * 0.3 * size];
      ctx.beginPath(); ctx.moveTo(...tip); ctx.lineTo(...l); ctx.lineTo(...tail); ctx.lineTo(...rr); ctx.closePath();
      ctx.fillStyle = fill; ctx.fill();
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.stroke(); }
    }

    // Segment i runs frames[i-1] -> frames[i] and is coloured by frames[i]'s status.
    function trail(ctx, fs, upto, t, colorBase, dashed) {
      const cur = interpAgent(fs, t);
      ctx.lineWidth = dashed ? 1.5 : 2.4;
      ctx.setLineDash(dashed ? [5, 4] : []);
      const seg = (a, b, fr) => {
        ctx.strokeStyle = (!dashed && statusColor(fr)) || colorBase;
        ctx.beginPath(); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); ctx.stroke();
      };
      for (let i = 1; i <= upto; i++) seg(fs[i - 1].agent, fs[i].agent, fs[i]);
      if (upto + 1 < fs.length) seg(fs[upto].agent, [cur.x, cur.y], fs[upto + 1]);
      ctx.setLineDash([]);
    }

    function draw() {
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
      const t = state.t;
      const cur = interpAgent(frames, t);
      const fr = frames[cur.k];

      // Static geometry.
      if (state.show.routes) {
        ctx.strokeStyle = C.route; ctx.lineWidth = 1;
        for (const r of Object.values(routeById)) { ctx.beginPath(); r.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke(); }
      }
      st.obstacles.forEach((ring) => {
        poly(ctx, ring); ctx.fillStyle = C.obstacle; ctx.globalAlpha = 0.85; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = C.obstacleEdge; ctx.lineWidth = 1; ctx.stroke();
      });
      poly(ctx, st.sector); ctx.strokeStyle = C.sector; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.arc(X(st.goal[0]), Y(st.goal[1]), st.goal_radius_km * scale, 0, 2 * Math.PI);
      ctx.setLineDash([4, 3]); ctx.strokeStyle = C.goal; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.goal; ctx.font = "bold 13px sans-serif"; ctx.textAlign = "center"; ctx.fillText("★", X(st.goal[0]), Y(st.goal[1]) + 5);

      // Ghost.
      let g = null;
      if (ghost && state.show.ghost) {
        g = interpAgent(ghost, Math.min(t, ghost[ghost.length - 1].t));
        if (state.show.trail) trail(ctx, ghost, g.k, t, C.ghost, true);
        plane(ctx, g.x, g.y, g.hdg, 9, C.ghost);
      }

      // Intruders: interpolate by id between frame k and k+1.
      const nxt = frames[cur.k + 1] || fr;
      const f = locate(frames, t)[1];
      const nextById = Object.fromEntries(nxt.intruders.map((it) => [it[0], it]));
      const conflict = new Set(fr.conflict_with || []);
      for (const it of fr.intruders) {
        const n = nextById[it[0]] || it;
        const x = lerp(it[1], n[1], f), y = lerp(it[2], n[2], f), h = lerpAngle(it[3], n[3], f);
        const hot = conflict.has(it[0]);
        if (state.show.circles) {
          ctx.beginPath(); ctx.arc(X(x), Y(y), st.intrusion_radius_km * scale, 0, 2 * Math.PI);
          ctx.fillStyle = hot ? "rgba(255,107,107,0.22)" : "rgba(227,233,245,0.05)"; ctx.fill();
          ctx.strokeStyle = hot ? C.conflict : "rgba(227,233,245,0.25)"; ctx.lineWidth = hot ? 1.6 : 1; ctx.stroke();
        }
        plane(ctx, x, y, h, 7, hot ? C.conflict : C.intruder);
      }

      // Agent.
      if (state.show.trail) trail(ctx, frames, cur.k, t, C.agent, false);
      ctx.beginPath(); ctx.arc(X(cur.x), Y(cur.y), 13, 0, 2 * Math.PI);
      ctx.strokeStyle = statusColor(fr) || C.agent; ctx.globalAlpha = 0.5; ctx.lineWidth = 2; ctx.stroke(); ctx.globalAlpha = 1;
      plane(ctx, cur.x, cur.y, cur.hdg, 15, statusColor(fr) || C.agent, "#0b1019");

      // Scale bar + north arrow.
      const barKm = niceKm((maxX - minX) / 6);
      ctx.strokeStyle = C.muted; ctx.fillStyle = C.muted; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(16, H - 18); ctx.lineTo(16 + barKm * scale, H - 18); ctx.stroke();
      ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.fillText(`${barKm} km`, 16, H - 24);
      ctx.textAlign = "center"; ctx.fillText("N ↑", W - 22, 22);

      // Clock overlay.
      ctx.textAlign = "left"; ctx.fillStyle = C.text; ctx.font = "12px ui-monospace, monospace";
      ctx.fillText(`t = ${t.toFixed(0)} s`, 14, 20);

      slider.value = t;
      clock.textContent = `${t.toFixed(0)} / ${tEnd.toFixed(0)} s`;
      renderHud(cur, fr, g);
      drawTimelineCursor();
    }

    function niceKm(v) { const steps = [5, 10, 20, 25, 50, 100]; return steps.find((s) => s >= v) || 100; }

    function renderHud(cur, fr, g) {
      const m = fr.metrics, cfg = opts.mdp || {};
      const goalRel = (cfg.ACTION_MODE || "relative") === "goal_relative";
      const a = fr.action;
      const hdgTxt = a ? (goalRel ? `目标方位 ${a[0] >= 0 ? "+" : ""}${(a[0] * (cfg.GOAL_MAX_OFFSET_DEG ?? 90)).toFixed(0)}°`
        : `转向 ${a[0] >= 0 ? "+" : ""}${(a[0] * (cfg.D_HEADING ?? 15)).toFixed(1)}°`) : "—";
      const spdTxt = a ? (goalRel ? `标称 ${a[1] >= 0 ? "+" : ""}${(a[1] * (cfg.SPEED_RANGE_KT ?? 40)).toFixed(0)} kt`
        : `${a[1] >= 0 ? "+" : ""}${(a[1] * (cfg.D_SPEED ?? 25)).toFixed(1)} kt`) : "—";
      const dGoal = Math.hypot(cur.x - st.goal[0], cur.y - st.goal[1]);
      const gm = g && ghost ? ghost[g.k].metrics : null;
      const row = (label, v, nv, unit = "s", lowerBetter = true) => {
        let cls = "";
        if (nv !== undefined && nv !== null) cls = v === nv ? "" : (lowerBetter ? v < nv : v > nv) ? "good" : "bad";
        return `<tr><td>${label}</td><td class="${cls}">${fmtN(v)}${unit}</td>${gm ? `<td class="muted">${fmtN(nv)}${unit}</td>` : ""}</tr>`;
      };
      const statusBits = [];
      if (fr.conflict_with && fr.conflict_with.length) statusBits.push(`<span class="bad">冲突：${fr.conflict_with.join(", ")}</span>`);
      if (fr.in_restricted) statusBits.push(`<span style="color:${C.restricted}">在限制区内</span>`);
      if (fr.outside) statusBits.push(`<span class="warn">在扇区外</span>`);
      if (m.waypoint_reached) statusBits.push(`<span class="good">已到达目标</span>`);
      const terms = Object.entries(fr.terms || {}).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).slice(0, 4);
      hud.innerHTML = `
        <div class="hud-status">${statusBits.join(" · ") || '<span class="muted">正常飞行</span>'}</div>
        <table class="data hud-table">
          <tr><td>航向 / 速度</td><td colspan="2">${cur.hdg.toFixed(0)}° · ${cur.spd.toFixed(0)} kt</td></tr>
          <tr><td>距目标</td><td colspan="2">${dGoal.toFixed(1)} km</td></tr>
          <tr><td>动作（航向）</td><td colspan="2">${hdgTxt}</td></tr>
          <tr><td>动作（速度）</td><td colspan="2">${spdTxt}</td></tr>
          <tr><td>本步奖励 / 累计</td><td colspan="2">${fmtN(fr.reward, 2)} / ${fmtN(cum[cur.k], 1)}</td></tr>
          ${terms.length ? `<tr><td>本步主要分项</td><td colspan="2" class="small">${terms.map(([k, v]) => `${k} ${fmtN(v, 2)}`).join("<br>")}</td></tr>` : ""}
        </table>
        <table class="data hud-table" style="margin-top:8px">
          <tr><th>累计指标</th><th>模型</th>${gm ? "<th>中性</th>" : ""}</tr>
          ${row("冲突时间", m.intrusion_time, gm && gm.intrusion_time)}
          ${row("冲突次数", m.intrusion_events, gm && gm.intrusion_events, "")}
          ${row("限制区时间", m.time_in_restricted_area, gm && gm.time_in_restricted_area)}
          ${row("扇区外时间", m.time_outside_sector, gm && gm.time_outside_sector)}
          ${row("飞行时间", m.flight_time, gm && gm.flight_time)}
        </table>`;
    }
    function fmtN(v, d = 0) { return v === undefined || v === null ? "—" : Number(v).toFixed(d); }

    function drawTimeline() {
      const ctx = timeline.getContext("2d", { willReadFrequently: true });
      const w = timeline.clientWidth, rows = ghost ? [["模型", frames], ["中性", ghost]] : [["模型", frames]];
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, timeline.height);
      const lab = 36, rowH = 22;
      rows.forEach(([name, fs], r) => {
        const y = 4 + r * (rowH + 4);
        ctx.fillStyle = "#1c2538"; ctx.fillRect(lab, y, w - lab, rowH);
        const end = fs[fs.length - 1].t;
        ctx.fillStyle = "#26324a"; ctx.fillRect(lab, y, ((w - lab) * end) / tEnd, rowH);
        for (let i = 1; i < fs.length; i++) {
          const col = statusColor(fs[i]);
          if (!col) continue;
          const x0 = lab + ((w - lab) * fs[i - 1].t) / tEnd, x1 = lab + ((w - lab) * fs[i].t) / tEnd;
          ctx.fillStyle = col; ctx.fillRect(x0, y + 3, Math.max(1, x1 - x0), rowH - 6);
        }
        if (fs[fs.length - 1].metrics.waypoint_reached) {
          ctx.fillStyle = C.goal; ctx.fillRect(lab + ((w - lab) * end) / tEnd - 2, y, 3, rowH);
        }
        ctx.fillStyle = C.muted; ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.fillText(name, 2, y + 15);
      });
      timeline._base = ctx.getImageData(0, 0, timeline.width, timeline.height);
    }
    function drawTimelineCursor() {
      if (!timeline._base) return;
      const ctx = timeline.getContext("2d", { willReadFrequently: true });
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.putImageData(timeline._base, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const lab = 36, w = timeline.clientWidth;
      const x = lab + ((w - lab) * state.t) / tEnd;
      ctx.fillStyle = "#ffffff"; ctx.fillRect(x - 1, 0, 2, timeline.height / dpr);
    }

    function tick(ts) {
      if (!state.playing) return;
      if (state.last !== null) state.t = Math.min(tEnd, state.t + ((ts - state.last) / 1000) * BASE_RATE * state.speed);
      state.last = ts;
      draw();
      if (state.t >= tEnd) { setPlaying(false); return; }
      state.raf = requestAnimationFrame(tick);
    }
    function setPlaying(p) {
      state.playing = p; state.last = null;
      playBtn.textContent = p ? "⏸ 暂停" : "▶ 播放";
      if (p) { if (state.t >= tEnd) state.t = 0; state.raf = requestAnimationFrame(tick); }
      else cancelAnimationFrame(state.raf);
    }
    function stepBy(dir) {
      setPlaying(false);
      const k = locate(frames, state.t)[0];
      const target = frames[Math.max(0, Math.min(frames.length - 1, k + dir))];
      state.t = dir > 0 && target.t <= state.t && k + 2 < frames.length ? frames[k + 2].t : target.t;
      draw();
    }

    root.querySelector('[data-act="play"]').onclick = () => setPlaying(!state.playing);
    root.querySelector('[data-act="start"]').onclick = () => { setPlaying(false); state.t = 0; draw(); };
    root.querySelector('[data-act="end"]').onclick = () => { setPlaying(false); state.t = tEnd; draw(); };
    root.querySelector('[data-act="back"]').onclick = () => stepBy(-1);
    root.querySelector('[data-act="fwd"]').onclick = () => stepBy(1);
    root.querySelector('[data-act="speed"]').onchange = (e) => { state.speed = Number(e.target.value); };
    slider.oninput = () => { state.t = Number(slider.value); draw(); };
    timeline.onclick = (e) => {
      const rect = timeline.getBoundingClientRect(), lab = 36;
      state.t = Math.max(0, Math.min(tEnd, ((e.clientX - rect.left - lab) / (rect.width - lab)) * tEnd));
      draw();
    };
    root.querySelectorAll("[data-show]").forEach((cb) => (cb.onchange = () => { state.show[cb.dataset.show] = cb.checked; draw(); }));
    const onKey = (e) => {
      if (!document.body.contains(root) || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && document.activeElement.type !== "range") return;
      if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
      if (e.code === "ArrowRight") { e.preventDefault(); stepBy(1); }
      if (e.code === "ArrowLeft") { e.preventDefault(); stepBy(-1); }
    };
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", resize);
    resize();
    if (opts.autoplay) setPlaying(true);

    return {
      destroy() {
        setPlaying(false);
        document.removeEventListener("keydown", onKey);
        window.removeEventListener("resize", resize);
      },
    };
  }

  window.ReplayViewer = { mount };
})();
