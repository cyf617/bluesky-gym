(function () {
  const { lineChart, barChart, fmt, ema } = window.Charts;
  const PALETTE = ["#5b9dff", "#3ecf8e", "#f5b84a", "#ff6b6b", "#b38cff", "#4dd0e1", "#ff8fb1", "#a3e635", "#f97316", "#94a3b8"];
  const METRICS = [
    { key: "waypoint_reached", label: "到达率", unit: "%", scale: 100, better: "high" },
    { key: "intrusion_time", label: "冲突时间", unit: "s", better: "low" },
    { key: "intrusion_events", label: "冲突次数", better: "low" },
    { key: "time_in_restricted_area", label: "限制区内时间", unit: "s", better: "low" },
    { key: "restricted_area_events", label: "进入限制区次数", better: "low" },
    { key: "time_outside_sector", label: "扇区外时间", unit: "s", better: "low" },
    { key: "sector_exit_events", label: "飞出扇区次数", better: "low" },
    { key: "flight_time", label: "飞行时间", unit: "s", better: "low" },
  ];
  const TERMS = {
    progress: "进度", reach: "到达", timeout: "超时", conflict_now: "当前冲突",
    conflict_predicted: "预测冲突", obstacle_proximity: "接近障碍", sector_proximity: "接近边界",
    restricted_event: "进入限制区", sector_exit_event: "飞出扇区",
    intrusion_time: "冲突秒数", restricted_time: "限制区秒数", drift: "偏航",
  };
  const SAFETY_TERMS = ["conflict_now", "conflict_predicted", "obstacle_proximity", "sector_proximity", "restricted_event", "sector_exit_event", "intrusion_time", "restricted_time"];
  const TUNABLE_HELP = {
    ACTION_MODE: "动作模式：relative = 在当前航向/速度上增减；goal_relative = 相对目标方位的偏角 + 绝对速度（0 动作 = 直飞目标）",
    D_HEADING: "relative 模式：|动作|=1 时的航向变化（度）",
    D_SPEED: "relative 模式：|动作|=1 时的速度变化（kt）",
    ACTION_FREQUENCY: "每个决策步对应的仿真秒数",
    GOAL_MAX_OFFSET_DEG: "goal_relative 模式：|动作|=1 时相对目标方位的最大偏角（度）",
    SPEED_RANGE_KT: "goal_relative 模式：|动作|=1 时相对标称速度的变化（kt）",
    PROGRESS_PER_KM: "每接近目标 1 km 的奖励",
    REACH_REWARD: "到达目标奖励",
    TIMEOUT_PENALTY: "3000 s 用完未到达的惩罚",
    DRIFT_PENALTY: "无威胁时每步偏航惩罚系数（乘以偏航弧度）",
    TCPA_HORIZON_S: "预测冲突的时间窗口（s）",
    DCPA_LIMIT_NM: "预测冲突的最近距离阈值（NM）",
    CONFLICT_WEIGHT: "当前/预测冲突惩罚权重（每步每架邻机）",
    PROXIMITY_HORIZON_KM: "障碍物/扇区边界接近惩罚的作用距离（km）",
    PROXIMITY_WEIGHT: "障碍物/扇区边界接近惩罚权重（每步）",
    RESTRICTED_EVENT_PENALTY: "每次进入限制区的惩罚",
    SECTOR_EXIT_EVENT_PENALTY: "每次飞出扇区的惩罚",
    INTRUSION_TIME_WEIGHT: "每秒处于冲突（< 5 NM，每架邻机）的惩罚，对应官方 intrusion_time",
    RESTRICTED_TIME_WEIGHT: "每秒处于限制区内的惩罚，对应官方 time_in_restricted_area",
  };

  const saved = JSON.parse(localStorage.getItem("bsdash") || "{}");
  const state = {
    runs: [], baselines: {}, tunables: null, details: {},
    focused: saved.focused || null,
    compare: new Set(saved.compare || []),
    tab: saved.tab || "overview",
    smooth: saved.smooth ?? 0.6,
  };
  const persist = () => localStorage.setItem("bsdash", JSON.stringify({
    focused: state.focused, compare: [...state.compare], tab: state.tab, smooth: state.smooth,
  }));

  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const colorOf = (name) => PALETTE[Math.max(0, state.runs.findIndex((r) => r.name === name)) % PALETTE.length];
  const shown = () => {
    const names = state.runs.map((r) => r.name).filter((n) => state.compare.has(n) || n === state.focused);
    return names.length ? names : (state.focused ? [state.focused] : []);
  };
  const STATE_LABEL = { running: "训练中", finished: "已完成", stopped: "已中断" };
  // Evals run on a validation seed; compare against the neutral policy on that seed.
  function baselineFor(name) {
    const run = state.runs.find((r) => r.name === name);
    const seed = String(run && run.eval_seed != null ? run.eval_seed : 123);
    const b = state.baselines[seed];
    if (b) return { ...b, matched: true };
    const fallback = state.baselines["42"] || Object.values(state.baselines)[0];
    return fallback ? { ...fallback, matched: false, wanted: seed } : null;
  }

  async function getJSON(url) {
    const res = await fetch(url, { cache: "no-store" });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    return body;
  }

  function ago(ts) {
    if (!ts) return "";
    const s = Date.now() / 1000 - ts;
    if (s < 60) return `${s.toFixed(0)} 秒前`;
    if (s < 3600) return `${(s / 60).toFixed(0)} 分钟前`;
    if (s < 86400) return `${(s / 3600).toFixed(1)} 小时前`;
    return `${(s / 86400).toFixed(1)} 天前`;
  }
  const duration = (s) => !isFinite(s) || s <= 0 ? "—" : s < 3600 ? `${(s / 60).toFixed(0)} 分钟` : `${(s / 3600).toFixed(1)} 小时`;
  const metricVal = (m, v) => v === null || v === undefined ? null : v * (m.scale || 1);
  const metricFmt = (m, v) => v === null || v === undefined ? "—" : m.unit === "%" ? `${v.toFixed(0)}%` : fmt(v);

  // ------------------------------------------------------------------ data
  async function refresh() {
    try {
      const data = await getJSON("/api/runs");
      state.runs = data.runs;
      state.baselines = data.baselines || {};
      $("#results-dir").textContent = data.results_dir;
      if (!state.focused || !state.runs.some((r) => r.name === state.focused)) {
        const latest = [...state.runs].sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))[0];
        state.focused = latest ? latest.name : null;
      }
      for (const n of [...state.compare]) if (!state.runs.some((r) => r.name === n)) state.compare.delete(n);
      if (!state.tunables) state.tunables = await getJSON("/api/tunables").catch(() => null);
      await Promise.all(shown().map(async (n) => { state.details[n] = await getJSON(`/api/run/${encodeURIComponent(n)}`); }));
      $("#last-updated").textContent = `更新于 ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      $("#last-updated").innerHTML = `<span class="bad">刷新失败：${esc(err.message)}</span>`;
    }
    renderSidebar();
    renderTab();
  }

  // --------------------------------------------------------------- sidebar
  function renderSidebar() {
    const list = $("#run-list");
    if (!state.runs.length) {
      list.innerHTML = `<div class="empty">results 目录下还没有运行。<br>运行 <code>python -m scripts.train_sa_sac</code> 开始训练。</div>`;
    } else {
      list.innerHTML = state.runs.map((r) => {
        const ev = r.last_eval || {};
        const pct = r.total_timesteps ? Math.min(100, (100 * (r.timesteps || 0)) / r.total_timesteps) : 0;
        return `<div class="run-card ${r.name === state.focused ? "focused" : ""}" data-run="${esc(r.name)}" style="--run-color:${colorOf(r.name)}">
          <div class="row">
            <label class="row" style="gap:6px" onclick="event.stopPropagation()">
              <input type="checkbox" data-compare="${esc(r.name)}" ${state.compare.has(r.name) ? "checked" : ""}>
              <span class="run-name">${esc(r.name)}</span>
            </label>
            <span class="badge ${r.state}">${STATE_LABEL[r.state] || r.state}</span>
          </div>
          <div class="run-meta">
            ${r.algo ? `<span class="tag">${r.algo.toUpperCase()}</span>` : ""}
            ${r.action_mode ? `<span class="tag">${esc(r.action_mode)}</span>` : ""}
            ${fmt(r.timesteps)} / ${fmt(r.total_timesteps)} 步 · ${ago(r.updated_at)}
          </div>
          <div class="run-kpis">
            <span>到达 <b>${ev.waypoint_reached != null ? (100 * ev.waypoint_reached).toFixed(0) + "%" : "—"}</b></span>
            <span>冲突 <b>${ev.intrusion_time != null ? fmt(ev.intrusion_time) + "s" : "—"}</b></span>
            <span>限制区 <b>${ev.time_in_restricted_area != null ? fmt(ev.time_in_restricted_area) + "s" : "—"}</b></span>
          </div>
          <div class="progress"><div style="width:${pct}%"></div></div>
        </div>`;
      }).join("");
    }
    list.querySelectorAll(".run-card").forEach((card) => card.addEventListener("click", () => {
      state.focused = card.dataset.run; persist(); refresh();
    }));
    list.querySelectorAll("input[data-compare]").forEach((cb) => cb.addEventListener("change", () => {
      cb.checked ? state.compare.add(cb.dataset.compare) : state.compare.delete(cb.dataset.compare);
      persist(); refresh();
    }));

    const seeds = Object.keys(state.baselines).sort((a, b) => Number(a) - Number(b));
    $("#baseline-box").innerHTML = seeds.length ? `<b>中性 [0,0] 基线</b>
      <table><tr><td></td>${seeds.map((sd) => `<td style="text-align:right">seed ${sd}</td>`).join("")}</tr>
      ${METRICS.map((m) => `<tr><td>${m.label}</td>${seeds.map((sd) => `<td style="text-align:right">${metricFmt(m, metricVal(m, state.baselines[sd].means[m.key]))}</td>`).join("")}</tr>`).join("")}</table>
      <div class="small" style="margin-top:6px">验证评估默认用 seed 123；官方成绩用 seed 42。</div>`
      : "未找到 baseline_sa_neutral*.json（运行 <code>python -m scripts.run_neutral_baseline --seed 123</code>）";
  }

  // ------------------------------------------------------------------ tabs
  function renderTab() {
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === state.tab));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${state.tab}`));
    ({ overview: renderOverview, metrics: renderMetrics, training: renderTraining,
       reward: renderReward, episodes: renderEpisodes, config: renderConfig })[state.tab]();
  }

  const focusedDetail = () => state.details[state.focused];
  const lastIdx = (cols, key) => (cols[key] || []).length - 1;

  function recentTermMeans(detail, fromEnd = true, n = 50) {
    const ep = detail.episodes || {};
    const out = {};
    for (const t of Object.keys(TERMS)) {
      const col = ep[`r_${t}`];
      if (!col || !col.length) continue;
      const slice = fromEnd ? col.slice(-n) : col.slice(0, n);
      const vals = slice.filter((v) => v !== null);
      if (vals.length) out[t] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    if (Object.keys(out).length) return out;
    const p = detail.progress || {};
    for (const t of Object.keys(TERMS)) {
      const col = (p[`terms/${t}`] || []).filter((v) => v !== null);
      if (col.length) out[t] = fromEnd ? col[col.length - 1] : col[0];
    }
    return out;
  }

  function diagnose(detail) {
    const hints = [];
    const ev = detail.summary.last_eval;
    const bl = baselineFor(detail.summary.name);
    const base = bl ? bl.means : {};
    if (!ev) {
      hints.push(["muted", "还没有评估结果：第一次评估在第 0 步（未训练策略），之后每 eval_freq 步一次。"]);
      return hints;
    }
    if (ev.waypoint_reached < 0.5 && ev.time_outside_sector > 300) {
      hints.push(["bad", "到达率很低且长时间在扇区外：智能体在“逃出扇区躲交通”（run0 的失败模式）。建议用 goal_relative 动作模式，或提高进度奖励 / 超时惩罚。"]);
    } else if (ev.waypoint_reached < 0.5) {
      hints.push(["bad", "到达率低于 50%：relative 动作模式下探索噪声会累积成随机游走、甚至原地绕圈。建议改用 goal_relative。"]);
    }
    if (ev.waypoint_reached >= 0.9 && base.intrusion_time && ev.intrusion_time >= 0.9 * base.intrusion_time) {
      hints.push(["warn", `到达率正常，但冲突时间（${fmt(ev.intrusion_time)} s）没有明显低于基线（${fmt(base.intrusion_time)} s）：安全惩罚相对“到达 + 进度”可能太弱。可以加大 CONFLICT_WEIGHT / PROXIMITY_WEIGHT。`]);
    }
    if (ev.waypoint_reached >= 0.9 && base.time_in_restricted_area && ev.time_in_restricted_area >= 0.9 * base.time_in_restricted_area) {
      hints.push(["warn", "限制区内时间没有比基线好：当前只有“进入事件”惩罚和 5 km 内的接近惩罚，可以加大 PROXIMITY_WEIGHT 或 RESTRICTED_EVENT_PENALTY。"]);
    }
    const terms = recentTermMeans(detail);
    if (Object.keys(terms).length) {
      const task = Math.abs((terms.reach || 0) + (terms.progress || 0));
      const safety = Math.abs(SAFETY_TERMS.reduce((a, t) => a + (terms[t] || 0), 0));
      if (task > 0 && safety > 0 && task / safety > 10) {
        hints.push(["warn", `奖励分解：任务项（到达 + 进度 ≈ ${fmt(task)}）是安全项（≈ ${fmt(safety)}）的 ${fmt(task / safety)} 倍，安全信号可能被淹没。`]);
      }
    }
    if (ev.waypoint_reached >= 0.9 && base.intrusion_time && ev.intrusion_time < 0.7 * base.intrusion_time) {
      hints.push(["good", `冲突时间比基线低 ${(100 * (1 - ev.intrusion_time / base.intrusion_time)).toFixed(0)}%，且到达率保持在 ${(100 * ev.waypoint_reached).toFixed(0)}%。`]);
    }
    if (!hints.length) hints.push(["muted", "没有发现明显问题。"]);
    return hints;
  }

  function renderOverview() {
    const root = $("#tab-overview");
    const d = focusedDetail();
    if (!d) { root.innerHTML = `<div class="empty">选择左侧的一个运行</div>`; return; }
    const s = d.summary, cfg = d.config || {};
    const bl = baselineFor(s.name);
    const base = bl ? bl.means : {};
    const ev = s.last_eval;
    const eta = s.state === "running" && s.fps ? ((s.total_timesteps || 0) - (s.timesteps || 0)) / s.fps : NaN;
    const kpis = METRICS.map((m) => {
      const v = ev ? metricVal(m, ev[m.key]) : null;
      const bv = metricVal(m, base[m.key]);
      let delta = "";
      if (v !== null && bv !== null && bv !== undefined) {
        const diff = v - bv;
        const good = m.better === "high" ? diff >= 0 : diff <= 0;
        const cls = Math.abs(diff) < 1e-9 ? "muted" : good ? "good" : "bad";
        delta = `<span class="${cls}">${diff >= 0 ? "+" : ""}${metricFmt(m, diff)}</span> <span class="muted">vs 基线 ${metricFmt(m, bv)}</span>`;
      }
      return `<div class="panel kpi"><div class="label">${m.label}${m.unit && m.unit !== "%" ? ` (${m.unit})` : ""}</div>
        <div class="value">${metricFmt(m, v)}</div><div class="delta">${delta}</div></div>`;
    }).join("");
    const sel = d.selection || {};
    root.innerHTML = `
      <div class="panel">
        <div class="row" style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
          <div>
            <h2 style="font-size:16px">${esc(s.name)} <span class="badge ${s.state}">${STATE_LABEL[s.state] || s.state}</span></h2>
            <div class="muted small" style="margin-top:4px">
              ${s.algo ? s.algo.toUpperCase() : "?"} · 动作模式 ${esc(s.action_mode || "relative（旧运行）")} ·
              覆盖参数 ${s.mdp_overrides && Object.keys(s.mdp_overrides).length ? esc(Object.entries(s.mdp_overrides).map(([k, v]) => `${k}=${v}`).join(", ")) : "无"}
              ${cfg.git_commit ? ` · commit ${esc(cfg.git_commit)}` : ""}
            </div>
          </div>
          <div class="muted small" style="text-align:right">
            ${fmt(s.timesteps)} / ${fmt(s.total_timesteps)} 步${s.fps ? ` · ${fmt(s.fps)} fps` : ""}${isFinite(eta) ? ` · 预计剩余 ${duration(eta)}` : ""}<br>
            最近更新 ${ago(s.updated_at)} · 评估 ${s.n_evals} 次
          </div>
        </div>
        ${cfg.command ? `<pre class="cmd">${esc(cfg.command)}</pre>` : ""}
      </div>
      <h3 class="section-title">最近一次评估${ev ? `（第 ${fmt(ev.timesteps)} 步，验证 seed ${s.eval_seed ?? 123}）` : ""}
        <span class="muted small">${bl ? (bl.matched ? `对比基线：${esc(bl.label)}` : `<span class="warn">没有 seed ${bl.wanted} 的基线，暂用 ${esc(bl.label)}；运行 python -m scripts.run_neutral_baseline --seed ${bl.wanted}</span>`) : ""}</span></h3>
      <div class="grid kpis">${kpis}</div>
      <h3 class="section-title">自动诊断</h3>
      <div class="panel">${diagnose(d).map(([cls, text]) => `<p class="${cls}" style="margin:4px 0">• ${esc(text)}</p>`).join("")}</div>
      <h3 class="section-title">检查点（只按官方指标选，不看训练回报）</h3>
      <div class="panel small">
        best_arrival：${sel.best_arrival_step != null ? `第 ${fmt(sel.best_arrival_step)} 步 · 到达 ${(100 * sel.best_arrival.waypoint_reached).toFixed(0)}% · 冲突 ${fmt(sel.best_arrival.intrusion_time)} s` : "—"}<br>
        best_safety（到达率 ≥ ${sel.min_arrival != null ? (100 * sel.min_arrival).toFixed(0) : 90}%）：${sel.best_safety_step != null ? `第 ${fmt(sel.best_safety_step)} 步 · 到达 ${(100 * sel.best_safety.waypoint_reached).toFixed(0)}% · 冲突 ${fmt(sel.best_safety.intrusion_time)} s · 限制区 ${fmt(sel.best_safety.time_in_restricted_area)} s` : "—"}<br>
        当前选用：<b>${esc(sel.selected || "—")}</b>
      </div>
      <h3 class="section-title">全部运行对比（最近一次评估）</h3>
      <div class="panel table-wrap">${compareTable()}</div>`;
    root.querySelectorAll("tr[data-run]").forEach((tr) => tr.addEventListener("click", () => {
      state.focused = tr.dataset.run; persist(); refresh();
    }));
  }

  function compareTable() {
    const bl = baselineFor(state.focused);
    const base = bl ? bl.means : null;
    const head = `<tr><th>运行</th><th>状态</th><th>算法</th><th>动作模式</th><th>步数</th>${METRICS.map((m) => `<th>${m.label}</th>`).join("")}</tr>`;
    const baseRow = base ? `<tr><td class="muted">中性基线 [0,0]（seed ${bl.seed}）</td><td></td><td></td><td></td><td></td>${METRICS.map((m) => `<td class="muted">${metricFmt(m, metricVal(m, base[m.key]))}</td>`).join("")}</tr>` : "";
    const rows = state.runs.map((r) => {
      const ev = r.last_eval || {};
      return `<tr data-run="${esc(r.name)}" style="cursor:pointer">
        <td><span style="color:${colorOf(r.name)}">●</span> ${esc(r.name)}</td>
        <td><span class="badge ${r.state}">${STATE_LABEL[r.state] || r.state}</span></td>
        <td>${(r.algo || "?").toUpperCase()}</td><td>${esc(r.action_mode || "—")}</td><td>${fmt(r.timesteps)}</td>
        ${METRICS.map((m) => {
          const v = metricVal(m, ev[m.key]);
          let cls = "";
          if (v !== null && base) {
            const bv = metricVal(m, base[m.key]);
            cls = m.better === "high" ? (v >= bv ? "good" : "bad") : (v <= bv ? "good" : "bad");
          }
          return `<td class="${cls}">${metricFmt(m, v)}</td>`;
        }).join("")}</tr>`;
    }).join("");
    return `<table class="data">${head}${baseRow}${rows}</table>`;
  }

  function chartPanel(grid, title, opts) {
    const panel = document.createElement("div");
    panel.className = "panel chart";
    panel.innerHTML = `<h3>${title}</h3><div class="chart-body"></div>`;
    grid.appendChild(panel);
    lineChart(panel.querySelector(".chart-body"), opts);
  }

  function renderMetrics() {
    const grid = $("#metrics-grid");
    grid.innerHTML = "";
    const names = shown();
    if (!names.length) { grid.innerHTML = `<div class="empty">选择左侧的运行</div>`; return; }
    for (const m of METRICS) {
      const series = names.map((n) => {
        const ev = (state.details[n] || {}).evals || {};
        const xs = ev.timesteps || [], ys = ev[m.key] || [], sd = ev[`${m.key}_std`] || [];
        const sc = m.scale || 1;
        return {
          name: n, color: colorOf(n),
          points: xs.map((x, i) => [x, ys[i] === null ? NaN : ys[i] * sc]),
          band: n === state.focused && sd.length
            ? xs.map((x, i) => [x, Math.max(0, ys[i] - sd[i]) * sc, (m.key === "waypoint_reached" ? Math.min(1, ys[i] + sd[i]) : ys[i] + sd[i]) * sc])
            : null,
        };
      });
      const bl = baselineFor(state.focused);
      const bv = bl ? bl.means[m.key] : undefined;
      chartPanel(grid, `${m.label}${m.unit ? ` (${m.unit})` : ""} <span class="muted">${m.better === "high" ? "越高越好" : "越低越好"}</span>`, {
        series,
        hLines: bv !== undefined ? [{ y: bv * (m.scale || 1), label: `中性基线 seed ${bl.seed}` }] : [],
        yMin: m.key === "waypoint_reached" ? 0 : undefined,
        yMax: m.key === "waypoint_reached" ? 100 : undefined,
        emptyText: "还没有评估数据",
      });
    }
  }

  function smoothSeries(names, key, scale = 1) {
    return names.map((n) => {
      const p = (state.details[n] || {}).progress || {};
      const xs = p["time/total_timesteps"] || [], ys = p[key] || [];
      const raw = xs.map((x, i) => [x, ys[i] === null || ys[i] === undefined ? NaN : ys[i] * scale]).filter(([, y]) => isFinite(y));
      return { name: n, color: colorOf(n), points: ema(raw, state.smooth), raw: state.smooth > 0 ? raw : null };
    });
  }

  function renderTraining() {
    $("#smooth").value = state.smooth;
    $("#smooth-val").textContent = Number(state.smooth).toFixed(2);
    const grid = $("#training-grid");
    grid.innerHTML = "";
    const names = shown();
    if (!names.length) { grid.innerHTML = `<div class="empty">选择左侧的运行</div>`; return; }
    const charts = [
      ["rollout/ep_rew_mean", "训练回报 ep_rew_mean（不用于选模型）"],
      ["rollout/ep_len_mean", "回合长度 ep_len_mean（决策步）"],
      ["train_metrics/waypoint_reached", "训练回合到达率 (%)", 100],
      ["train_metrics/intrusion_time", "训练回合冲突时间 (s)"],
      ["train_metrics/time_in_restricted_area", "训练回合限制区时间 (s)"],
      ["train_metrics/time_outside_sector", "训练回合扇区外时间 (s)"],
    ];
    const keys = new Set();
    for (const n of names) for (const k of Object.keys((state.details[n] || {}).progress || {})) if (k.startsWith("train/") && k !== "train/learning_rate" && k !== "train/n_updates" && k !== "train/clip_range") keys.add(k);
    for (const k of [...keys].sort()) charts.push([k, k]);
    charts.push(["time/fps", "吞吐 fps"]);
    for (const [key, title, scale] of charts) {
      chartPanel(grid, title, { series: smoothSeries(names, key, scale || 1), emptyText: "这个运行没有记录该项" });
    }
  }

  function renderReward() {
    const root = $("#reward-content");
    const d = focusedDetail();
    if (!d) { root.innerHTML = `<div class="empty">选择左侧的运行</div>`; return; }
    const recent = recentTermMeans(d, true), early = recentTermMeans(d, false);
    if (!Object.keys(recent).length) {
      root.innerHTML = `<div class="panel empty">这个运行没有奖励分项记录（旧版训练脚本）。用新版 <code>scripts.train_sa_sac</code> 重新训练后会自动记录。</div>`;
      return;
    }
    const order = Object.keys(TERMS).filter((t) => t in recent);
    const items = (m) => order.map((t) => ({ label: TERMS[t] + ` (${t})`, value: m[t] ?? 0 }));
    const task = (recent.reach || 0) + (recent.progress || 0);
    const safety = SAFETY_TERMS.reduce((a, t) => a + (recent[t] || 0), 0);
    root.innerHTML = `
      <div class="grid kpis" style="margin-bottom:14px">
        <div class="panel kpi"><div class="label">任务项（到达 + 进度）</div><div class="value good">${fmt(task)}</div><div class="delta muted">最近 50 个训练回合平均</div></div>
        <div class="panel kpi"><div class="label">安全项合计</div><div class="value bad">${fmt(safety)}</div><div class="delta muted">冲突 + 接近 + 事件</div></div>
        <div class="panel kpi"><div class="label">任务 / 安全 比值</div><div class="value ${Math.abs(task / (safety || 1e-9)) > 10 ? "warn" : ""}">${safety ? fmt(Math.abs(task / safety)) + "×" : "—"}</div><div class="delta muted">&gt;10× 时安全信号容易被淹没</div></div>
        <div class="panel kpi"><div class="label">超时项</div><div class="value">${fmt(recent.timeout || 0)}</div><div class="delta muted">非零说明有回合没到达</div></div>
      </div>
      <div class="two-col">
        <div class="panel"><h3>训练初期（前 50 个回合）</h3><div id="bar-early"></div></div>
        <div class="panel"><h3>最近（最后 50 个回合）</h3><div id="bar-recent"></div></div>
      </div>
      <h3 class="section-title">各分项随训练变化</h3>
      <div class="grid charts" id="terms-grid"></div>
      <h3 class="section-title">对比运行（最近 50 个回合的分项平均）</h3>
      <div class="panel table-wrap" id="terms-compare"></div>`;
    barChart($("#bar-early"), items(early));
    barChart($("#bar-recent"), items(recent));

    const grid = $("#terms-grid");
    const p = d.progress || {};
    const xs = p["time/total_timesteps"] || [];
    const termSeries = order.map((t, i) => {
      const raw = xs.map((x, j) => [x, (p[`terms/${t}`] || [])[j]]).filter(([, y]) => y !== null && y !== undefined);
      return { name: TERMS[t], color: PALETTE[i % PALETTE.length], points: ema(raw, state.smooth) };
    });
    chartPanel(grid, "安全相关分项（回合总和，EMA 平滑）", { series: termSeries.filter((s, i) => SAFETY_TERMS.includes(order[i]) || order[i] === "drift"), emptyText: "progress.csv 中没有 terms/*" });
    chartPanel(grid, "任务相关分项（回合总和，EMA 平滑）", { series: termSeries.filter((s, i) => ["progress", "reach", "timeout"].includes(order[i])), emptyText: "progress.csv 中没有 terms/*" });

    const names = shown();
    const per = names.map((n) => [n, state.details[n] ? recentTermMeans(state.details[n]) : {}]);
    $("#terms-compare").innerHTML = `<table class="data"><tr><th>分项</th>${per.map(([n]) => `<th style="color:${colorOf(n)}">${esc(n)}</th>`).join("")}</tr>
      ${Object.keys(TERMS).map((t) => `<tr><td>${TERMS[t]} <span class="muted">${t}</span></td>${per.map(([, m]) => `<td>${m[t] === undefined ? "—" : fmt(m[t])}</td>`).join("")}</tr>`).join("")}</table>`;
  }

  function rolling(xs, ys, w) {
    const out = [];
    let sum = 0, q = [];
    for (let i = 0; i < xs.length; i++) {
      const v = ys[i];
      if (v === null || v === undefined) continue;
      q.push(v); sum += v;
      if (q.length > w) sum -= q.shift();
      out.push([xs[i], sum / q.length]);
    }
    return out;
  }

  function renderEpisodes() {
    const root = $("#episodes-content");
    const d = focusedDetail();
    if (!d) { root.innerHTML = `<div class="empty">选择左侧的运行</div>`; return; }
    const ep = d.episodes || {};
    if (!(ep.timesteps || []).length) {
      root.innerHTML = `<div class="panel empty">这个运行没有 episodes.csv（旧版训练脚本，或还没有回合结束）。</div>`;
      return;
    }
    root.innerHTML = `<p class="hint">每个点是一个训练回合（带探索噪声）。绿色 = 到达目标，红色 = 超时未到达。滑动平均窗口 50 个回合。</p>
      <div class="grid charts" id="ep-grid"></div>
      <h3 class="section-title">最近 30 个训练回合（${esc(d.summary.name)}）</h3>
      <div class="panel table-wrap" id="ep-table"></div>`;
    const grid = $("#ep-grid");
    const xs = ep.timesteps;
    chartPanel(grid, "飞行时间 (s) — 每个训练回合", {
      series: [{ name: d.summary.name, color: colorOf(d.summary.name), scatter: true, hideLegend: true,
        points: xs.map((x, i) => [x, ep.flight_time[i], ep.waypoint_reached[i] ? "#3ecf8e" : "#ff6b6b"]) }],
    });
    const names = shown();
    const roll = (key, scale = 1) => names.map((n) => {
      const e = (state.details[n] || {}).episodes || {};
      return { name: n, color: colorOf(n), points: rolling(e.timesteps || [], (e[key] || []).map((v) => v === null ? null : v * scale), 50) };
    });
    chartPanel(grid, "训练回合到达率 (%)，滑动 50", { series: roll("waypoint_reached", 100), yMin: 0, yMax: 100 });
    const bl = baselineFor(state.focused);
    chartPanel(grid, "训练回合冲突时间 (s)，滑动 50", { series: roll("intrusion_time"),
      hLines: bl ? [{ y: bl.means.intrusion_time, label: `中性基线 seed ${bl.seed}` }] : [] });
    chartPanel(grid, "训练回合限制区时间 (s)，滑动 50", { series: roll("time_in_restricted_area"),
      hLines: bl ? [{ y: bl.means.time_in_restricted_area, label: `中性基线 seed ${bl.seed}` }] : [] });

    const n = xs.length, start = Math.max(0, n - 30);
    const cols = ["timesteps", "ep_return", "waypoint_reached", "flight_time", "intrusion_time", "time_in_restricted_area", "time_outside_sector"];
    const labels = ["步数", "回报", "到达", "飞行时间", "冲突时间", "限制区时间", "扇区外时间"];
    let rows = "";
    for (let i = n - 1; i >= start; i--) {
      rows += `<tr>${cols.map((c) => {
        const v = (ep[c] || [])[i];
        if (c === "waypoint_reached") return `<td class="${v ? "good" : "bad"}">${v ? "✓" : "✗"}</td>`;
        return `<td>${fmt(v)}</td>`;
      }).join("")}</tr>`;
    }
    $("#ep-table").innerHTML = `<table class="data"><tr>${labels.map((l) => `<th>${l}</th>`).join("")}</tr>${rows}</table>`;
  }

  // ---------------------------------------------------------------- config
  function flatConfig(cfg) {
    if (!cfg) return {};
    const out = { algo: cfg.algo, timesteps: cfg.timesteps };
    for (const [k, v] of Object.entries(cfg.mdp || {})) out[`mdp.${k}`] = v;
    for (const [k, v] of Object.entries(cfg.hyperparams || {})) out[`hp.${k}`] = Array.isArray(v) ? v.join(",") : v;
    for (const [k, v] of Object.entries(cfg.eval || {})) out[`eval.${k}`] = v;
    return out;
  }

  function renderConfig() {
    const root = $("#config-content");
    const names = shown();
    const flats = names.map((n) => [n, flatConfig((state.details[n] || {}).config)]);
    const keys = [...new Set(flats.flatMap(([, f]) => Object.keys(f)))];
    const defaults = state.tunables ? state.tunables.defaults : {};
    const diffTable = keys.length ? `<table class="data"><tr><th>参数</th><th>默认</th>${flats.map(([n]) => `<th style="color:${colorOf(n)}">${esc(n)}</th>`).join("")}</tr>
      ${keys.map((k) => {
        const vals = flats.map(([, f]) => f[k]);
        const differs = new Set(vals.map((v) => JSON.stringify(v))).size > 1;
        const def = k.startsWith("mdp.") ? defaults[k.slice(4)] : undefined;
        return `<tr class="${differs ? "diff" : ""}"><td class="mono" title="${esc(TUNABLE_HELP[k.slice(4)] || "")}">${esc(k)}</td><td class="muted">${def === undefined ? "" : esc(def)}</td>${vals.map((v) => `<td class="${def !== undefined && v !== undefined && String(v) !== String(def) ? "warn" : ""}">${v === undefined ? "—" : esc(v)}</td>`).join("")}</tr>`;
      }).join("")}</table>` : `<div class="empty">所选运行没有 config.json（旧版训练脚本）。</div>`;

    const focusCfg = (focusedDetail() || {}).config || {};
    const mdp = { ...defaults, ...(focusCfg.mdp || {}) };
    const hp = focusCfg.hyperparams || {};
    const algo = focusCfg.algo || "sac";
    const tunKeys = state.tunables ? state.tunables.keys : [];
    root.innerHTML = `
      <p class="hint">表格对比所选运行的全部参数；黄色底 = 运行之间不同，黄色字 = 与 config.py 默认值不同。</p>
      <div class="panel table-wrap" style="max-height:520px">${diffTable}</div>
      <h3 class="section-title">下一次实验的命令生成器（以聚焦运行 ${esc(state.focused || "")} 为起点）</h3>
      <div class="panel">
        <div class="form-grid" id="cmd-form">
          <label><span class="key">--algo</span><select data-arg="algo">${["sac", "ppo"].map((a) => `<option ${a === algo ? "selected" : ""}>${a}</option>`).join("")}</select></label>
          <label><span class="key">--timesteps</span><input data-arg="timesteps" value="${focusCfg.timesteps || 500000}"></label>
          <label><span class="key">--n-envs</span><input data-arg="n-envs" value="${hp.n_envs || 4}"></label>
          <label><span class="key">--run-name</span><input data-arg="run-name" value="${esc((state.focused || "run") + "_next")}"></label>
          <label><span class="key">--lr</span><input data-arg="lr" value="${hp.lr ?? 0.0003}"></label>
          <label><span class="key">--gamma</span><input data-arg="gamma" value="${hp.gamma ?? 0.99}"></label>
          <label><span class="key">--eval-episodes</span><input data-arg="eval-episodes" value="${(focusCfg.eval || {}).episodes || 20}"></label>
          <label><span class="key">--seed</span><input data-arg="seed" value="${hp.seed ?? 0}"></label>
        </div>
        <h3 class="section-title">MDP 参数（与默认值不同的会生成 --set）</h3>
        <div class="form-grid" id="mdp-form">
          ${tunKeys.map((k) => k === "ACTION_MODE"
            ? `<label title="${esc(TUNABLE_HELP[k] || "")}"><span class="key">${k}</span><select data-key="${k}">${state.tunables.action_modes.map((m) => `<option ${m === mdp[k] ? "selected" : ""}>${m}</option>`).join("")}</select><span>${esc(TUNABLE_HELP[k] || "")}</span></label>`
            : `<label title="${esc(TUNABLE_HELP[k] || "")}"><span class="key">${k}</span><input data-key="${k}" value="${esc(mdp[k])}"><span>${esc(TUNABLE_HELP[k] || "")}</span></label>`).join("")}
        </div>
        <pre class="cmd" id="cmd-out"></pre>
        <button class="btn primary" id="copy-cmd">复制训练命令</button>
        <h3 class="section-title">官方评估（seed 42，1000 回合）</h3>
        <pre class="cmd" id="eval-cmd">python -m scripts.evaluate_competition --env sa --run-dir competitor/results/${esc(state.focused || "<run>")}</pre>
        <button class="btn" id="copy-eval">复制评估命令</button>
      </div>`;
    const update = () => {
      const parts = ["SDL_VIDEODRIVER=dummy python -m scripts.train_sa_sac"];
      root.querySelectorAll("#cmd-form [data-arg]").forEach((inp) => { if (inp.value !== "") parts.push(`--${inp.dataset.arg} ${inp.value}`); });
      root.querySelectorAll("#mdp-form [data-key]").forEach((inp) => {
        const k = inp.dataset.key, def = defaults[k];
        const changed = String(inp.value) !== String(def) && Number(inp.value) !== Number(def);
        inp.classList.toggle("changed", changed);
        if (changed) parts.push(k === "ACTION_MODE" ? `--action-mode ${inp.value}` : `--set ${k}=${inp.value}`);
      });
      $("#cmd-out").textContent = parts.join(" \\\n    ");
    };
    root.querySelectorAll("input, select").forEach((i) => i.addEventListener("input", update));
    update();
    const copy = (id, btn) => {
      const text = $(id).textContent.replace(/ \\\n\s+/g, " ");
      navigator.clipboard.writeText(text).then(() => { btn.textContent = "已复制 ✓"; setTimeout(() => (btn.textContent = btn.id === "copy-cmd" ? "复制训练命令" : "复制评估命令"), 1500); });
    };
    $("#copy-cmd").addEventListener("click", (e) => copy("#cmd-out", e.target));
    $("#copy-eval").addEventListener("click", (e) => copy("#eval-cmd", e.target));
  }

  // ------------------------------------------------------------------ init
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
    state.tab = b.dataset.tab; persist(); renderTab();
  }));
  $("#refresh-btn").addEventListener("click", refresh);
  $("#smooth").addEventListener("input", (e) => {
    state.smooth = Number(e.target.value); persist();
    $("#smooth-val").textContent = state.smooth.toFixed(2);
    renderTab();
  });
  let timer = null;
  const schedule = () => {
    clearInterval(timer);
    if ($("#auto-refresh").checked) timer = setInterval(() => { if (state.tab !== "config") refresh(); }, 10000);
  };
  $("#auto-refresh").addEventListener("change", schedule);
  window.addEventListener("resize", () => renderTab());
  schedule();
  refresh();
})();
