"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const SVG_NS = "http://www.w3.org/2000/svg";
const CENTER_SIZES = [7947, 2531, 2156, 1448, 525, 281];
const TERMINAL_STATES = new Set(["completed", "done", "failed", "cancelled"]);
const ACTIVE_STATES = new Set(["preparing", "preflight", "waiting", "training", "validating", "stopping"]);

const ui = {
  mode: $("#mode"), strategy: $("#strategy"), device: $("#device"), rounds: $("#rounds"),
  localEpochs: $("#local-epochs"), maxBatches: $("#max-batches"), epsilon: $("#epsilon"),
  delta: $("#delta"), clipNorm: $("#clip-norm"), warmStart: $("#warm-start"),
  freezeEdge: $("#freeze-edge"), start: $("#start-run"), cancel: $("#cancel-run"),
  runtime: $("#runtime-state"), form: $("#run-form"), badge: $("#run-badge"),
  truth: $("#truth-line"), progress: $("#phase-progress"), chart: $("#live-chart"),
  events: $("#event-stream"), budgetBars: $("#budget-bars"), toast: $("#toast"),
  error: $("#run-error"),
};

const state = {
  evidence: null,
  runtime: null,
  token: null,
  history: [],
  clients: new Map(),
  events: [],
  selectedClient: 0,
  running: false,
  polling: null,
  replayTimer: null,
  planTimer: null,
  planSequence: 0,
  currentPlan: null,
  currentRunId: null,
  centerSizes: [...CENTER_SIZES],
  lastMode: null,
  resumeActive: false,
};

const fallbackEvidence = {
  method_comparison: { methods: [] },
  heterogeneity: { class_names: [], centers: [] },
  privacy_utility: { points: [] },
  live_replay: { history: [], timings: [] },
};

function svgNode(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== null) node.textContent = text;
  return node;
}

function htmlNode(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = text;
  return node;
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatScore(value, digits = 3) {
  const numeric = finite(value);
  return numeric === null ? "—" : numeric.toFixed(digits);
}

function formatDelta(value) {
  const exponent = Math.round(Math.log10(Number(value)));
  const superscripts = { "-4": "⁻⁴", "-5": "⁻⁵", "-6": "⁻⁶" };
  return `10${superscripts[String(exponent)] || `^${exponent}`}`;
}

function showToast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => ui.toast.classList.remove("show"), 3200);
}

function clearError() {
  ui.error.textContent = "";
  ui.error.hidden = true;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
  showToast(message);
}

function apiUrl(path) {
  return new URL(path.replace(/^\//, ""), `${window.location.origin}/`).toString();
}

async function loadEvidence() {
  try {
    const response = await fetch(new URL("./evidence.json", document.baseURI), { cache: "no-store" });
    if (!response.ok) throw new Error("Evidence bundle unavailable");
    state.evidence = await response.json();
  } catch (_error) {
    state.evidence = fallbackEvidence;
    showToast("The evidence bundle could not be loaded.");
  }
  renderEvidence();
}

async function detectRuntime() {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 1200);
  try {
    const response = await fetch(apiUrl("/api/v1/health"), {
      cache: "no-store",
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error("Runtime unavailable");
    const data = await response.json();
    if (data.runtime !== "local" || !data.connected || typeof data.token !== "string") {
      throw new Error("Unexpected runtime contract");
    }
    state.runtime = data;
    state.token = data.token;
    state.centerSizes = Array.isArray(data.dataset?.center_sizes) && data.dataset.center_sizes.length === 6
      ? data.dataset.center_sizes.map(Number)
      : [...CENTER_SIZES];
    $$(".silo").forEach((node) => {
      const id = Number(node.dataset.client);
      node.querySelector("small").textContent = `${state.centerSizes[id].toLocaleString()} train images`;
    });
    ui.runtime.classList.add("connected");
    const kind = data.dataset?.kind === "fixture" ? "Fixture partitions indexed" : data.dataset?.kind === "fed_isic_sized" ? "Fed-ISIC-sized partitions indexed" : "Folder partitions indexed";
    ui.runtime.querySelector("strong").textContent = kind;
    ui.runtime.querySelector("small").textContent = `${data.dataset?.training_images ?? 0} training images · ${data.dataset?.centers ?? 0} centers`;
    ui.mode.value = "experiment";
    if (data.capabilities?.cuda) ui.device.value = "cuda";
    state.resumeActive = Boolean(data.active_run);
  } catch (_error) {
    state.runtime = null;
    state.token = null;
    state.resumeActive = false;
    state.centerSizes = [...CENTER_SIZES];
    ui.runtime.classList.remove("connected");
    ui.runtime.querySelector("strong").textContent = "Hosted evidence mode";
    ui.runtime.querySelector("small").textContent = "Connect the local runner for genuine training";
    ui.mode.value = "replay";
  } finally {
    window.clearTimeout(timeout);
    syncControls();
  }
}

function modeCopy(mode) {
  if (mode === "experiment") return "All six partitions train sequentially on one host. DP-SGD uses full logical-batch schedules.";
  if (mode === "networked") return "Two genuine Flower clients plus a coordinator run locally. FedAvg only; no DP or physical Pi.";
  return "Recorded laptop + Raspberry Pi measurements. No training is running now.";
}

function syncControls() {
  const mode = ui.mode.value;
  const modeChanged = state.lastMode !== mode;
  const replay = mode === "replay";
  const networked = mode === "networked";
  const fixture = state.runtime?.dataset?.kind === "fixture";
  const modelLabel = replay || !state.runtime
    ? "ResNet-18"
    : state.runtime.training_profile?.model || "Configured model";
  $("#coordinator-model").textContent = modelLabel;
  $("#round-model-name").textContent = modelLabel;
  if ((replay || networked) && ui.strategy.value !== "fedavg") ui.strategy.value = "fedavg";
  [...ui.strategy.options].forEach((option) => {
    option.disabled = (replay || networked) && option.value !== "fedavg";
  });

  const privateRun = mode === "experiment" && ui.strategy.value === "dp_fedavg";
  const trainingInputs = [ui.mode, ui.strategy, ui.device, ui.rounds, ui.localEpochs, ui.maxBatches, ui.warmStart, ui.freezeEdge];
  trainingInputs.forEach((input) => { input.disabled = state.running; });
  if (!state.running) {
    ui.device.disabled = replay;
    ui.rounds.disabled = replay;
    ui.localEpochs.disabled = replay;
    ui.maxBatches.disabled = replay || privateRun;
    ui.warmStart.disabled = replay || fixture;
  }
  if (fixture) ui.warmStart.checked = false;
  ui.warmStart.closest("label").querySelector("span").textContent = fixture
    ? "Warm-start unavailable for fixture"
    : "Warm-start from fixed checkpoint";
  [ui.epsilon, ui.delta, ui.clipNorm].forEach((input) => { input.disabled = state.running || !privateRun; });
  $("#privacy-controls").classList.toggle("disabled", !privateRun);
  $("#hardware-controls").hidden = !networked;
  $("#rounds-output").value = ui.rounds.value;
  $("#epsilon-output").value = Number(ui.epsilon.value).toFixed(1);
  $("#mode-help").textContent = modeCopy(mode);

  const connected = Boolean(state.runtime && state.token);
  if (replay) ui.start.textContent = "Replay measured run";
  else if (!connected) ui.start.textContent = "Local runtime required";
  else if (state.runtime.dataset?.kind === "fixture") ui.start.textContent = "Start fixture training";
  else ui.start.textContent = "Start real training";
  ui.start.disabled = state.running;
  ui.cancel.disabled = !state.running || replay;
  $("#privacy-availability").textContent = privateRun ? "all planned logical batches" : "DP experiment only";
  $("#control-note").textContent = privateRun
    ? "DP mode executes every planned logical batch for each Poisson-sampled local epoch so accounting matches the optimizer steps. Expect a longer run."
    : connected
      ? "Inputs are validated server-side; one supervised training process can run at a time."
      : "A hosted page cannot access the licensed dataset or local GPU. Start the repository’s local runner to enable real training.";

  setTopology(mode, modeChanged && !state.running);
  if (modeChanged && !state.running) resetModeView(mode);
  else if (!state.running && !state.history.length) {
    renderMetrics(null, 0, Number(ui.rounds.value), { enabled: privateRun });
    setProgress(0, Number(ui.rounds.value), "Ready", modeCopy(mode));
  }
  state.lastMode = mode;
  schedulePrivacyPlan(privateRun);
}

function setTopology(mode, resetStatuses = false) {
  const activeIds = mode === "experiment" ? [0, 1, 2, 3, 4, 5] : [0, 5];
  $$(".silo").forEach((node) => {
    const id = Number(node.dataset.client);
    if (id === 5) {
      node.querySelector("strong").textContent = mode === "replay" ? "Center 5 · Pi" : mode === "networked" ? "Center 5 · edge profile" : "Center 5";
      node.classList.toggle("pi", mode !== "experiment");
    }
    node.classList.toggle("active", activeIds.includes(id));
    if (resetStatuses) {
      node.classList.remove("training", "complete");
      node.querySelector("em").textContent = activeIds.includes(id) ? "waiting" : "not participating";
    }
  });
  $$(".connections line").forEach((line) => line.classList.toggle("active", state.running && activeIds.includes(Number(line.dataset.client))));
}

// Integer-order Rényi-DP accountant, matching src/fl_med/privacy/accounting.py.
const logFactorial = [0];
for (let index = 1; index <= 256; index += 1) logFactorial[index] = logFactorial[index - 1] + Math.log(index);

function logAdd(logX, logY) {
  if (logX === -Infinity) return logY;
  if (logY === -Infinity) return logX;
  const low = Math.min(logX, logY);
  const high = Math.max(logX, logY);
  return Math.log1p(Math.exp(low - high)) + high;
}

function epsilonForBatches(batchesPerEpoch, sigma, rounds, epochs, delta) {
  const q = 1 / batchesPerEpoch;
  const steps = batchesPerEpoch * rounds * epochs;
  let best = Infinity;
  for (let alpha = 2; alpha <= 256; alpha += 1) {
    let singleStepRdp;
    if (q === 1) {
      singleStepRdp = alpha / (2 * sigma * sigma);
    } else {
      let logA = -Infinity;
      for (let i = 0; i <= alpha; i += 1) {
        const logBinom = logFactorial[alpha] - logFactorial[i] - logFactorial[alpha - i];
        const logCoefficient = logBinom + i * Math.log(q) + (alpha - i) * Math.log1p(-q);
        logA = logAdd(logA, logCoefficient + (i * i - i) / (2 * sigma * sigma));
      }
      singleStepRdp = logA / (alpha - 1);
    }
    const rdp = singleStepRdp * steps;
    const converted = rdp + Math.log1p(-1 / alpha) - (Math.log(delta) + Math.log(alpha)) / (alpha - 1);
    if (converted < best) best = converted;
  }
  return best;
}

function calculatePrivacyPlan(target, delta, rounds, epochs, centerSizes = CENTER_SIZES) {
  const batches = centerSizes.map((size) => Math.max(1, Math.ceil(size / 128)));
  const worstBatches = Math.min(...batches);
  let low = .1;
  let high = 32;
  if (epsilonForBatches(worstBatches, high, rounds, epochs, delta) > target) return null;
  for (let index = 0; index < 40; index += 1) {
    const midpoint = (low + high) / 2;
    if (epsilonForBatches(worstBatches, midpoint, rounds, epochs, delta) > target) low = midpoint;
    else high = midpoint;
  }
  const perCenter = batches.map((count) => epsilonForBatches(count, high, rounds, epochs, delta));
  return { sigma: high, perCenter, epsilonMax: Math.max(...perCenter), delta, target, rounds, epochs };
}

function schedulePrivacyPlan(enabled) {
  window.clearTimeout(state.planTimer);
  const sequence = ++state.planSequence;
  if (!enabled) {
    state.currentPlan = null;
    $("#sigma-output").textContent = "σ = —";
    $("#privacy-plan-copy").textContent = "Choose DP-FedAvg to calculate a complete-schedule budget.";
    renderBudget(null);
    return;
  }
  $("#sigma-output").textContent = "calculating…";
  state.planTimer = window.setTimeout(() => {
    const plannerSizes = state.centerSizes;
    const plan = calculatePrivacyPlan(Number(ui.epsilon.value), Number(ui.delta.value), Number(ui.rounds.value), Number(ui.localEpochs.value), plannerSizes);
    if (sequence !== state.planSequence) return;
    state.currentPlan = plan;
    if (!plan) {
      $("#sigma-output").textContent = "outside range";
      return;
    }
    $("#sigma-output").textContent = `σ = ${plan.sigma.toFixed(3)}`;
    $("#privacy-plan-copy").textContent = `Worst-center projection: ε ${plan.epsilonMax.toFixed(2)} across ${plan.rounds} round${plan.rounds === 1 ? "" : "s"}. No accuracy is predicted.`;
    renderBudget(plan);
  }, 80);
}

function renderBudget(plan) {
  ui.budgetBars.replaceChildren();
  if (!plan) {
    $("#budget-summary").textContent = "Choose DP-FedAvg in the lab to calculate the full planned schedule.";
    $("#budget-delta").textContent = `δ = ${formatDelta(Number(ui.delta.value))}`;
    return;
  }
  plan.perCenter.forEach((epsilon, index) => {
    const row = htmlNode("div", "budget-row");
    row.append(htmlNode("span", "", `Center ${index}`));
    const track = htmlNode("div");
    const bar = htmlNode("i");
    bar.style.width = `${Math.max(2, (epsilon / plan.target) * 100)}%`;
    track.append(bar);
    row.append(track, htmlNode("strong", "", epsilon.toFixed(2)));
    ui.budgetBars.append(row);
  });
  $("#budget-delta").textContent = `δ = ${formatDelta(plan.delta)}`;
  $("#budget-summary").textContent = `Target ε ≤ ${plan.target.toFixed(1)} calibrates σ = ${plan.sigma.toFixed(3)}. Smaller centers spend faster under one shared noise multiplier.`;
}

function renderEvidence() {
  renderMethods(state.evidence.method_comparison?.methods || []);
  renderPrivacyEvidence((state.evidence.privacy_utility?.points || []).filter((point) => point.noise_multiplier !== null));
  renderHeatmap(state.evidence.heterogeneity || { class_names: [], centers: [] });
  attachProvenance(".method-card", state.evidence.method_comparison);
  attachProvenance(".historical-card", state.evidence.privacy_utility);
  attachProvenance(".heterogeneity-card", state.evidence.heterogeneity);
}

function attachProvenance(selector, record) {
  const container = $(selector);
  container?.querySelector(".provenance")?.remove();
  if (!container || (!record?.source && !record?.sources)) return;
  const details = htmlNode("details", "provenance");
  details.append(htmlNode("summary", "", "Artifact provenance"));
  const sources = Array.isArray(record.sources) ? record.sources.join(" · ") : record.source;
  details.append(htmlNode("code", "", sources));
  if (record.notes?.length) details.append(htmlNode("p", "", record.notes.join(" ")));
  container.append(details);
}

function renderMethods(methods) {
  const bars = $("#method-bars");
  const body = $("#method-table tbody");
  bars.replaceChildren();
  body.replaceChildren();
  methods.forEach((method) => {
    const row = htmlNode("div", `method-row ${method.id}`);
    row.append(htmlNode("span", "", method.label));
    const track = htmlNode("div");
    const bar = htmlNode("i");
    bar.style.width = `${Math.min(100, (method.balanced_accuracy_best_mean / .5) * 100)}%`;
    track.append(bar);
    row.append(track, htmlNode("strong", "", method.balanced_accuracy_best_mean.toFixed(3)));
    bars.append(row);

    const tr = document.createElement("tr");
    [method.label, method.balanced_accuracy_best_mean.toFixed(3), method.balanced_accuracy_best_std.toFixed(3), method.client_drift_final_mean === null ? "—" : method.client_drift_final_mean.toFixed(2)].forEach((value, index) => {
      const cell = htmlNode(index === 0 ? "th" : "td", "", value);
      if (index === 0) cell.scope = "row";
      tr.append(cell);
    });
    body.append(tr);
  });
}

function renderPrivacyEvidence(points) {
  const body = $("#privacy-table tbody");
  body.replaceChildren();
  points.forEach((point) => {
    const tr = document.createElement("tr");
    [point.noise_multiplier.toFixed(1), point.epsilon_max.toFixed(2), point.balanced_accuracy_final.toFixed(3)].forEach((value, index) => {
      const cell = htmlNode(index === 0 ? "th" : "td", "", value);
      if (index === 0) cell.scope = "row";
      tr.append(cell);
    });
    body.append(tr);
  });

  const svg = $("#privacy-chart");
  const title = svg.querySelector("title")?.textContent || "Historical privacy and utility";
  const description = svg.querySelector("desc")?.textContent || "Historical measured points";
  svg.replaceChildren(
    svgNode("title", { id: "privacy-chart-title" }, title),
    svgNode("desc", { id: "privacy-chart-desc" }, description),
  );
  if (!points.length) return;
  const width = 680, height = 300, left = 54, right = 24, top = 18, bottom = 40;
  const xMin = Math.log10(4), xMax = Math.log10(260), yMin = .12, yMax = .27;
  const x = (value) => left + ((Math.log10(value) - xMin) / (xMax - xMin)) * (width - left - right);
  const y = (value) => top + (1 - (value - yMin) / (yMax - yMin)) * (height - top - bottom);
  [.125, .15, .20, .25].forEach((tick) => {
    svg.append(svgNode("line", { x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: "chart-grid" }));
    svg.append(svgNode("text", { x: left - 9, y: y(tick) + 4, "text-anchor": "end", class: "chart-axis" }, tick.toFixed(3)));
  });
  [5, 10, 25, 50, 100, 250].forEach((tick) => svg.append(svgNode("text", { x: x(tick), y: height - 14, "text-anchor": "middle", class: "chart-axis" }, String(tick))));
  svg.append(svgNode("text", { x: (left + width - right) / 2, y: height - 1, "text-anchor": "middle", class: "chart-axis" }, "Historical ε max (log scale; lower is tighter)"));
  points.forEach((point) => {
    svg.append(svgNode("circle", { cx: x(point.epsilon_max), cy: y(point.balanced_accuracy_final), r: 7, class: "scatter-dot" }));
    svg.append(svgNode("text", { x: x(point.epsilon_max) + 10, y: y(point.balanced_accuracy_final) - 9, class: "scatter-label" }, `σ ${point.noise_multiplier}`));
  });
}

function renderHeatmap(data) {
  const heatmap = $("#class-heatmap");
  heatmap.parentElement.querySelector(".heatmap-table")?.remove();
  heatmap.replaceChildren(htmlNode("span", "heat-label", "Center"));
  (data.class_names || []).forEach((name) => heatmap.append(htmlNode("span", "heat-label heat-class", name)));
  (data.centers || []).forEach((center) => {
    heatmap.append(htmlNode("span", "heat-label", `Center ${center.id}`));
    center.class_counts.forEach((count, classIndex) => {
      const share = count / center.samples;
      const cell = htmlNode("span", "heat-cell", `${Math.round(share * 100)}%`);
      cell.style.background = `rgba(67, 214, 228, ${(.04 + share * .85).toFixed(3)})`;
      cell.title = `${data.class_names[classIndex]}: ${count.toLocaleString()} images (${(share * 100).toFixed(1)}%)`;
      heatmap.append(cell);
    });
  });
  const table = htmlNode("table", "heatmap-table sr-only");
  const caption = htmlNode("caption", "", "Class counts and within-center shares for each training partition");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.append(htmlNode("th", "", "Center"));
  (data.class_names || []).forEach((name) => headRow.append(htmlNode("th", "", name)));
  head.append(headRow);
  const body = document.createElement("tbody");
  (data.centers || []).forEach((center) => {
    const row = document.createElement("tr");
    const label = htmlNode("th", "", `Center ${center.id}`); label.scope = "row"; row.append(label);
    center.class_counts.forEach((count) => row.append(htmlNode("td", "", `${count} images, ${((count / center.samples) * 100).toFixed(1)} percent`)));
    body.append(row);
  });
  table.append(caption, head, body);
  heatmap.parentElement.append(table);
}

function drawLiveChart(history) {
  const svg = ui.chart;
  svg.replaceChildren(svgNode("title", { id: "live-chart-title" }, "Balanced accuracy by training round"));
  const valid = history.filter((point) => finite(point.round) !== null && finite(point.balanced_accuracy) !== null);
  const summary = valid.length ? `${valid.length} observed point${valid.length === 1 ? "" : "s"}; latest balanced accuracy ${formatScore(valid.at(-1).balanced_accuracy)}.` : "No active observations yet.";
  svg.append(svgNode("desc", { id: "live-chart-desc" }, summary));
  const width = 680, height = 220, left = 44, right = 15, top = 13, bottom = 30;
  const maxRound = Math.max(1, ...valid.map((point) => Number(point.round)));
  const maxObserved = Math.max(.22, ...valid.map((point) => Number(point.balanced_accuracy)));
  const yMin = .10;
  const yMax = Math.max(.25, Math.ceil((maxObserved + .02) * 20) / 20);
  const x = (value) => left + (Number(value) / maxRound) * (width - left - right);
  const y = (value) => top + (1 - (Number(value) - yMin) / (yMax - yMin)) * (height - top - bottom);
  [yMin, .125, (yMin + yMax) / 2, yMax].forEach((tick) => {
    svg.append(svgNode("line", { x1: left, x2: width - right, y1: y(tick), y2: y(tick), class: "chart-grid" }));
    svg.append(svgNode("text", { x: left - 7, y: y(tick) + 4, "text-anchor": "end", class: "chart-axis" }, tick.toFixed(3)));
  });
  svg.append(svgNode("text", { x: left, y: height - 8, class: "chart-axis" }, "0"));
  svg.append(svgNode("text", { x: width - right, y: height - 8, "text-anchor": "end", class: "chart-axis" }, `round ${maxRound}`));
  if (!valid.length) return;
  const points = valid.map((point) => `${x(point.round)},${y(point.balanced_accuracy)}`).join(" ");
  const gradient = svgNode("linearGradient", { id: "area-gradient", x1: "0", y1: "0", x2: "0", y2: "1" });
  gradient.append(svgNode("stop", { offset: "0", "stop-color": "#43d6e4", "stop-opacity": ".45" }), svgNode("stop", { offset: "1", "stop-color": "#43d6e4", "stop-opacity": "0" }));
  const defs = svgNode("defs"); defs.append(gradient); svg.append(defs);
  const area = `${x(valid[0].round)},${y(yMin)} ${points} ${x(valid.at(-1).round)},${y(yMin)}`;
  svg.append(svgNode("polygon", { points: area, class: "chart-area" }), svgNode("polyline", { points, class: "chart-path" }));
  valid.forEach((point) => svg.append(svgNode("circle", { cx: x(point.round), cy: y(point.balanced_accuracy), r: 4, class: "chart-point" })));
}

function eventLabel(event) {
  const kind = String(event.event || event.kind || "update").replaceAll("_", " ");
  const center = event.client_id === undefined ? "" : ` · center ${event.client_id}`;
  return `${kind}${center}`;
}

function renderEvents(events) {
  state.events = events.slice(-40);
  ui.events.replaceChildren();
  const visible = state.events.slice(-12).reverse();
  if (!visible.length) {
    const item = document.createElement("li"); item.append(htmlNode("time", "", "—"), htmlNode("span", "", "Waiting for a run")); ui.events.append(item);
  }
  visible.forEach((event) => {
    const item = document.createElement("li");
    const marker = event.round === undefined ? `#${event.sequence ?? "—"}` : `R${String(event.round).padStart(2, "0")}`;
    item.append(htmlNode("time", "", marker), htmlNode("span", "", eventLabel(event)));
    ui.events.append(item);
  });
  $("#event-count").textContent = `${state.events.length} event${state.events.length === 1 ? "" : "s"}`;
}

function updateCenterDrawer(id = state.selectedClient) {
  state.selectedClient = id;
  $$(".silo").forEach((node) => node.classList.toggle("selected", Number(node.dataset.client) === id));
  const client = state.clients.get(id) || {};
  $("#center-title").textContent = id === 5 && ui.mode.value === "replay" ? "Center 5 · Raspberry Pi edge" : id === 5 && ui.mode.value === "networked" ? "Center 5 · local edge profile" : `Center ${id}`;
  $("#center-size").textContent = state.centerSizes[id].toLocaleString();
  $("#center-state").textContent = String(client.status || (ui.mode.value === "experiment" || [0, 5].includes(id) ? "Waiting" : "Not participating"));
  const seconds = finite(client.fit_seconds);
  $("#center-time").textContent = seconds === null ? "—" : `${seconds.toFixed(2)} s`;
  const epsilon = finite(client.epsilon);
  $("#center-epsilon").textContent = epsilon === null ? (ui.strategy.value === "dp_fedavg" ? "Projected after start" : "Not enabled") : `ε ${epsilon.toFixed(2)}`;
}

function resetModeView(mode) {
  stopTimers();
  clearError();
  state.running = false;
  const replay = mode === "replay" ? state.evidence?.live_replay : null;
  const first = replay?.history?.[0] || null;
  state.history = first ? [first] : [];
  state.events = [];
  state.clients.clear();
  drawLiveChart(state.history);
  renderEvents([]);
  renderMetrics(first, 0, mode === "replay" ? 8 : Number(ui.rounds.value), {
    enabled: mode === "experiment" && ui.strategy.value === "dp_fedavg",
  });
  setTopology(mode, true);
  $("#coordinator").classList.remove("working");
  $("#coordinator-state").textContent = "global model ready";
  ui.badge.className = `run-badge ${mode === "replay" ? "replay" : ""}`;
  ui.badge.querySelector("strong").textContent = mode === "replay" ? "Recorded replay" : "Ready";
  ui.truth.className = `truth-line ${mode !== "replay" && state.runtime ? "live" : ""}`;
  if (mode === "replay") {
    ui.truth.querySelector("strong").textContent = "RECORDED REPLAY";
    ui.truth.querySelector("span").textContent = "Measured laptop and Raspberry Pi run; no computation is occurring now.";
  } else if (!state.runtime) {
    ui.truth.querySelector("strong").textContent = "LOCAL RUNTIME REQUIRED";
    ui.truth.querySelector("span").textContent = "Run the loopback controller beside the dataset to enable genuine training.";
  } else if (mode === "networked") {
    ui.truth.querySelector("strong").textContent = "LOCAL TWO-CLIENT FLOWER";
    ui.truth.querySelector("span").textContent = "Two client processes plus a coordinator run on this machine; the Raspberry Pi is represented only in measured replay.";
  } else {
    const kind = state.runtime.dataset?.kind;
    ui.truth.querySelector("strong").textContent = kind === "fixture" ? "FIXTURE RUNTIME READY" : kind === "fed_isic_sized" ? "FED-ISIC-SIZED PARTITIONS READY" : "FOLDER DATA RUNTIME READY";
    ui.truth.querySelector("span").textContent = "Six optimization clients will execute sequentially after you start the run.";
  }
  setProgress(0, mode === "replay" ? 8 : Number(ui.rounds.value), "Ready", modeCopy(mode));
  updateCenterDrawer();
}

function renderMetrics(latest, round, total, privacy) {
  $("#metric-round").textContent = `${round ?? 0} / ${total ?? 0}`;
  $("#metric-accuracy").textContent = formatScore(latest?.balanced_accuracy);
  $("#metric-f1").textContent = formatScore(latest?.macro_f1);
  const epsilon = finite(latest?.epsilon_max);
  if (epsilon !== null) $("#metric-privacy").textContent = `ε ${epsilon.toFixed(2)} @ δ ${formatDelta(latest?.delta || privacy?.delta || 1e-5)}`;
  else if (privacy?.enabled) {
    const partial = [...state.clients.values()]
      .map((client) => finite(client.epsilon))
      .filter((value) => value !== null);
    $("#metric-privacy").textContent = partial.length
      ? `Partial round · ε max ${Math.max(...partial).toFixed(2)}`
      : "Planned · no update released";
  }
  else $("#metric-privacy").textContent = "No finite DP claim";
}

function setProgress(round, total, phase, detail) {
  const percent = total > 0 ? Math.max(0, Math.min(100, (round / total) * 100)) : 0;
  ui.progress.style.width = `${percent}%`;
  ui.progress.parentElement.setAttribute("aria-valuenow", String(Math.round(percent)));
  ui.progress.parentElement.setAttribute("aria-valuetext", `${phase}; round ${round} of ${total}`);
  $("#phase-label").textContent = phase;
  $("#phase-detail").textContent = detail;
}

function applyReplayTimings(timings, status = "complete") {
  timings.forEach((timing) => {
    const id = timing.tag === "pi5" ? 5 : 0;
    state.clients.set(id, { status, fit_seconds: timing.fit_seconds, device: timing.device, num_samples: timing.samples });
  });
  $$(".silo").forEach((node) => {
    const id = Number(node.dataset.client);
    const client = state.clients.get(id);
    if (!client) return;
    node.querySelector("em").textContent = client.fit_seconds ? `${client.fit_seconds.toFixed(2)} s` : client.status;
    node.classList.toggle("training", status === "training");
    node.classList.toggle("complete", status === "complete");
  });
  updateCenterDrawer();
}

function stopTimers() {
  window.clearInterval(state.polling); state.polling = null;
  window.clearInterval(state.replayTimer); state.replayTimer = null;
}

function startReplay() {
  stopTimers();
  const replay = state.evidence?.live_replay;
  if (!replay?.history?.length) return showToast("Measured replay data is unavailable.");
  state.running = true;
  state.currentRunId = "recorded-hardware-run";
  state.history = [replay.history[0]];
  state.events = [{ sequence: 1, round: 0, event: "recorded baseline loaded" }];
  ui.badge.className = "run-badge replay";
  ui.badge.querySelector("strong").textContent = "Recorded replay";
  ui.truth.className = "truth-line";
  ui.truth.querySelector("strong").textContent = "RECORDED REPLAY";
  ui.truth.querySelector("span").textContent = "Measured laptop and Raspberry Pi run; no computation is occurring now.";
  $("#coordinator").classList.add("working");
  syncControls();
  drawLiveChart(state.history);
  renderEvents(state.events);
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const finalPoint = replay.history.at(-1);
    state.history = [...replay.history];
    state.events.push({ sequence: 2, round: finalPoint.round, event: "recorded replay complete" });
    applyReplayTimings(replay.timings.filter((item) => item.round === finalPoint.round), "complete");
    drawLiveChart(state.history);
    renderEvents(state.events);
    renderMetrics(finalPoint, finalPoint.round, 8, null);
    setProgress(8, 8, "Replay complete", "Reduced motion: all committed observations are shown without animation");
    state.running = false;
    $("#coordinator").classList.remove("working");
    ui.badge.querySelector("strong").textContent = "Replay complete";
    syncControls();
    return;
  }
  let index = 1;
  const delay = 850;
  state.replayTimer = window.setInterval(() => {
    const point = replay.history[index];
    state.history.push(point);
    state.events.push({ sequence: state.events.length + 1, round: point.round, event: "archived round metrics loaded" });
    applyReplayTimings(replay.timings.filter((item) => item.round === point.round), "complete");
    drawLiveChart(state.history);
    renderEvents(state.events);
    renderMetrics(point, point.round, 8, null);
    setProgress(point.round, 8, `Replaying measured round ${point.round}`, "Metrics and timings are read from experiments/live/history.json");
    index += 1;
    if (index >= replay.history.length) {
      stopTimers();
      state.running = false;
      $("#coordinator").classList.remove("working");
      ui.badge.querySelector("strong").textContent = "Replay complete";
      state.events.push({ sequence: state.events.length + 1, round: 8, event: "recorded replay complete" });
      renderEvents(state.events);
      setProgress(8, 8, "Replay complete", "All displayed values came from the committed hardware-run artifact");
      syncControls();
    }
  }, delay);
}

function runPayload() {
  const payload = {
    mode: ui.mode.value,
    strategy: ui.strategy.value,
    rounds: Number(ui.rounds.value),
    local_epochs: Number(ui.localEpochs.value),
    seed: 42,
    device: ui.device.value,
    freeze_edge: ui.freezeEdge.checked,
    warm_start: ui.warmStart.checked,
  };
  if (ui.strategy.value === "dp_fedavg") {
    payload.target_epsilon = Number(ui.epsilon.value);
    payload.delta = Number(ui.delta.value);
    payload.clip_norm = Number(ui.clipNorm.value);
  } else {
    payload.max_batches = Number(ui.maxBatches.value);
  }
  return payload;
}

async function controllerRequest(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    ...options,
    cache: "no-store",
    headers: { Accept: "application/json", "Content-Type": "application/json", "X-Demo-Token": state.token, ...(options.headers || {}) },
  });
  let body = {};
  try { body = await response.json(); } catch (_error) { /* sanitized below */ }
  if (!response.ok) throw new Error(body.error?.message || `Controller returned ${response.status}`);
  return body;
}

async function startLocalRun() {
  clearError();
  if (!state.runtime || !state.token) {
    showError("Start the local runner from the repository to enable genuine training.");
    $("#methods").scrollIntoView({ behavior: "smooth" });
    return;
  }
  if (ui.strategy.value === "dp_fedavg" && !state.runtime.capabilities?.dp) return showError("Install the privacy extra to run DP-SGD.");
  if (ui.mode.value === "networked" && !state.runtime.capabilities?.networked) return showError("Install the compatible Flower runtime to use networked mode.");
  state.running = true;
  syncControls();
  try {
    const status = await controllerRequest("/api/v1/runs", { method: "POST", body: JSON.stringify(runPayload()) });
    state.currentRunId = status.run_id;
    renderStatus(status);
    beginPolling();
  } catch (error) {
    state.running = false;
    syncControls();
    showError(error.message);
  }
}

function beginPolling() {
  stopTimers();
  const poll = async () => {
    try {
      const status = await controllerRequest("/api/v1/runs/current");
      if (state.currentRunId && status.run_id && status.run_id !== state.currentRunId) return;
      state.currentRunId = status.run_id || state.currentRunId;
      renderStatus(status);
      if (TERMINAL_STATES.has(status.status)) {
        window.clearInterval(state.polling); state.polling = null;
        state.running = false;
        syncControls();
      }
    } catch (_error) {
      window.clearInterval(state.polling); state.polling = null;
      state.running = false;
      ui.badge.className = "run-badge failed";
      ui.badge.querySelector("strong").textContent = "Disconnected";
      setProgress(0, 1, "Local runtime disconnected", "Last validated observations are preserved; no rounds are invented");
      showError("The local runtime disconnected. Last validated observations are preserved.");
      syncControls();
    }
  };
  poll();
  state.polling = window.setInterval(poll, 800);
}

function normalizeClientId(client) {
  if (Number.isInteger(client.client_id)) return client.client_id;
  const match = String(client.tag || "").match(/(?:center-|c)([0-5])$/);
  return match ? Number(match[1]) : null;
}

function renderStatus(status) {
  const active = ACTIVE_STATES.has(status.status);
  state.running = active;
  state.history = (status.history || []).filter((point) => finite(point.round) !== null);
  state.clients.clear();
  (status.clients || []).forEach((client) => {
    const id = normalizeClientId(client);
    if (id !== null) state.clients.set(id, {
      ...client,
      epsilon: client.epsilon ?? status.history?.at(-1)?.epsilon_by_client?.[String(id)],
    });
  });
  const latest = state.history.at(-1);
  renderMetrics(latest, latest?.round ?? 0, status.total_rounds || 0, status.privacy);
  drawLiveChart(state.history);
  renderEvents(status.events || []);
  updateCenterDrawer();
  const completedRounds = status.completed_rounds ?? latest?.round ?? 0;
  const activeRound = status.active_round ?? status.round ?? completedRounds;
  const activeDetail = active ? ` · active round ${activeRound}` : "";
  setProgress(completedRounds, status.total_rounds || 1, String(status.phase || status.status).replaceAll("_", " "), `${status.elapsed_seconds ?? 0} s elapsed${activeDetail} · run ${String(status.run_id || "").slice(-8)}`);
  ui.badge.className = `run-badge ${active ? "live" : status.status === "failed" ? "failed" : ""}`;
  ui.badge.querySelector("strong").textContent = active ? "Live training" : status.status === "completed" || status.status === "done" ? "Run complete" : String(status.status || "idle");
  ui.truth.className = "truth-line live";
  const kind = status.dataset_kind === "fixture" ? "SYNTHETIC FIXTURE — REAL TRAINING" : status.dataset_kind === "fed_isic_sized" ? "FED-ISIC-SIZED FOLDER — REAL TRAINING" : "USER-PROVIDED FOLDER DATA — REAL TRAINING";
  ui.truth.querySelector("strong").textContent = kind;
  ui.truth.querySelector("span").textContent = status.mode === "networked" ? "Two actual Flower clients train locally as center 0 and center 5 under a separate coordinator; no physical Pi is connected." : "Six real optimization clients execute sequentially on this training host.";
  $("#coordinator").classList.toggle("working", active);
  $("#coordinator-state").textContent = String(status.phase || status.status).replaceAll("_", " ");
  $$(".silo").forEach((node) => {
    const id = Number(node.dataset.client);
    const client = state.clients.get(id);
    node.classList.remove("training", "complete");
    if (client) {
      const clientStatus = String(client.status || (active ? "training" : "complete"));
      node.classList.add(clientStatus === "training" ? "training" : "complete");
      node.querySelector("em").textContent = finite(client.fit_seconds) === null ? clientStatus : `${Number(client.fit_seconds).toFixed(2)} s`;
    }
  });
  $$(".connections line").forEach((line) => line.classList.toggle("active", active && (status.mode === "experiment" || [0, 5].includes(Number(line.dataset.client)))));
}

async function cancelRun() {
  if (!state.running || !state.token) return;
  ui.cancel.disabled = true;
  try {
    const status = await controllerRequest("/api/v1/runs/current/cancel", { method: "POST", body: "{}" });
    renderStatus(status);
  } catch (error) {
    ui.cancel.disabled = false;
    showError(error.message);
  }
}

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (ui.mode.value === "replay") startReplay();
  else startLocalRun();
});
ui.cancel.addEventListener("click", cancelRun);
[ui.mode, ui.strategy, ui.device, ui.localEpochs, ui.maxBatches, ui.delta, ui.clipNorm].forEach((control) => control.addEventListener("change", syncControls));
[ui.rounds, ui.epsilon].forEach((control) => control.addEventListener("input", syncControls));
$$('.silo').forEach((node) => node.addEventListener("click", () => updateCenterDrawer(Number(node.dataset.client))));
$("#copy-command").addEventListener("click", async () => {
  const command = "DATA_ROOT=/absolute/path/to/fed_isic2019/raw make demo-site";
  try { await navigator.clipboard.writeText(command); showToast("Local training command copied."); }
  catch (_error) { showToast(command); }
});

async function initialize() {
  syncControls();
  await Promise.all([loadEvidence(), detectRuntime()]);
  if (state.resumeActive) {
    try {
      const status = await controllerRequest("/api/v1/runs/current");
      ui.mode.value = status.mode || "experiment";
      ui.strategy.value = status.strategy || "fedavg";
      state.lastMode = ui.mode.value;
      state.currentRunId = status.run_id;
      renderStatus(status);
      beginPolling();
      syncControls();
      return;
    } catch (_error) {
      state.resumeActive = false;
    }
  }
  resetModeView(ui.mode.value);
  syncControls();
}

initialize();
