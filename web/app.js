// Corner Chance Planner front end: the planner, insights and playbook, with data from app.py.

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clone = (value) => JSON.parse(JSON.stringify(value));
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
// how a shape reads after "against", e.g. "a deep-line defence"
const shapePhrase = (id) => META?.shapes?.find((s) => s.id === id)?.phrase ?? `a ${id} defence`;

const store = {
  get(key) {
    try {
      return JSON.parse(localStorage.getItem(key));
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // storage blocked (e.g. private mode): nothing to save
    }
  },
};

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

const api = {
  meta: () => request("/api/meta"),
  evaluate: (setup) =>
    request("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(setup),
    }),
  playbook: (params) => request(`/api/playbook?${new URLSearchParams(params)}`),
  insights: () => request("/api/insights"),
};

const ICONS = {
  real: '<svg viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3.5" width="14" height="13" rx="2"/><path d="M3 8h14M7.5 3.5v13"/></svg>',
  model: '<svg viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 16l4-5 3 3 6-8"/><circle cx="16" cy="6" r="1.2" fill="currentColor"/></svg>',
  check: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  minus: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 8h8" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
  alert: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M8 4.8v3.6M8 11v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  up: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 16V4M4.5 9.5L10 4l5.5 5.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  down: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 4v12M4.5 10.5L10 16l5.5-5.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  equal: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 7.5h11M4.5 12.5h11" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
  target: '<svg viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.9"><circle cx="10" cy="10" r="7"/><circle cx="10" cy="10" r="3.2"/></svg>',
  warn: '<svg viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"><path d="M10 3l8 14H2z"/><path d="M10 8.2v3.6M10 14.4v.1"/></svg>',
  chevron: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  moon: '<svg viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M16.5 12.2A6.8 6.8 0 0 1 7.8 3.5a6.8 6.8 0 1 0 8.7 8.7z"/></svg>',
  sun: '<svg viewBox="0 0 20 20" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="10" cy="10" r="3.6"/><path d="M10 2v1.8M10 16.2V18M2 10h1.8M16.2 10H18M4.3 4.3l1.3 1.3M14.4 14.4l1.3 1.3M4.3 15.7l1.3-1.3M14.4 5.6l1.3-1.3"/></svg>',
};


// Pitch drawing: corner from the right, goal at the top, 8.5 units per yard across,
// slightly stretched in depth so the six-yard box has room.

const VIEWBOX = "96 0 604 424";
const GK = { x: 340, y: 46 };
const BALL_START = { x: 672, y: 38 };
const PADS = {
  n_att_far: { x: 166, y: 40, w: 116, h: 104 },
  n_blockers: { x: 290, y: 34, w: 100, h: 66 },
  n_att_near: { x: 398, y: 40, w: 116, h: 104 },
  n_att_central: { x: 290, y: 106, w: 100, h: 64 },
  n_edge_of_box: { x: 262, y: 178, w: 156, h: 66 },
  n_rest_defence: { x: 140, y: 352, w: 400, h: 58 },
};
const BLOCKER_SPOTS = [
  { x: 318, y: 64 },
  { x: 362, y: 64 },
  { x: 298, y: 50 },
  { x: 382, y: 50 },
  { x: 340, y: 80 },
];
// short zone names for the phone-sized pitch
const SHORT_LABELS = {
  n_att_near: "Near",
  n_att_central: "Middle",
  n_att_far: "Far",
  n_blockers: "Keeper",
  n_edge_of_box: "Spot",
  n_rest_defence: "Outside",
};
const AIM_ZONE = { near: "n_att_near", central: "n_att_central", far: "n_att_far" };
const AIM_POINT = { near: { x: 456, y: 86 }, central: { x: 340, y: 128 }, far: { x: 224, y: 86 } };

function dotPositions(zoneId, n, compact) {
  if (zoneId === "n_blockers") return BLOCKER_SPOTS.slice(0, n);
  const pad = PADS[zoneId];
  const spacing = zoneId === "n_rest_defence" ? 30 : compact ? 24 : 22;
  const areaTop = pad.y + 6;
  const areaHeight = pad.h - (compact ? 8 : 22);
  const perRow = Math.max(1, Math.floor((pad.w - 8) / spacing));
  const rows = Math.ceil(n / perRow);
  const firstY = areaTop + areaHeight / 2 - ((rows - 1) * spacing) / 2;
  const out = [];
  for (let row = 0; row < rows; row++) {
    const inRow = Math.min(perRow, n - row * perRow);
    const firstX = pad.x + pad.w / 2 - ((inRow - 1) * spacing) / 2;
    for (let col = 0; col < inRow; col++) out.push({ x: firstX + col * spacing, y: firstY + row * spacing });
  }
  return out;
}

function ballPath(delivery, zone) {
  const start = BALL_START;
  const end = AIM_POINT[zone] || AIM_POINT.central;
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  // direction pointing away from the goal line
  let nx = -dy / length;
  let ny = dx / length;
  if (ny < 0) {
    nx = -nx;
    ny = -ny;
  }
  // an inswinger bends away then curls in; an outswinger does the opposite
  const bend = delivery === "inswing" ? 0.2 * length : delivery === "outswing" ? -0.2 * length : 0;
  const cx = (start.x + end.x) / 2 + nx * bend;
  const cy = (start.y + end.y) / 2 + ny * bend;
  return { d: `M ${start.x} ${start.y} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${end.x} ${end.y}`, end };
}

function pitchInner(setup, { compact = false, highlight = null, enter = null } = {}) {
  const slots = META.slots;
  const parts = [];

  parts.push(`<rect x="-14" y="0" width="708" height="424" fill="var(--pitch)"/>`);
  for (let y = 0, i = 0; y < 424; y += 53, i++) {
    if (i % 2) parts.push(`<rect x="-14" y="${y}" width="708" height="53" fill="var(--pitch-stripe)"/>`);
  }

  parts.push(`
    <rect class="net" x="306" y="14" width="68" height="16" rx="2"/>
    <g class="lines">
      <path d="M 0 424 V 30 H 680 V 424"/>
      <rect x="153" y="30" width="374" height="252"/>
      <rect x="255" y="30" width="170" height="84"/>
      <path d="M 272 282 A 85 140 0 0 0 408 282"/>
      <path d="M 668 30 A 12 12 0 0 0 680 42"/>
    </g>
    <circle class="spot" cx="340" cy="198" r="3"/>`);

  for (const slot of slots) {
    const pad = PADS[slot.id];
    const n = setup.players[slot.id] ?? 0;
    const classes = ["zone"];
    if (AIM_ZONE[setup.zone] === slot.id) classes.push("aim");
    if (highlight === slot.id) classes.push("hl");
    parts.push(`<g class="${classes.join(" ")}" data-zone="${slot.id}">`);
    parts.push(`<rect class="pad" x="${pad.x}" y="${pad.y}" width="${pad.w}" height="${pad.h}" rx="10"/>`);
    if (!compact) {
      const x = pad.x + pad.w / 2;
      const labelY = slot.id === "n_blockers" ? pad.y + pad.h - 5 : pad.y + pad.h - 8;
      parts.push(
        `<text class="zone-label full" x="${x}" y="${labelY}" text-anchor="middle">${esc(slot.label.toUpperCase())}<tspan class="zone-count"> · ${n}</tspan></text>`,
        `<text class="zone-label short" x="${x}" y="${labelY - 1}" text-anchor="middle">${esc((SHORT_LABELS[slot.id] ?? slot.label).toUpperCase())}</text>`,
      );
    }
    parts.push("</g>");
  }

  const path = ballPath(setup.delivery, setup.zone);
  parts.push(`<path class="ball-path" d="${path.d}"/>`);
  parts.push(`<circle class="target" cx="${path.end.x}" cy="${path.end.y}" r="10"/><circle class="target-dot" cx="${path.end.x}" cy="${path.end.y}" r="2.5"/>`);

  parts.push(`<g class="gk"><circle cx="${GK.x}" cy="${GK.y}" r="9"/><text x="${GK.x}" y="${GK.y + 2.7}" text-anchor="middle">GK</text></g>`);

  const radius = compact ? 9.5 : 8;
  for (const slot of slots) {
    const n = setup.players[slot.id] ?? 0;
    dotPositions(slot.id, n, compact).forEach((p, i) => {
      const isNew = enter === slot.id && i === n - 1;
      parts.push(
        `<g class="player${isNew ? " enter" : ""}" data-zone="${slot.id}"><circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${radius}"/>` +
          (compact ? "" : `<path class="minus" d="M ${(p.x - 3.5).toFixed(1)} ${p.y.toFixed(1)} h 7"/>`) +
          "</g>",
      );
    });
  }

  parts.push(`<path class="flag" d="M 680 30 V 8 L 694 13 L 680 18 Z"/><path d="M 680 30 V 8" stroke="#fff" stroke-width="2"/>`);
  parts.push(`<circle class="ball" cx="${BALL_START.x}" cy="${BALL_START.y}" r="5"/>`);
  return parts.join("");
}

function setupSummary(setup) {
  const players = META.slots
    .filter((s) => setup.players[s.id] > 0)
    .map((s) => `${setup.players[s.id]} ${s.label.toLowerCase()}`)
    .join(", ");
  const delivery = META.deliveries.find((d) => d.id === setup.delivery)?.name ?? setup.delivery;
  const zone = META.zones.find((z) => z.id === setup.zone)?.name ?? setup.zone;
  return `${delivery} aimed at ${zone.toLowerCase()}. Players: ${players}.`;
}

// State

let META = null;
let SLOT_BY_ID = {};

const state = {
  view: "planner",
  setup: null,
  undo: [],
  status: "idle", // idle | loading | ready | invalid | error
  result: null,
  error: "",
  requestId: 0,
  markers: {},
  hoverZone: null,
  justAdded: null,
  numbersOpen: false,
  sourceTab: null, // "real" | "model"; null follows the source the headline uses (remembered per viewer)
  insights: null,
  insightsLoading: false,
  pb: { shape: "All", sort: "chance", includeOther: false, minUsed: 10, limit: 9, data: null, loading: false, requestId: 0 },
};

const controlsEl = $("#controls");
const pitchEl = $("#pitch");
const countEl = $("#count-pill");
const resultsEl = $("#results");
const winsEl = $("#wins");
const mobileBar = $("#mobile-bar");
const toastEl = $("#toast");

const totalPlayers = (setup = state.setup) => META.slots.reduce((sum, s) => sum + (setup.players[s.id] ?? 0), 0);

function validSetup(setup) {
  if (!setup || typeof setup !== "object" || !setup.players) return false;
  const shapeIds = [...META.shapes.map((s) => s.id), META.all_shapes];
  if (!shapeIds.includes(setup.shape)) return false;
  if (!META.deliveries.some((d) => d.id === setup.delivery)) return false;
  if (!META.zones.some((z) => z.id === setup.zone)) return false;
  const countsOk = META.slots.every((s) => {
    const n = setup.players[s.id];
    return Number.isInteger(n) && n >= s.min && n <= s.max;
  });
  return countsOk && totalPlayers(setup) <= META.players_total;
}

// Re-draw a container and keep keyboard focus on the same control.
function renderInto(el, html) {
  const active = document.activeElement;
  const key = active && el.contains(active) ? active.dataset.focusKey : null;
  el.innerHTML = html;
  if (key) el.querySelector(`[data-focus-key="${CSS.escape(key)}"]`)?.focus();
}

// Controls

function segmented(action, options, current, labelledBy) {
  return `<div class="segmented" role="radiogroup" aria-labelledby="${labelledBy}">${options
    .map(
      (o) =>
        `<button type="button" class="seg" role="radio" aria-checked="${o.id === current}" data-action="${action}" data-value="${esc(o.id)}" data-focus-key="${action}-${esc(o.id)}">${esc(o.name)}</button>`,
    )
    .join("")}</div>`;
}

function stepperHTML(slot) {
  const n = state.setup.players[slot.id];
  const full = totalPlayers() >= META.players_total;
  return `
    <div class="stepper-row${state.hoverZone === slot.id ? " hl" : ""}" data-zone="${slot.id}">
      <div class="stepper-info">
        <div class="stepper-label">${esc(slot.label)}</div>
        <div class="stepper-hint">${esc(slot.hint)}</div>
      </div>
      <div class="stepper" role="group" aria-label="${esc(slot.label)}">
        <button type="button" class="step-btn" data-action="count" data-zone="${slot.id}" data-delta="-1" data-focus-key="minus-${slot.id}" aria-label="One fewer player: ${esc(slot.label)}" ${n <= slot.min ? "disabled" : ""}>−</button>
        <output class="step-val" aria-label="${esc(slot.label)}: ${n}">${n}</output>
        <button type="button" class="step-btn" data-action="count" data-zone="${slot.id}" data-delta="1" data-focus-key="plus-${slot.id}" aria-label="One more player: ${esc(slot.label)}" ${n >= slot.max || full ? "disabled" : ""}>+</button>
      </div>
    </div>`;
}

function renderControls() {
  const setup = state.setup;
  const shapeOptions = [
    ...META.shapes,
    { id: META.all_shapes, name: "Compare all", description: "See which defence this corner works best against." },
  ];
  renderInto(
    controlsEl,
    `
    <div class="step">
      <div class="step-head"><span class="step-num">1</span><h2>Their defence</h2></div>
      <div class="option-grid" role="radiogroup" aria-label="Opponent's defence">
        ${shapeOptions
          .map(
            (o) => `
          <button type="button" class="option" role="radio" aria-checked="${o.id === setup.shape}" data-action="shape" data-value="${esc(o.id)}" data-focus-key="shape-${esc(o.id)}">
            <span class="option-name">${esc(o.name)}<span class="radio-dot" aria-hidden="true"></span></span>
            <span class="option-desc">${esc(o.description)}</span>
          </button>`,
          )
          .join("")}
      </div>
      <p class="control-note">${ICONS.alert}<span>This picks which real corners you're compared with. In this data the best setup is the same against every shape.</span></p>
    </div>
    <div class="step">
      <div class="step-head"><span class="step-num">2</span><h2>Delivery</h2></div>
      <div class="field"><span class="field-label" id="lbl-delivery">Type of ball</span>${segmented("delivery", META.deliveries, setup.delivery, "lbl-delivery")}</div>
      <div class="field"><span class="field-label" id="lbl-zone">Aimed at</span>${segmented("zone", META.zones, setup.zone, "lbl-zone")}</div>
    </div>
    <div class="step">
      <div class="step-head">
        <span class="step-num">3</span><h2>Your players</h2>
        <button type="button" class="link-btn" data-action="reset" data-focus-key="reset">Reset</button>
      </div>
      <div class="steppers">${META.slots.map(stepperHTML).join("")}</div>
    </div>`,
  );
}

function renderPitch() {
  pitchEl.innerHTML = pitchInner(state.setup, { highlight: state.hoverZone, enter: state.justAdded });
  pitchEl.setAttribute("aria-label", `Pitch diagram. ${setupSummary(state.setup)}`);
}

function renderCount() {
  const total = totalPlayers();
  const max = META.players_total;
  countEl.className = `count-pill ${total === max ? "ok" : "warn"}`;
  countEl.innerHTML = `<span class="count-dots" aria-hidden="true">${Array.from({ length: max }, (_, i) => `<i class="${i < total ? "on" : ""}"></i>`).join("")}</span><span>${total} / ${max} players</span>`;
}

function setHover(zoneId) {
  if (state.hoverZone === zoneId) return;
  state.hoverZone = zoneId;
  $$(".zone", pitchEl).forEach((g) => g.classList.toggle("hl", g.dataset.zone === zoneId));
  $$(".stepper-row", controlsEl).forEach((row) => row.classList.toggle("hl", row.dataset.zone === zoneId));
}

// Setup changes

function onSetupChange() {
  store.set("apv-setup", state.setup);
  renderControls();
  renderPitch();
  renderCount();
  state.justAdded = null;
  scheduleEvaluate();
}

function setField(field, value) {
  if (state.setup[field] === value) return;
  state.setup[field] = value;
  onSetupChange();
}

function changeCount(zoneId, delta) {
  const slot = SLOT_BY_ID[zoneId];
  const n = state.setup.players[zoneId];
  if (delta > 0) {
    if (totalPlayers() >= META.players_total) return toast("All 10 players are placed. Remove one first.");
    if (n >= slot.max) return toast(`Real teams never put more than ${slot.max} here.`);
  } else if (n <= slot.min) {
    if (slot.min > 0) toast(`Real corners always had at least ${plural(slot.min, "player")} here.`);
    return;
  }
  state.setup.players[zoneId] = n + delta;
  state.justAdded = delta > 0 ? zoneId : null;
  onSetupChange();
}

function pushUndo() {
  state.undo.push(clone(state.setup));
  if (state.undo.length > 25) state.undo.shift();
}

function undo() {
  const previous = state.undo.pop();
  hideToast();
  if (!previous) return;
  state.setup = previous;
  onSetupChange();
}

function resetPlayers() {
  pushUndo();
  state.setup.players = clone(META.defaults.players);
  onSetupChange();
  toast("Players reset to the most common real setup.", true);
}

function applySuggestion(index) {
  const suggestion = state.result?.suggestions?.[index];
  if (!suggestion) return;
  pushUndo();
  // stay on "Compare all" when the suggestion is for the best-matching defence
  state.setup = { ...clone(suggestion.setup), shape: state.setup.shape };
  onSetupChange();
  toast("Change applied.", true);
}

// Rating

let evaluateTimer = null;

function scheduleEvaluate(immediate = false) {
  clearTimeout(evaluateTimer);
  if (totalPlayers() !== META.players_total) {
    state.requestId++; // drop any in-flight response
    state.status = "invalid";
    renderResults();
    return;
  }
  state.status = "loading";
  renderResults();
  evaluateTimer = setTimeout(runEvaluate, immediate ? 0 : 200);
}

async function runEvaluate() {
  const id = ++state.requestId;
  try {
    const result = await api.evaluate(state.setup);
    if (id !== state.requestId) return;
    state.result = result;
    state.status = "ready";
  } catch (err) {
    if (id !== state.requestId) return;
    state.status = "error";
    state.error = err.message;
  }
  renderResults();
}

const pill = (label, tone) => `<span class="pill pill-${tone}">${esc(label)}</span>`;
const pctText = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const loadingDot = () => (state.status === "loading" ? '<span class="loading-dot" role="status" aria-label="Updating"></span>' : "");
const VERDICT_ICON = { better: ICONS.up, worse: ICONS.down, similar: ICONS.equal, unknown: ICONS.equal };
const fraction = (r) => (r?.n ? `${r.k} of ${r.n}` : "");

// One bar: the rate, its 90% range (whisker) and an average tick. size "main" makes
// it thicker, segments stack parts ({value, kind}), dot adds a matching colour dot.
function rateBar({ label, rate, note = "", ref = null, refLabel = "", max = 1, badge = "", size = "", segments = null, dot = "" }) {
  const frac = (x) => Math.max(0, Math.min(1, x / max));
  const pos = (x) => `${(frac(x) * 100).toFixed(2)}%`;
  const r = rate || {};
  const hasRange = r.low != null && r.high != null;
  const range = hasRange ? `likely between ${pctText(r.low)} and ${pctText(r.high)}` : "";
  const tip = `${label}: ${pctText(r.rate)}${note ? ` (${note})` : ""}${range ? `, ${range}` : ""}`;
  let fill = "";
  if (r.rate != null && segments?.length) {
    let left = 0;
    const parts = segments.filter((seg) => seg.value > 0);
    fill = parts
      .map((seg, i) => {
        const html = `<span class="rate-fill rate-seg seg-${seg.kind}${i === parts.length - 1 ? " is-last" : ""}" style="left:${(frac(left) * 100).toFixed(2)}%;width:${(frac(seg.value) * 100).toFixed(2)}%"></span>`;
        left += seg.value;
        return html;
      })
      .join("");
  } else if (r.rate != null) {
    fill = `<span class="rate-fill" style="width:${pos(r.rate)}"></span>`;
  }
  return `
    <div class="rate-row${size ? ` rate-row-${size}` : ""}" title="${esc(tip)}">
      <div class="rate-head"><span class="rate-label">${dot ? `<i class="seg-dot seg-${dot}"></i>` : ""}${esc(label)}${badge ? ` <span class="rate-badge">${badge}</span>` : ""}</span><span class="rate-value">${pctText(r.rate)}${note ? `<small>${esc(note)}</small>` : ""}</span></div>
      <div class="rate-track" role="img" aria-label="${esc(tip)}">
        ${fill}
        ${hasRange ? `<span class="rate-range" style="left:${pos(r.low)};width:${pos(r.high - r.low)}"></span>` : ""}
        ${ref != null ? `<span class="rate-ref" style="left:${pos(ref)}" title="${esc(refLabel)}"></span>` : ""}
      </div>
    </div>`;
}

// good chance = won × chance after winning + lost × chance after losing
function formulaHTML(win, ifWon, ifLost) {
  if (win == null || ifWon == null || ifLost == null) return "";
  const total = win * ifWon + (1 - win) * ifLost;
  return `
    <div class="chain-formula">
      <span class="formula-label">Formula</span>
      <span class="formula-words">first ball won × good chance after winning + first ball lost × good chance after losing</span>
      <span class="formula-nums">${pctText(win)} × ${pctText(ifWon)} + ${pctText(1 - win)} × ${pctText(ifLost)} ≈ ${pctText(total)}</span>
    </div>`;
}

// legend for a split main bar
function segKey(parts) {
  return `<div class="seg-key">${parts
    .filter((p) => p.value > 0)
    .map((p) => `<span><i class="seg-dot seg-${p.kind}"></i>${esc(p.label)} <b>${pctText(p.value)}</b></span>`)
    .join("")}</div>`;
}

function rateLegend(refLabel) {
  return `
  <div class="rate-legend">
    <span><i class="lg-fill"></i>Rate</span>
    <span><i class="lg-whisker"></i>90% range</span>
    <span><i class="lg-ref"></i>${esc(refLabel)}</span>
  </div>`;
}

function rareHTML(rare) {
  if (!rare?.length) return "";
  return `<div class="notice">${ICONS.warn}<div><strong>Real teams rarely did this:</strong> ${esc(rare.join("; "))}. Treat the estimate as a rough guide.</div></div>`;
}

// enough similar real corners to rely on (same rule as the server)
const realIsEnough = (ev) => ev.reliability.tone === "good" || ev.reliability.tone === "fair";
// the sources agree when the model's number is inside the real corners' 90% range
const agrees = (real, model) => real.low != null && real.high != null && model.value != null && model.value >= real.low && model.value <= real.high;

const VERDICT_TEXT = {
  better: "Better than the average",
  worse: "Worse than the average",
  similar: "About the same as the average",
  unknown: "Too little to compare with the average",
};

const STEP_TEXT = {
  win: "Win the first ball",
  if_won: "Good chance if you win it",
  if_lost: "Good chance if you lose it",
};

function levelPill(level) {
  if (level === "tested") return pill("Tested", "good");
  if (level === "rough") return pill("Rough guide", "warn");
  return pill("Weak", "neutral");
}

const LEVEL_TIP = {
  tested: "Beat the average on competitions it had not seen",
  rough: "Only slightly better than the average on unseen competitions",
  none: "No better than quoting the average on unseen competitions",
};

// how each step did on unseen competitions (for the help dialog)
const STEP_PHRASE = { win: "winning the first ball", if_won: "good chances after winning it", if_lost: "good chances after losing it" };

function accuracyText() {
  const steps = META.model.steps || {};
  const keys = Object.keys(STEP_PHRASE).filter((k) => steps[k]);
  const tested = keys.filter((k) => steps[k].level === "tested");
  const weak = keys.filter((k) => steps[k].level !== "tested");
  const parts = [];
  if (tested.length)
    parts.push(
      `On competitions it had not seen, the model ranked corners correctly ${tested
        .map((k) => `${Math.round((steps[k].auc ?? 0.5) * 100)}% of the time for ${STEP_PHRASE[k]}`)
        .join(" and ")} (a coin flip is 50%).`,
    );
  if (weak.length) parts.push(`It is weak on ${weak.map((k) => STEP_PHRASE[k]).join(" and ")}, so the final number is a rough guide.`);
  return parts.join(" ");
}

// Real corners and the model as two tabs in one card. The headline uses the more
// trusted source; the tabs show either in detail.
function sourcesHTML(r, eyebrow) {
  const real = r.evidence.chance;
  const model = r.estimate.chain.chance;
  const modelOn = model.level !== "none" && model.value != null;
  const realOn = real.n > 0 && real.rate != null;
  // the server picks the headline source, so the number, verdict and tabs agree
  const trustReal = r.source ? r.source !== "model" : realIsEnough(r.evidence) || !modelOn;
  const fallback = r.source === "average";
  const main = fallback ? r.base?.chance?.rate : trustReal ? real.rate : model.value;
  const from = fallback
    ? "Too few similar corners and no reliable model, so this is the average"
    : trustReal
      ? "From real corners"
      : "From the model, as few real corners match";
  const tab = modelOn ? state.sourceTab ?? (trustReal ? "real" : "model") : "real";

  let agree;
  if (!modelOn) {
    agree = { tone: "warn", icon: ICONS.alert, text: "The model can't tell setups apart reliably, so only real corners are shown." };
  } else if (realOn && realIsEnough(r.evidence)) {
    agree = agrees(real, model)
      ? { tone: "good", icon: ICONS.check, text: "Both sources agree: the model's number is within the range of what really happened." }
      : { tone: "warn", icon: ICONS.warn, text: `The two disagree. With ${real.n} similar real corners, trust what really happened.` };
  } else {
    agree = {
      tone: "warn",
      icon: ICONS.alert,
      text: real.n
        ? `Only ${plural(real.n, "similar real corner")}, so the model carries more weight here.`
        : "No similar real corners, so this relies on the model alone.",
    };
  }

  const tabBtn = (id, kind, label, value, note, badge) => `
    <button type="button" class="source-tile" role="tab" id="tab-${id}" aria-selected="${tab === id}" aria-controls="panel-source" tabindex="${tab === id ? 0 : -1}" data-action="source-tab" data-value="${id}" data-focus-key="source-${id}">
      <span class="source-label">${ICONS[kind]}${label}</span>
      <span class="source-num">${pctText(value)}</span>
      <span class="source-note">${esc(note)}</span>
      ${badge}
    </button>`;

  const avg = r.base?.chance?.rate;
  return `
    <div class="card verdict verdict-${r.verdict}">
      <div class="verdict-top"><span class="eyebrow">${esc(eyebrow)}</span>${loadingDot()}</div>
      <div class="verdict-headline">
        <span class="verdict-icon">${VERDICT_ICON[r.verdict] ?? ICONS.equal}</span>
        <div>
          <h2>A good chance in about ${pctText(main)} of corners</h2>
          <p class="verdict-sub">${from}. ${esc(VERDICT_TEXT[r.verdict] ?? VERDICT_TEXT.unknown)} against ${esc(shapePhrase(r.shape))} (${pctText(avg)}).</p>
        </div>
      </div>
      <div class="source-grid" role="tablist" aria-label="Where the numbers come from">
        ${tabBtn("real", "real", "Real corners", real.rate, realOn ? `${real.k} of ${real.n} similar corners` : "No similar corners", pill(r.evidence.reliability.label, r.evidence.reliability.tone))}
        ${modelOn ? tabBtn("model", "model", "Model", model.value, model.low != null ? `likely ${pctText(model.low)} to ${pctText(model.high)}` : "", levelPill(model.level)) : ""}
      </div>
      <p class="source-agree source-agree-${agree.tone}">${agree.icon}<span>${esc(agree.text)}</span></p>
      <div class="source-panel" role="tabpanel" id="panel-source" aria-labelledby="tab-${tab}">
        ${tab === "model" ? modelPanel(r) : realPanel(r)}
      </div>
    </div>`;
}

function realPanel(r) {
  const ev = r.evidence;
  const base = r.base || {};
  const refLabel = `All corners vs ${r.shape}`;
  const n = ev.chance.n || 0;
  const kWon = ev.chance_if_won.k || 0;
  const kLost = ev.chance_if_lost.k || 0;
  // chances from corners where the first ball wasn't contested (out, foul)
  const kOther = Math.max(0, (ev.chance.k || 0) - kWon - kLost);
  const parts = n
    ? [
        { kind: "won", value: kWon / n, label: "after winning the first ball" },
        { kind: "lost", value: kLost / n, label: "after losing it" },
        { kind: "other", value: kOther / n, label: "no contest" },
      ]
    : [];
  return `
    <div class="source-head"><span class="source-icon">${ICONS.real}</span><div><h3 class="section-title">What happened in real corners</h3><p class="section-sub">Counted from past matches. No model involved.</p></div></div>
    <p class="source-desc">${esc(ev.description)}</p>
    ${rateBar({ label: "Good chance created", rate: ev.chance, note: fraction(ev.chance), ref: base.chance?.rate, refLabel, size: "main", segments: parts })}
    ${parts.length ? segKey(parts) : ""}
    <div class="breakdown">
      <div class="chain-label">How it got there</div>
      ${rateBar({ label: "1 · First ball won", rate: ev.first_contact, note: fraction(ev.first_contact), ref: base.first_contact?.rate, refLabel, size: "sub" })}
      ${rateBar({ label: "2 · Good chance after winning it", rate: ev.chance_if_won, note: fraction(ev.chance_if_won), ref: base.chance_if_won?.rate, refLabel, size: "sub", dot: "won" })}
      ${rateBar({ label: "2 · Good chance after losing it", rate: ev.chance_if_lost, note: fraction(ev.chance_if_lost), ref: base.chance_if_lost?.rate, refLabel, size: "sub", dot: "lost" })}
    </div>
    ${rateLegend(refLabel)}
    ${formulaHTML(ev.first_contact.rate, ev.chance_if_won.rate, ev.chance_if_lost.rate)}`;
}

function modelPanel(r) {
  const ch = r.estimate.chain;
  const goal = ch.chance;
  const base = r.base || {};
  const refLabel = `All corners vs ${r.shape}`;
  const badge = (step) => `<span title="${esc(LEVEL_TIP[step.level] ?? "")}">${levelPill(step.level)}</span>`;
  const bar = (label, step, ref, extra = {}) =>
    rateBar({ label, rate: { rate: step.value, low: step.low, high: step.high }, ref, refLabel, badge: badge(step), ...extra });
  const win = ch.win.value;
  const ready = win != null && ch.if_won.value != null && ch.if_lost.value != null && goal.value != null;
  // each path's share of the estimate, scaled to add up to the headline number
  // (the chain averages over real situations)
  let parts = [];
  if (ready) {
    const won = win * ch.if_won.value;
    const lost = (1 - win) * ch.if_lost.value;
    const scale = won + lost > 0 ? goal.value / (won + lost) : 0;
    parts = [
      { kind: "won", value: won * scale, label: "after winning the first ball" },
      { kind: "lost", value: lost * scale, label: "after losing it" },
    ];
  }
  return `
    <div class="source-head"><span class="source-icon">${ICONS.model}</span><div><h3 class="section-title">What the model predicts</h3><p class="section-sub">For your exact setup, learned from all ${META.n_corners} corners. Most useful when few real corners match.</p></div></div>
    ${bar("Good chance created", goal, base.chance?.rate, { size: "main", segments: parts })}
    ${parts.length ? segKey(parts) : ""}
    <div class="breakdown">
      <div class="chain-label">How it gets there</div>
      ${bar("1 · First ball won", ch.win, base.first_contact?.rate, { size: "sub" })}
      ${bar("2 · Good chance after winning it", ch.if_won, base.chance_if_won?.rate, { size: "sub", dot: "won" })}
      ${bar("2 · Good chance after losing it", ch.if_lost, base.chance_if_lost?.rate, { size: "sub", dot: "lost" })}
    </div>
    ${rateLegend(refLabel)}
    ${formulaHTML(win, ch.if_won.value, ch.if_lost.value)}`;
}

function whyHTML(why, againstShape) {
  if (META.model.level === "none" || (!why.helps.length && !why.hurts.length)) return "";
  const group = (label, tone, icon, items) => `
    <div>
      <div class="why-label why-${tone}">${label}</div>
      <ul class="chip-list">${
        items.length
          ? items.map((text) => `<li class="chip chip-${tone}">${icon}${esc(text)}</li>`).join("")
          : '<li class="chip chip-empty">Nothing stands out</li>'
      }</ul>
    </div>`;
  return `
    <div class="card">
      <h3 class="section-title">What the model sees${againstShape ? ` <span class="muted">vs ${esc(againstShape)}</span>` : ""}</h3>
      <p class="section-sub">Which of your choices push the chance of winning the first ball (step 1) up or down.</p>
      <div class="why-grid">
        ${group("Helping", "good", ICONS.check, why.helps)}
        ${group("Holding it back", "bad", ICONS.minus, why.hurts)}
      </div>
    </div>`;
}

function effectHTML(m, label) {
  const p = m.points;
  const tone = p > 0 ? "good" : p < 0 ? "bad" : "muted";
  const icon = p > 0 ? ICONS.check : p < 0 ? ICONS.minus : "";
  const sign = p > 0 ? "+" : p < 0 ? "−" : "±";
  const tip = `${m.agreement}% of models agree`;
  return `<span class="effect effect-${tone}" title="${esc(tip)}">${icon}${sign}${Math.abs(p)} pts ${label}<span class="effect-agree">${m.agreement}%</span></span>`;
}

function suggestionsHTML(suggestions, againstShape) {
  let body;
  if (META.model.level === "none") {
    body = `<div class="empty-inline">${ICONS.alert}<span>Suggestions need a model that beats the average on unseen competitions. This one doesn't, so none are shown.</span></div>`;
  } else if (!suggestions.length) {
    body = `<div class="empty-inline">${ICONS.check}<span>No single change makes a reliable difference. A change is only suggested when at least ${Math.round(META.model.agreement * 100)}% of the models agree it helps, and it doesn't clearly hurt the other measure.</span></div>`;
  } else {
    body = `<ol class="sg-list">${suggestions
      .map(
        (s, i) => `
        <li class="suggestion">
          <span class="sg-rank">${i + 1}</span>
          <div class="sg-body">
            <div class="sg-changes">${s.changes.map((c) => `<span class="change change-${c.kind}">${esc(c.text)}</span>`).join("")}</div>
            <p class="sg-text">${esc(s.text)}</p>
            <div class="sg-effects">
              ${effectHTML(s.chance, "good chances")}
              ${effectHTML(s.first_ball, "first-ball wins")}
              ${s.trade_off ? `<span class="effect effect-warn">${ICONS.warn}Trade-off</span>` : ""}
            </div>
            ${s.real ? `<p class="sg-real">Real corners (${esc(s.real.scope)}), ${s.real.measure === "chance" ? "good chances" : "first ball won"}: ${pctText(s.real.now.rate)} <small>(${fraction(s.real.now)})</small> now, ${pctText(s.real.after.rate)} <small>(${fraction(s.real.after)})</small> after the change</p>` : `<p class="sg-real">Model only: too few similar real corners to check this against.</p>`}
          </div>
          <button type="button" class="btn btn-secondary btn-small" data-action="apply" data-index="${i}">Try it</button>
        </li>`,
      )
      .join("")}</ol>`;
  }
  return `
    <div class="card">
      <h3 class="section-title">Quick wins${againstShape ? ` <span class="muted">vs ${esc(againstShape)}</span>` : ""}</h3>
      <p class="section-sub">Single changes the models agree would create more good chances or win the first ball more often, without clearly hurting the other. Percentages show how many models agree. Changes that real similar corners contradict are left out.</p>
      ${body}
    </div>`;
}

function numbersHTML(rows) {
  const xg = (n) =>
    n?.value == null
      ? "–"
      : `${n.value.toFixed(3)}${n.low != null ? `<small class="range-text">${n.low.toFixed(3)} to ${n.high.toFixed(3)}</small>` : ""}`;
  const prob = (n) =>
    n?.value == null ? "–" : `${pctText(n.value)}${n.low != null ? `<small class="range-text">${pctText(n.low)} to ${pctText(n.high)}</small>` : ""}`;
  return `
    <details class="card details" data-numbers ${state.numbersOpen ? "open" : ""}>
      <summary><span>Numbers for analysts</span>${ICONS.chevron}</summary>
      <p class="section-sub">Model outputs with 90% ranges from models refit on resampled matches. xG created counts your shots while you keep the ball after the corner; xG conceded counts their shots in the possession that follows, both within ${META.rules.horizon_secs} seconds.</p>
      <div class="table-wrap">
        <table class="num-table">
          <thead><tr><th>Defence</th><th>First ball</th><th>Chance if won</th><th>Chance if lost</th><th>Good chance</th></tr></thead>
          <tbody>${rows
            .map(
              (r) =>
                `<tr><td>${esc(r.shape)}</td><td>${prob(r.estimate.chain.win)}</td><td>${prob(r.estimate.chain.if_won)}</td><td>${prob(r.estimate.chain.if_lost)}</td><td>${prob(r.estimate.chain.chance)}</td></tr>`,
            )
            .join("")}</tbody>
        </table>
      </div>
      <div class="table-wrap">
        <table class="num-table">
          <thead><tr><th>Defence</th><th>xG created</th><th>xG conceded</th><th>Net xG</th></tr></thead>
          <tbody>${rows
            .map((r) => `<tr><td>${esc(r.shape)}</td><td>${xg(r.estimate.xg_for)}</td><td>${xg(r.estimate.xg_against)}</td><td>${xg(r.estimate.net)}</td></tr>`)
            .join("")}</tbody>
        </table>
      </div>
      <p class="tiny muted">Good chance = a shot of at least ${META.rules.xg_threshold} xG. Good chance = first ball × chance if won + (1 − first ball) × chance if lost, on contested corners. Net xG = PV_for − PV_against = APV_net.</p>
    </details>`;
}

function singleHTML(r) {
  return `
    ${sourcesHTML(r, `Against ${shapePhrase(r.shape)}`)}
    ${rareHTML(r.rare)}
    ${whyHTML(r.why)}
    ${numbersHTML([r])}`;
}

function compareHTML(r) {
  const rows = r.shapes
    .map((s) => {
      const fc = s.evidence.chance;
      const est = s.estimate.chain.chance;
      const useModel = est.level !== "none" && est.value != null;
      const best = r.clear_winner && s.shape === r.best_shape;
      const bar = rateBar({
        label: useModel ? "Estimate" : "Similar corners",
        rate: useModel ? { rate: est.value, low: est.low, high: est.high } : fc,
        ref: s.base?.chance?.rate,
        refLabel: `All corners vs ${s.shape}`,
      });
      return `
      <div class="compare-row${best ? " best" : ""}">
        <div class="compare-name">${esc(s.shape)}${best ? '<span class="best-tag">Best match</span>' : ""}</div>
        <div class="compare-bar">${bar}</div>
        <div class="compare-evidence"><small>Similar corners</small><b>${fc.n ? `${fc.k} of ${fc.n}` : "none"}</b></div>
      </div>`;
    })
    .join("");
  return `
    <div class="card verdict verdict-${r.clear_winner ? "better" : "similar"}">
      <div class="verdict-top"><span class="eyebrow">Good chances, across all defences</span>${loadingDot()}</div>
      <div class="verdict-headline"><span class="verdict-icon">${ICONS.target}</span><h2>${esc(r.headline)}</h2></div>
      <div class="compare-list">${rows}</div>
      ${rateLegend("All corners vs that defence")}
    </div>
    ${rareHTML(r.rare)}
    ${whyHTML(r.why, r.best_shape)}
    ${numbersHTML(r.shapes)}`;
}

function invalidHTML() {
  const total = totalPlayers();
  const max = META.players_total;
  const missing = max - total;
  return `
    <div class="card empty-state">
      <div class="ring" style="--p:${Math.min(total / max, 1)}"><span>${total}/${max}</span></div>
      <h2>${missing > 0 ? `Place ${plural(missing, "more player")}` : `Remove ${plural(-missing, "player")}`}</h2>
      <p class="muted">The rating appears once all ${max} outfield players are placed. Click a zone on the pitch or use the + buttons.</p>
    </div>`;
}

function errorHTML(message) {
  return `
    <div class="card empty-state">
      <div class="error-icon">${ICONS.warn}</div>
      <h2>Couldn't rate this corner</h2>
      <p class="muted">${esc(message)}</p>
      <button type="button" class="btn btn-secondary" data-action="retry">Try again</button>
    </div>`;
}

function skeletonHTML() {
  return `
    <div class="card" aria-busy="true">
      <div class="skeleton sk-line" style="width:40%"></div>
      <div class="skeleton sk-title"></div>
      <div class="skeleton sk-bar"></div>
      <div class="skeleton sk-bar"></div>
    </div>
    <div class="card" aria-hidden="true">
      <div class="skeleton sk-line" style="width:50%"></div>
      <div class="skeleton sk-line"></div>
      <div class="skeleton sk-line" style="width:80%"></div>
    </div>`;
}

function renderResults() {
  const loading = state.status === "loading" && !!state.result;
  resultsEl.classList.toggle("is-loading", loading);
  winsEl.classList.toggle("is-loading", loading);
  if (state.status === "invalid") {
    resultsEl.innerHTML = invalidHTML();
    winsEl.innerHTML = "";
  } else if (state.status === "error") {
    resultsEl.innerHTML = errorHTML(state.error);
    winsEl.innerHTML = "";
  } else if (!state.result) {
    resultsEl.innerHTML = skeletonHTML();
    winsEl.innerHTML = "";
  } else {
    const r = state.result;
    renderInto(resultsEl, r.mode === "compare" ? compareHTML(r) : singleHTML(r));
    winsEl.innerHTML = suggestionsHTML(r.suggestions, r.mode === "compare" ? r.best_shape : null);
  }
  renderMobileBar();
}

function renderMobileBar() {
  const r = state.result;
  const show = state.view === "planner" && state.status !== "invalid" && state.status !== "error" && !!r;
  mobileBar.hidden = !show;
  document.body.classList.toggle("has-mobile-bar", show);
  if (!show) return;
  const focus = r.mode === "compare" ? r.shapes[0] : r;
  const fc = focus.evidence.first_contact;
  const ch = focus.evidence.chance;
  mobileBar.innerHTML = `
    <div class="mb-text">
      <div class="mb-title">${esc(r.headline)}</div>
      <div class="mb-pills"><span class="pill-row"><small>Chances</small>${pill(ch.n ? fraction(ch) : "–", "neutral")}</span><span class="pill-row"><small>First ball</small>${pill(fc.n ? fraction(fc) : "–", "neutral")}</span>${pill(focus.evidence.reliability.label, focus.evidence.reliability.tone)}</div>
    </div>
    <button type="button" class="btn btn-primary btn-small" data-action="scroll-results">Details</button>`;
}

// Insights

// A small bar for a table cell: value and count, then the bar.
function cellBar(rate, ref, refLabel = "All corners") {
  const r = rate || {};
  const pos = (x) => `${(Math.max(0, Math.min(1, x)) * 100).toFixed(2)}%`;
  const hasRange = r.low != null && r.high != null;
  const tip = `${pctText(r.rate)}${r.n ? ` (${fraction(r)})` : ""}${hasRange ? `, likely between ${pctText(r.low)} and ${pctText(r.high)}` : ""}`;
  return `
    <div class="cell-bar" title="${esc(tip)}">
      <div class="cell-val">${pctText(r.rate)}<small>${r.n ? fraction(r) : "none"}</small></div>
      <div class="rate-track" role="img" aria-label="${esc(tip)}">
        ${r.rate != null ? `<span class="rate-fill" style="width:${pos(r.rate)}"></span>` : ""}
        ${hasRange ? `<span class="rate-range" style="left:${pos(r.low)};width:${pos(r.high - r.low)}"></span>` : ""}
        ${ref != null ? `<span class="rate-ref" style="left:${pos(ref)}" title="${esc(refLabel)}"></span>` : ""}
      </div>
    </div>`;
}

const MEASURE_MARK = { first_contact: "1", chance_if_won: "2", chance: "=", shot_conceded: "" };

// One factor across several measures, on one 0-100% scale, with a table view.
function factorCard(f, measures, overall, extra = "") {
  const rows = f.rows.filter((row) => row.corners > 0);
  const head = measures
    .map(
      ([id, name]) =>
        `<div class="fg-head">${MEASURE_MARK[id] ? `<span class="step-num">${MEASURE_MARK[id]}</span>` : ""}<span>${esc(name)}</span></div>`,
    )
    .join("");
  const body = rows
    .map(
      (row) => `
      <div class="fg-label">${esc(row.label)}<small>${plural(row.corners, "corner")}</small></div>
      ${measures
        .map(
          ([id, name]) =>
            `<div class="fg-cell"><span class="fg-measure">${esc(name)}</span>${cellBar(row[id], overall[id]?.rate)}</div>`,
        )
        .join("")}`,
    )
    .join("");
  const table = rows
    .map(
      (row) =>
        `<tr><td>${esc(row.label)}</td>${measures
          .map(([id]) => `<td>${pctText(row[id].rate)} <small>${fraction(row[id])}</small><br><small>${pctText(row[id].low)} to ${pctText(row[id].high)}</small></td>`)
          .join("")}</tr>`,
    )
    .join("");
  return `
    <article class="card factor-card">
      <h3 class="section-title">${esc(f.title)}</h3>
      <p class="section-sub">${esc(f.subtitle)}</p>
      ${extra}
      <div class="factor-grid" style="--cols:${measures.length}">
        <div class="fg-head fg-corner"></div>${head}
        ${body}
      </div>
      ${rateLegend("All corners")}
      <details class="chart-table">
        <summary>Show as table</summary>
        <div class="table-wrap"><table class="num-table">
          <thead><tr><th>Group</th>${measures.map(([, name]) => `<th>${esc(name)}</th>`).join("")}</tr></thead>
          <tbody>${table}</tbody>
        </table></div>
      </details>
    </article>`;
}

function renderInsights() {
  const el = $("#insights-body");
  const d = state.insights;
  if (!d) {
    el.innerHTML = skeletonHTML();
    return;
  }
  const o = d.overall;
  const chain = d.measures.map((m) => [m.id, m.name]);

  // takeaways grouped by factor, with a note when a factor has a trade-off
  const byFactor = new Map();
  d.takeaways.forEach((t) => byFactor.set(t.factor, [...(byFactor.get(t.factor) || []), t]));
  const takeaways = [...byFactor.entries()]
    .map(
      ([factor, items]) => `
      <div class="card takeaway">
        <span class="eyebrow">${esc(factor)}</span>
        <ul>${items.map((t) => `<li><span class="tk-measure">${esc(t.measure)}</span>${esc(t.text)}</li>`).join("")}</ul>
        ${items.some((t) => t.trade_off) ? `<p class="tk-note">${ICONS.alert}<span>A trade-off: the ball that is easiest to win is not the one that most often leads to a chance.</span></p>` : ""}
      </div>`,
    )
    .join("");

  el.innerHTML = `
    <div class="stat-row">
      <div class="card stat"><span class="stat-label">Real corners</span><span class="stat-value">${o.corners.toLocaleString()}</span>${d.competitions ? `<span class="stat-note">from ${plural(d.competitions, "competition")}</span>` : ""}</div>
      <div class="card stat"><span class="stat-label"><span class="step-num">1</span>First ball won by the attack</span><span class="stat-value">${pctText(o.first_contact.rate)}</span><span class="stat-note">${fraction(o.first_contact)} contested</span></div>
      <div class="card stat"><span class="stat-label"><span class="step-num">2</span>Good chance after winning it</span><span class="stat-value">${pctText(o.chance_if_won.rate)}</span><span class="stat-note">${fraction(o.chance_if_won)}</span></div>
      <div class="card stat"><span class="stat-label"><span class="step-num">=</span>Good chance overall</span><span class="stat-value">${pctText(o.chance.rate)}</span><span class="stat-note">${fraction(o.chance)}</span></div>
    </div>

    ${takeaways ? `<h2 class="findings-title">What to take away</h2><p class="section-lead">Only differences that are clear despite the small samples: the 90% ranges of the best and worst group don't overlap.</p><div class="takeaways">${takeaways}</div>` : ""}

    <h2 class="findings-title">What makes a difference</h2>
    <p class="section-lead">Each factor through the two steps of a corner: <b>1</b> win the first ball, <b>2</b> turn it into a good chance, <b>=</b> good chances overall. All bars share the same 0–100% scale.</p>
    <div class="factor-list">${d.factors.map((f) => factorCard(f, chain, o)).join("")}</div>

    <h2 class="findings-title">What makes little difference</h2>
    <div class="factor-list factor-list-2">
      ${d.little.map((f) => factorCard(f, f.measures, o, `<p class="factor-note">${ICONS.alert}<span>${esc(f.note)}</span></p>`)).join("")}
    </div>

    <h2 class="findings-title">How the first ball was decided</h2>
    <article class="card chart-card">
      <p class="section-sub">Share of contested corners.</p>
      <div class="chart-rows">
        ${d.contact_types.map((t) => rateBar({ label: t.label, rate: { rate: t.share }, note: `${t.count}` })).join("")}
      </div>
    </article>`;
}

async function loadInsights() {
  state.insightsLoading = true;
  renderInsights();
  try {
    state.insights = await api.insights();
  } catch (err) {
    $("#insights-body").innerHTML = errorHTML(err.message);
    state.insightsLoading = false;
    return;
  }
  state.insightsLoading = false;
  renderInsights();
}

// Playbook

function renderPlaybookFilters() {
  const pb = state.pb;
  const max = META.playbook.max_used[pb.shape] ?? 1;
  const shapes = [{ id: META.all_shapes, name: "All defences" }, ...META.shapes];
  renderInto(
    $("#pb-filters"),
    `
    <div class="field"><span class="field-label" id="lbl-pb-shape">Against</span>${segmented("pb-shape", shapes, pb.shape, "lbl-pb-shape")}</div>
    <div class="field"><span class="field-label" id="lbl-pb-sort">Rank by</span>${segmented("pb-sort", META.playbook.sorts, pb.sort, "lbl-pb-sort")}</div>
    <div class="field">
      <span class="field-label" id="lbl-pb-used">At least</span>
      <div class="stepper" role="group" aria-labelledby="lbl-pb-used">
        <button type="button" class="step-btn" data-action="pb-used" data-delta="-5" data-focus-key="pb-minus" aria-label="Fewer corners" ${pb.minUsed <= 5 ? "disabled" : ""}>−</button>
        <output class="step-val">${pb.minUsed}</output>
        <button type="button" class="step-btn" data-action="pb-used" data-delta="5" data-focus-key="pb-plus" aria-label="More corners" ${pb.minUsed + 5 > max ? "disabled" : ""}>+</button>
      </div>
      <span class="field-hint">corners</span>
    </div>
    <label class="check"><input type="checkbox" data-action="pb-other" data-focus-key="pb-other" ${pb.includeOther ? "checked" : ""}> Include short corners and other deliveries</label>
    <p class="pb-count" id="pb-count"></p>`,
  );
  renderPlaybookCount();
}

function renderPlaybookCount() {
  const el = $("#pb-count");
  const data = state.pb.data;
  if (!el) return;
  el.textContent = data && !state.pb.loading ? `Showing ${data.items.length} of ${plural(data.total, "group")}` : "";
}

function pbCardHTML(item, index) {
  const sort = state.pb.data?.sort ?? "chance";
  const stats = [
    ["chance", "Good chance", item.chance],
    ["first_contact", "First ball won", item.first_contact],
  ].sort((a, b) => (a[0] === sort ? -1 : b[0] === sort ? 1 : 0));
  const [mainId, mainName, main] = stats[0];
  const [, otherName, other] = stats[1];
  return `
    <article class="card pb-card">
      <div class="pb-thumb">
        <svg class="pitch compact" viewBox="${VIEWBOX}" role="img" aria-label="${esc(setupSummary(item))}">${pitchInner(item, { compact: true })}</svg>
      </div>
      <div class="pb-body">
        <h3 class="pb-title">${esc(item.delivery_name)} <span class="arrow">→</span> ${esc(item.zone_name)}</h3>
        <p class="pb-meta">${esc(item.box)} · ${plural(item.used, "real corner")}</p>
        <div class="pb-main">
          <div>
            <span class="pb-main-label">${esc(mainName)}</span>
            <span class="pb-main-num">${pctText(main.rate)}</span>
            <span class="pb-main-note">${main.n ? `${fraction(main)} · likely ${pctText(main.low)} to ${pctText(main.high)}` : mainId === "first_contact" ? "never contested" : ""}</span>
          </div>
          ${pill(item.reliability.label, item.reliability.tone)}
        </div>
        <p class="pb-other">${esc(otherName)}: <b>${pctText(other.rate)}</b> <small>${other.n ? fraction(other) : "never contested"}</small></p>
        <button type="button" class="btn btn-secondary btn-block" data-action="pb-open" data-index="${index}" ${
          item.plannable ? "" : 'disabled title="Short corners and other deliveries are not available in the planner"'
        }>Open in planner</button>
      </div>
    </article>`;
}

function renderPlaybook() {
  const grid = $("#pb-grid");
  const pb = state.pb;
  renderPlaybookCount();
  if (pb.loading && !pb.data) {
    grid.innerHTML = Array.from(
      { length: 6 },
      () => `<div class="card pb-card" aria-hidden="true"><div class="skeleton sk-thumb"></div><div class="pb-body"><div class="skeleton sk-line" style="width:70%"></div><div class="skeleton sk-line" style="width:45%"></div><div class="skeleton sk-line"></div></div></div>`,
    ).join("");
    $("#pb-more").hidden = true;
    return;
  }
  const data = pb.data;
  if (!data) return;
  if (!data.items.length) {
    const against = pb.shape === META.all_shapes ? "any defence" : shapePhrase(pb.shape);
    grid.innerHTML = `<div class="card pb-empty"><h2>No groups match</h2><p class="muted">No group has ${pb.minUsed} or more real corners against ${esc(against)}. Lower “At least”.</p></div>`;
  } else {
    grid.innerHTML = data.items.map(pbCardHTML).join("");
  }
  $("#pb-more").hidden = data.items.length >= data.total;
}

async function loadPlaybook() {
  const pb = state.pb;
  const id = ++pb.requestId;
  pb.loading = true;
  renderPlaybook();
  try {
    const data = await api.playbook({ shape: pb.shape, min_used: pb.minUsed, limit: pb.limit, sort: pb.sort, include_other: pb.includeOther });
    if (id !== pb.requestId) return;
    pb.data = data;
  } catch (err) {
    if (id !== pb.requestId) return;
    $("#pb-grid").innerHTML = `<div class="card pb-empty">${errorHTML(err.message)}</div>`;
    pb.loading = false;
    return;
  }
  pb.loading = false;
  renderPlaybook();
}

function openInPlanner(index) {
  const item = state.pb.data?.items?.[index];
  if (!item?.plannable) return;
  pushUndo();
  state.setup = { shape: item.shape, delivery: item.delivery, zone: item.zone, players: clone(item.players) };
  setView("planner");
  onSetupChange();
  toast(`Loaded the average setup of this group, against ${shapePhrase(item.shape)}.`, true);
}

// Views, help, theme, toast

const VIEWS = ["planner", "insights", "playbook"];

function setView(view, { updateUrl = true } = {}) {
  state.view = view;
  $$(".nav-tab").forEach((tab) => {
    const on = tab.dataset.view === view;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
  });
  for (const name of VIEWS) $(`#view-${name}`).hidden = view !== name;
  if (updateUrl) history.replaceState(null, "", view === "planner" ? location.pathname : `#${view}`);
  if (view === "playbook" && !state.pb.data && !state.pb.loading) loadPlaybook();
  if (view === "insights" && !state.insights && !state.insightsLoading) loadInsights();
  renderMobileBar();
  window.scrollTo({ top: 0 });
}

function helpHTML() {
  const steps = META.model.steps || {};
  const level = (k) => ({ tested: "tested", rough: "a rough guide", none: "weak" })[steps[k]?.level] ?? "untested";
  return `
    <h3>What the planner measures</h3>
    <p>A corner works in two steps, and the planner shows both:</p>
    <dl class="def-list">
      <dt>1 · First ball won</dt>
      <dd>An attacker is the first to play the ball after the delivery (a header, a shot, a flick-on, or controlling it) rather than a defender or the keeper. Corners that go straight out or end in a foul don't count. For short corners, the first ball into the box within 3 passes counts as the delivery.</dd>
      <dt>2 · Good chance</dt>
      <dd>A shot worth at least ${META.rules.xg_threshold} xG while you keep the ball, within ${META.rules.horizon_secs} seconds. Shown separately for corners where you won the first ball and where you lost it.</dd>
      <dt>= Good chance overall</dt>
      <dd>first ball won × good chance after winning + first ball lost × good chance after losing. This is the headline number.</dd>
    </dl>
    <h3>Real corners and the model</h3>
    <p><b>Real corners</b> counts what happened in past corners like yours: same defence and delivery, nearly the same player setup. The search widens until there are at least 15 to learn from, and the card says how wide it went. Each bar shows the real rate, the whisker its 90% range, and the thin line the average for all corners against that defence.</p>
    <p><b>The model</b> estimates your exact setup, including ones nobody tried, and is labelled by how it did on competitions it never saw. Winning the first ball is ${level("win")}, a good chance after winning it is ${level("if_won")}, and a good chance after losing it is ${level("if_lost")}, so the overall estimate is ${level("chance")}. ${esc(accuracyText())}</p>
    <p>The headline uses real corners when there are at least 15 similar ones, and the model otherwise. When the two disagree, trust the real corners.</p>
    <h3>Their defence</h3>
    <dl class="def-list">${META.shapes.map((s) => `<dt>${esc(s.name)}</dt><dd>${esc(s.description)}</dd>`).join("")}</dl>
    <p style="margin-top:10px">The shape comes from two simple rules on where defenders stood as the corner was taken: is anyone on a post, and how close to goal do the box defenders stand on average (7.5 yards splits deep from high). One snapshot can't show who is marking whom, so the shapes describe positions, not marking. In this data the shape changes which real corners you're compared with, but not which setup works best.</p>
    <h3>Player zones</h3>
    <dl class="def-list">${META.slots.map((s) => `<dt>${esc(s.label)}</dt><dd>${esc(s.hint)}</dd>`).join("")}</dl>
    <h3>Quick wins</h3>
    <p>Each suggestion changes one thing: moves one player, or switches the delivery or target. It is shown only if at least ${Math.round(META.model.agreement * 100)}% of 100 models, each refit on resampled matches, agree it creates more good chances or wins the first ball more often, without clearly hurting the other. It is then checked against real corners and left out if they point the other way. Changes towards setups real teams almost never used are left out too.</p>
    <h3>Where the data comes from</h3>
    <p>${META.n_corners.toLocaleString()} real corners with StatsBomb 360 freeze-frame data: men's World Cup and Euros, Bundesliga, Ligue 1 and La Liga seasons, and the women's World Cup and Euros. All women's data comes from international tournaments.</p>`;
}

const effectiveTheme = () =>
  document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

function renderThemeButton() {
  $("#theme-btn").innerHTML = effectiveTheme() === "dark" ? ICONS.sun : ICONS.moon;
}

let toastTimer = null;
function toast(message, withUndo = false) {
  toastEl.innerHTML = `<span>${esc(message)}</span>${withUndo ? '<button type="button" class="toast-btn" data-action="undo">Undo</button>' : ""}`;
  toastEl.hidden = false;
  requestAnimationFrame(() => toastEl.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, withUndo ? 6000 : 3200);
}

function hideToast() {
  clearTimeout(toastTimer);
  toastEl.classList.remove("show");
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, 220);
}

// Events

controlsEl.addEventListener("click", (event) => {
  const btn = event.target.closest("[data-action]");
  if (!btn || btn.disabled) return;
  const { action, value } = btn.dataset;
  if (action === "shape" || action === "delivery" || action === "zone") setField(action, value);
  else if (action === "count") changeCount(btn.dataset.zone, Number(btn.dataset.delta));
  else if (action === "reset") resetPlayers();
});
controlsEl.addEventListener("mouseover", (event) => setHover(event.target.closest(".stepper-row")?.dataset.zone ?? null));
controlsEl.addEventListener("mouseleave", () => setHover(null));

pitchEl.addEventListener("click", (event) => {
  const player = event.target.closest(".player");
  if (player) return changeCount(player.dataset.zone, -1);
  const zone = event.target.closest(".zone");
  if (zone) changeCount(zone.dataset.zone, 1);
});
pitchEl.addEventListener("mouseover", (event) => setHover(event.target.closest("[data-zone]")?.dataset.zone ?? null));
pitchEl.addEventListener("mouseleave", () => setHover(null));

resultsEl.addEventListener("click", (event) => {
  const btn = event.target.closest("[data-action]");
  if (!btn) return;
  const { action } = btn.dataset;
  if (action === "apply") applySuggestion(Number(btn.dataset.index));
  else if (action === "source-tab") {
    state.sourceTab = btn.dataset.value;
    store.set("apv-source-tab", state.sourceTab);
    renderResults();
  } else if (action === "help") $("#help-dialog").showModal();
  else if (action === "retry") scheduleEvaluate(true);
});
resultsEl.addEventListener("keydown", (event) => {
  const tab = event.target.closest('[role="tab"][data-action="source-tab"]');
  if (!tab || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = $$('[role="tab"][data-action="source-tab"]', resultsEl);
  if (tabs.length < 2) return;
  event.preventDefault();
  const i = tabs.indexOf(tab);
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (i + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  state.sourceTab = tabs[next].dataset.value;
  store.set("apv-source-tab", state.sourceTab);
  renderResults();
  resultsEl.querySelector(`[data-focus-key="source-${state.sourceTab}"]`)?.focus();
});
winsEl.addEventListener("click", (event) => {
  const btn = event.target.closest('[data-action="apply"]');
  if (btn) applySuggestion(Number(btn.dataset.index));
});
resultsEl.addEventListener(
  "toggle",
  (event) => {
    if (event.target.matches("[data-numbers]")) state.numbersOpen = event.target.open;
  },
  true,
);

mobileBar.addEventListener("click", (event) => {
  if (event.target.closest('[data-action="scroll-results"]')) resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
});

toastEl.addEventListener("click", (event) => {
  if (event.target.closest('[data-action="undo"]')) undo();
});

$("#pb-filters").addEventListener("click", (event) => {
  const btn = event.target.closest("[data-action]");
  if (!btn || btn.disabled) return;
  const pb = state.pb;
  if (btn.dataset.action === "pb-shape") {
    if (pb.shape === btn.dataset.value) return;
    pb.shape = btn.dataset.value;
    pb.minUsed = Math.max(5, Math.min(pb.minUsed, META.playbook.max_used[pb.shape] ?? 5));
  } else if (btn.dataset.action === "pb-sort") {
    if (pb.sort === btn.dataset.value) return;
    pb.sort = btn.dataset.value;
  } else if (btn.dataset.action === "pb-used") {
    pb.minUsed = Math.max(5, pb.minUsed + Number(btn.dataset.delta));
  } else if (btn.dataset.action === "pb-other") {
    pb.includeOther = btn.checked;
  } else {
    return;
  }
  pb.limit = 9;
  pb.data = null;
  renderPlaybookFilters();
  loadPlaybook();
});

$("#pb-grid").addEventListener("click", (event) => {
  const btn = event.target.closest('[data-action="pb-open"]');
  if (btn && !btn.disabled) openInPlanner(Number(btn.dataset.index));
});

$("#pb-more").addEventListener("click", () => {
  state.pb.limit += 9;
  loadPlaybook();
});

$$(".nav-tab").forEach((tab) => {
  tab.addEventListener("click", () => setView(tab.dataset.view));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const step = event.key === "ArrowRight" ? 1 : -1;
    const next = VIEWS[(VIEWS.indexOf(tab.dataset.view) + step + VIEWS.length) % VIEWS.length];
    setView(next);
    $(`.nav-tab[data-view="${next}"]`).focus();
  });
});

const helpDialog = $("#help-dialog");
$("#help-btn").addEventListener("click", () => helpDialog.showModal());
$("#help-close").addEventListener("click", () => helpDialog.close());
helpDialog.addEventListener("click", (event) => {
  if (event.target === helpDialog) helpDialog.close();
});

$("#theme-btn").addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store.set("apv-theme", next);
  renderThemeButton();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderThemeButton);

$("#intro-dismiss").addEventListener("click", () => {
  $("#intro").hidden = true;
  store.set("apv-intro-dismissed", true);
});

// Start

async function init() {
  renderThemeButton();
  try {
    META = await api.meta();
  } catch (err) {
    resultsEl.innerHTML = errorHTML(`Could not reach the server (${err.message}). Is app.py running?`);
    resultsEl.querySelector('[data-action="retry"]')?.addEventListener("click", () => location.reload());
    return;
  }
  SLOT_BY_ID = Object.fromEntries(META.slots.map((s) => [s.id, s]));
  state.pb.shape = META.all_shapes;
  state.pb.minUsed = META.playbook.min_used;

  const savedTab = store.get("apv-source-tab");
  if (savedTab === "real" || savedTab === "model") state.sourceTab = savedTab;
  const saved = store.get("apv-setup");
  state.setup = validSetup(saved) ? saved : clone(META.defaults);

  $("#help-body").innerHTML = helpHTML();
  $("#intro").hidden = !!store.get("apv-intro-dismissed");

  renderControls();
  renderPitch();
  renderCount();
  renderPlaybookFilters();
  setView(VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "planner", { updateUrl: false });
  scheduleEvaluate(true);
}

init();
