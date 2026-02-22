/* Leaderboard UI:
 * - loads docs/leaderboard.json (fallback ../leaderboard/leaderboard.csv)
 * - search + filters + sortable columns + optional column visibility
 * - renders summary cards and a top-3 podium
 */
function parseCSV(text){
  const lines = text.trim().split(/\r?\n/);
  if(!lines.length){ return []; }
  const header = lines[0].split(",");
  const rows = [];

  for(let i = 1; i < lines.length; i++){
    if(!lines[i].trim()){ continue; }
    const cols = [];
    let cur = "";
    let inQ = false;

    for(let j = 0; j < lines[i].length; j++){
      const ch = lines[i][j];
      if(ch === '"'){ inQ = !inQ; continue; }
      if(ch === "," && !inQ){ cols.push(cur); cur = ""; continue; }
      cur += ch;
    }
    cols.push(cur);

    const obj = {};
    header.forEach((h, idx) => { obj[h] = (cols[idx] ?? "").trim(); });
    rows.push(obj);
  }
  return rows;
}

function toLower(value){
  return (value ?? "").toString().toLowerCase();
}

function parseUtc(value){
  const d = new Date(value);
  if(Number.isNaN(d.getTime())){ return null; }
  return d;
}

function daysAgo(dateStr){
  const d = parseUtc(dateStr);
  if(!d){ return Infinity; }
  return (Date.now() - d.getTime()) / (1000 * 60 * 60 * 24);
}

function formatScore(row){
  if(Number.isFinite(row.score_value)){ return row.score_value.toFixed(8); }
  if(row.score_text){ return row.score_text; }
  return "--";
}

function formatUtc(dateStr){
  const d = parseUtc(dateStr);
  if(!d){ return "--"; }
  const value = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  }).format(d);
  return `${value} UTC`;
}

function sourceClassName(source){
  return `source-${toLower(source).replace(/[^a-z0-9_-]/g, "_") || "manual"}`;
}

const REPO_BLOB_BASE = "https://github.com/Bechirdardouri/gnn-challenge/blob/main/";

function encodeRepoPath(path){
  return path
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join("/");
}

function resolveRowLink(row){
  const notes = (row.notes ?? "").toString().trim();
  if(/^https?:\/\//i.test(notes)){
    return notes;
  }
  if(notes.startsWith("submissions/") || notes.startsWith("docs/") || notes.startsWith("leaderboard/")){
    return `${REPO_BLOB_BASE}${encodeRepoPath(notes)}`;
  }
  return "";
}

function appendLinkedText(cell, text, url, className){
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.className = className;
  link.textContent = text;
  cell.appendChild(link);
}

const state = {
  rows: [],
  filtered: [],
  sortKey: "score",
  sortDir: "desc",
  hiddenCols: new Set(),
};

function sortRows(rows){
  const dir = state.sortDir === "asc" ? 1 : -1;
  const key = state.sortKey;

  return [...rows].sort((a, b) => {
    if(key === "score"){
      const av = Number.isFinite(a.score_value) ? a.score_value : Number.NEGATIVE_INFINITY;
      const bv = Number.isFinite(b.score_value) ? b.score_value : Number.NEGATIVE_INFINITY;
      return (av - bv) * dir;
    }

    if(key === "rank"){
      const av = Number.isFinite(a.score_value) ? a.score_value : Number.NEGATIVE_INFINITY;
      const bv = Number.isFinite(b.score_value) ? b.score_value : Number.NEGATIVE_INFINITY;
      const rankDir = state.sortDir === "asc" ? -1 : 1;
      return (av - bv) * rankDir;
    }

    if(key === "timestamp_utc"){
      const av = parseUtc(a.timestamp_utc)?.getTime() ?? Number.NEGATIVE_INFINITY;
      const bv = parseUtc(b.timestamp_utc)?.getTime() ?? Number.NEGATIVE_INFINITY;
      return (av - bv) * dir;
    }

    const av = toLower(a[key]);
    const bv = toLower(b[key]);
    if(av < bv){ return -1 * dir; }
    if(av > bv){ return 1 * dir; }
    return 0;
  });
}

function updateSortIndicators(){
  document.querySelectorAll("#tbl thead th").forEach((th) => {
    const key = th.dataset.key || "";
    if(key === state.sortKey){
      th.dataset.order = state.sortDir;
      th.setAttribute("aria-sort", state.sortDir === "asc" ? "ascending" : "descending");
    }else{
      th.dataset.order = "";
      th.setAttribute("aria-sort", "none");
    }
  });
}

function updateStatus(rows){
  const status = document.getElementById("status");
  status.textContent = rows.length ? `${rows.length} result(s)` : "No results";
}

function renderSummary(rows){
  const statSubmissions = document.getElementById("statSubmissions");
  const statTopScore = document.getElementById("statTopScore");
  const statLeader = document.getElementById("statLeader");
  const updatedAt = document.getElementById("updatedAt");

  statSubmissions.textContent = `${rows.length}`;

  const ordered = [...rows].sort((a, b) => {
    const av = Number.isFinite(a.score_value) ? a.score_value : Number.NEGATIVE_INFINITY;
    const bv = Number.isFinite(b.score_value) ? b.score_value : Number.NEGATIVE_INFINITY;
    if(av !== bv){ return bv - av; }
    const ad = parseUtc(a.timestamp_utc)?.getTime() ?? 0;
    const bd = parseUtc(b.timestamp_utc)?.getTime() ?? 0;
    return bd - ad;
  });

  const leader = ordered[0];
  statTopScore.textContent = leader ? formatScore(leader) : "--";
  statLeader.textContent = leader ? leader.team : "--";

  const latest = rows
    .map((row) => parseUtc(row.timestamp_utc))
    .filter(Boolean)
    .sort((a, b) => b.getTime() - a.getTime())[0];
  updatedAt.textContent = latest ? `Updated: ${formatUtc(latest.toISOString())}` : "Updated: --";
}

function renderPodium(rows){
  const ordered = [...rows].sort((a, b) => {
    const av = Number.isFinite(a.score_value) ? a.score_value : Number.NEGATIVE_INFINITY;
    const bv = Number.isFinite(b.score_value) ? b.score_value : Number.NEGATIVE_INFINITY;
    if(av !== bv){ return bv - av; }
    const ad = parseUtc(a.timestamp_utc)?.getTime() ?? 0;
    const bd = parseUtc(b.timestamp_utc)?.getTime() ?? 0;
    return bd - ad;
  });

  const slots = [
    { rank: 1, row: ordered[0], cls: "first" },
    { rank: 2, row: ordered[1], cls: "second" },
    { rank: 3, row: ordered[2], cls: "third" },
  ];

  slots.forEach((slot) => {
    const team = document.getElementById(`podium${slot.rank}Team`);
    const score = document.getElementById(`podium${slot.rank}Score`);
    const item = document.querySelector(`.podium-item.${slot.cls}`);
    if(!team || !score || !item){ return; }

    if(!slot.row){
      team.textContent = "--";
      score.textContent = "--";
      item.classList.add("empty");
      return;
    }

    const rowUrl = resolveRowLink(slot.row);
    team.textContent = "";
    if(rowUrl){
      const link = document.createElement("a");
      link.href = rowUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "inline-link podium-link";
      link.textContent = slot.row.team || "--";
      team.appendChild(link);
    }else{
      team.textContent = slot.row.team || "--";
    }
    score.textContent = formatScore(slot.row);
    item.classList.remove("empty");
  });
}

function renderTable(){
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = "";

  state.filtered.forEach((row, idx) => {
    const tr = document.createElement("tr");
    tr.className = "row-in";
    tr.style.animationDelay = `${Math.min(idx * 26, 320)}ms`;
    const rowLink = resolveRowLink(row);

    if(rowLink){
      tr.classList.add("clickable-row");
      tr.addEventListener("click", (event) => {
        if(event.target.closest("a")){ return; }
        window.open(rowLink, "_blank", "noopener,noreferrer");
      });
    }

    const cellDefs = [
      { key: "rank", text: `${idx + 1}` },
      { key: "team", text: row.team || "--" },
      { key: "model", text: row.model || "--" },
      { key: "score", text: formatScore(row) },
      { key: "source", text: row.source || "manual" },
      { key: "timestamp_utc", text: formatUtc(row.timestamp_utc), title: row.timestamp_utc || "--" },
      { key: "notes", text: row.notes || "--" },
    ];

    cellDefs.forEach((cell) => {
      const td = document.createElement("td");
      td.dataset.key = cell.key;
      td.classList.add(cell.key);

      if(cell.key === "source"){
        const span = document.createElement("span");
        span.className = `source-pill ${sourceClassName(cell.text)}`;
        span.textContent = cell.text;
        td.appendChild(span);
      }else if(cell.key === "team" && rowLink){
        appendLinkedText(td, cell.text, rowLink, "inline-link team-link");
      }else if(cell.key === "notes" && rowLink){
        const noteLabel = cell.text && cell.text !== "--" ? cell.text : "Open";
        appendLinkedText(td, noteLabel, rowLink, "inline-link notes-link");
      }else{
        td.textContent = cell.text;
      }

      if(cell.title){
        td.title = cell.title;
      }
      if(state.hiddenCols.has(cell.key)){
        td.style.display = "none";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  document.querySelectorAll("#tbl thead th").forEach((th) => {
    const key = th.dataset.key || "";
    th.style.display = state.hiddenCols.has(key) ? "none" : "";
  });

  updateSortIndicators();
  updateStatus(state.filtered);
}

function applyFilters(){
  const query = toLower(document.getElementById("search").value.trim());
  const model = document.getElementById("modelFilter").value;
  const source = document.getElementById("sourceFilter").value;
  const date = document.getElementById("dateFilter").value;

  let rows = [...state.rows];

  if(model !== "all"){
    rows = rows.filter((row) => toLower(row.model) === model);
  }

  if(source !== "all"){
    rows = rows.filter((row) => toLower(row.source) === source);
  }

  if(date !== "all"){
    const maxDays = date === "last30" ? 30 : 180;
    rows = rows.filter((row) => daysAgo(row.timestamp_utc) <= maxDays);
  }

  if(query){
    rows = rows.filter((row) => {
      const haystack = toLower(`${row.team} ${row.model} ${row.source} ${row.notes} ${row.timestamp_utc}`);
      return haystack.includes(query);
    });
  }

  state.filtered = sortRows(rows);
  renderTable();
  renderPodium(state.filtered);
}

function setupColumnToggles(){
  const columns = [
    ["rank", "Rank"],
    ["team", "Team"],
    ["model", "Model"],
    ["score", "Score"],
    ["source", "Source"],
    ["timestamp_utc", "Date (UTC)"],
    ["notes", "Notes"],
  ];

  const wrap = document.getElementById("columnToggles");
  wrap.innerHTML = "";

  columns.forEach(([key, label]) => {
    const chip = document.createElement("label");
    chip.className = "toggle-pill";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !state.hiddenCols.has(key);
    cb.addEventListener("change", () => {
      if(cb.checked){ state.hiddenCols.delete(key); }
      else{ state.hiddenCols.add(key); }
      renderTable();
    });

    const text = document.createElement("span");
    text.textContent = label;

    chip.appendChild(cb);
    chip.appendChild(text);
    wrap.appendChild(chip);
  });
}

function setupSorting(){
  document.querySelectorAll("#tbl thead th").forEach((th) => {
    const applySort = () => {
      const key = th.dataset.key;
      if(!key){ return; }
      if(state.sortKey === key){
        state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      }else{
        state.sortKey = key;
        state.sortDir = key === "score" || key === "rank" ? "desc" : "asc";
      }
      applyFilters();
    };

    th.tabIndex = 0;
    th.addEventListener("click", applySort);
    th.addEventListener("keydown", (event) => {
      if(event.key === "Enter" || event.key === " "){
        event.preventDefault();
        applySort();
      }
    });
  });
}

function appendSelectOptions(selectId, rows, field){
  const select = document.getElementById(selectId);
  const values = new Map();

  rows.forEach((row) => {
    const raw = (row[field] ?? "").toString().trim();
    const lowered = toLower(raw);
    if(!lowered || values.has(lowered)){ return; }
    values.set(lowered, raw);
  });

  [...values.entries()]
    .sort((a, b) => a[1].localeCompare(b[1], undefined, { sensitivity: "base" }))
    .forEach(([value, label]) => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      select.appendChild(opt);
    });
}

function normalizeRows(rows){
  return rows
    .filter((row) => (row.team ?? "").toString().trim())
    .map((row) => {
      const scoreText = (row.score ?? "").toString().trim();
      const parsed = Number.parseFloat(scoreText);
      return {
        timestamp_utc: (row.timestamp_utc ?? "").toString().trim(),
        team: (row.team ?? "").toString().trim(),
        model: (row.model ?? "").toString().trim(),
        source: (row.source ?? "manual").toString().trim(),
        notes: (row.notes ?? "").toString().trim(),
        score_text: scoreText,
        score_value: Number.isFinite(parsed) ? parsed : Number.NaN,
      };
    });
}

async function loadRows(){
  try{
    const jsonRes = await fetch("leaderboard.json", { cache: "no-store" });
    if(jsonRes.ok){
      const jsonRows = await jsonRes.json();
      if(Array.isArray(jsonRows)){
        return normalizeRows(jsonRows);
      }
    }
  }catch(_error){
    // fallback to CSV below
  }

  const csvRes = await fetch("../leaderboard/leaderboard.csv", { cache: "no-store" });
  if(!csvRes.ok){
    throw new Error(`Failed to fetch leaderboard CSV (${csvRes.status})`);
  }
  const csvText = await csvRes.text();
  return normalizeRows(parseCSV(csvText));
}

async function main(){
  const status = document.getElementById("status");
  try{
    state.rows = await loadRows();
    appendSelectOptions("modelFilter", state.rows, "model");
    appendSelectOptions("sourceFilter", state.rows, "source");

    setupColumnToggles();
    setupSorting();
    renderSummary(state.rows);

    document.getElementById("search").addEventListener("input", applyFilters);
    document.getElementById("modelFilter").addEventListener("change", applyFilters);
    document.getElementById("sourceFilter").addEventListener("change", applyFilters);
    document.getElementById("dateFilter").addEventListener("change", applyFilters);

    state.sortKey = "score";
    state.sortDir = "desc";
    applyFilters();
  }catch(error){
    status.textContent = "Failed to load leaderboard.";
    console.error(error);
  }
}

main();
