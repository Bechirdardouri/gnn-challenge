/* Minimal, dependency-free leaderboard UI:
 * - loads leaderboard.json (fallback to ../leaderboard/leaderboard.csv)
 * - search + filters (model, source, date)
 * - sortable columns
 * - column toggles
 */
function parseCSV(text){
  const lines = text.trim().split(/\r?\n/);
  if(!lines.length) return [];
  const header = lines[0].split(",");
  const rows = [];
  for(let i=1;i<lines.length;i++){
    if(!lines[i].trim()) continue;
    const cols = [];
    let cur="", inQ=false;
    for(let j=0;j<lines[i].length;j++){
      const ch = lines[i][j];
      if(ch === '"'){ inQ = !inQ; continue; }
      if(ch === "," && !inQ){ cols.push(cur); cur=""; continue; }
      cur += ch;
    }
    cols.push(cur);
    const obj = {};
    header.forEach((h, idx) => obj[h] = (cols[idx] ?? "").trim());
    rows.push(obj);
  }
  return rows;
}

function daysAgo(dateStr){
  const d = new Date(dateStr);
  if(isNaN(d.getTime())) return Infinity;
  const now = new Date();
  return (now - d) / (1000*60*60*24);
}

const state = {
  rows: [],
  filtered: [],
  sortKey: "score",
  sortDir: "desc",
  hiddenCols: new Set(),
};

function renderTable(){
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = "";
  const rows = state.filtered;

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");
    const rank = idx + 1;
    const cells = [
      ["rank", rank],
      ["team", r.team],
      ["model", r.model],
      ["score", r.score],
      ["source", r.source],
      ["timestamp_utc", r.timestamp_utc],
      ["notes", r.notes || ""],
    ];
    cells.forEach(([k, v]) => {
      const td = document.createElement("td");
      td.dataset.key = k;
      td.textContent = v;
      if(k === "rank") td.classList.add("rank");
      if(k === "score") td.classList.add("score");
      if(state.hiddenCols.has(k)) td.style.display = "none";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  document.querySelectorAll("#tbl thead th").forEach(th => {
    const k = th.dataset.key;
    th.style.display = state.hiddenCols.has(k) ? "none" : "";
  });

  document.getElementById("status").textContent =
    rows.length ? `${rows.length} result(s)` : "No results";
}

function applyFilters(){
  const q = document.getElementById("search").value.toLowerCase().trim();
  const model = document.getElementById("modelFilter").value;
  const source = document.getElementById("sourceFilter").value;
  const date = document.getElementById("dateFilter").value;

  let rows = [...state.rows];

  if(model !== "all"){
    rows = rows.filter(r => (r.model || "").toLowerCase() === model);
  }

  if(source !== "all"){
    rows = rows.filter(r => (r.source || "").toLowerCase() === source);
  }

  if(date !== "all"){
    const maxDays = (date === "last30") ? 30 : 180;
    rows = rows.filter(r => daysAgo(r.timestamp_utc) <= maxDays);
  }

  if(q){
    rows = rows.filter(r => {
      const hay = `${r.team} ${r.model} ${r.source} ${r.notes} ${r.timestamp_utc}`.toLowerCase();
      return hay.includes(q);
    });
  }

  const k = state.sortKey;
  const dir = state.sortDir === "asc" ? 1 : -1;
  rows.sort((a,b) => {
    let av = a[k], bv = b[k];
    if(k === "score"){
      av = parseFloat(av); bv = parseFloat(bv);
      if(isNaN(av)) av = -Infinity;
      if(isNaN(bv)) bv = -Infinity;
      return (av - bv) * dir;
    }
    av = (av ?? "").toString().toLowerCase();
    bv = (bv ?? "").toString().toLowerCase();
    if(av < bv) return -1 * dir;
    if(av > bv) return  1 * dir;
    return 0;
  });

  state.filtered = rows;
  renderTable();
}

function setupColumnToggles(){
  const cols = [
    ["rank","Rank"],
    ["team","Team"],
    ["model","Model"],
    ["score","Score"],
    ["source","Source"],
    ["timestamp_utc","Date (UTC)"],
    ["notes","Notes"],
  ];
  const wrap = document.getElementById("columnToggles");
  wrap.innerHTML = "";
  cols.forEach(([k,label]) => {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !state.hiddenCols.has(k);
    cb.addEventListener("change", () => {
      if(cb.checked) state.hiddenCols.delete(k);
      else state.hiddenCols.add(k);
      renderTable();
    });
    lab.appendChild(cb);
    const sp = document.createElement("span");
    sp.textContent = label;
    lab.appendChild(sp);
    wrap.appendChild(lab);
  });
}

function setupSorting(){
  document.querySelectorAll("#tbl thead th").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      if(!k) return;
      if(state.sortKey === k){
        state.sortDir = (state.sortDir === "asc") ? "desc" : "asc";
      }else{
        state.sortKey = k;
        state.sortDir = (k === "score") ? "desc" : "asc";
      }
      applyFilters();
    });
  });
}

function normalizeRows(rows){
  return rows
    .filter(r => r.team)
    .map(r => ({
      timestamp_utc: r.timestamp_utc || "",
      team: r.team || "",
      model: (r.model || "").toLowerCase(),
      score: r.score || "",
      source: (r.source || "manual").toLowerCase(),
      notes: r.notes || "",
    }));
}

function appendSelectOptions(selectId, values){
  const sel = document.getElementById(selectId);
  [...new Set(values.filter(Boolean))].sort().forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  });
}

async function loadRows(){
  try{
    const jsonRes = await fetch("leaderboard.json", {cache:"no-store"});
    if(jsonRes.ok){
      const jsonRows = await jsonRes.json();
      if(Array.isArray(jsonRows)) return normalizeRows(jsonRows);
    }
  }catch(_err){
    // fallback to CSV below
  }
  const csvRes = await fetch("../leaderboard/leaderboard.csv", {cache:"no-store"});
  const csvText = await csvRes.text();
  return normalizeRows(parseCSV(csvText));
}

async function main(){
  const status = document.getElementById("status");
  try{
    state.rows = await loadRows();
    appendSelectOptions("modelFilter", state.rows.map(r => r.model));
    appendSelectOptions("sourceFilter", state.rows.map(r => r.source));

    setupColumnToggles();
    setupSorting();

    document.getElementById("search").addEventListener("input", applyFilters);
    document.getElementById("modelFilter").addEventListener("change", applyFilters);
    document.getElementById("sourceFilter").addEventListener("change", applyFilters);
    document.getElementById("dateFilter").addEventListener("change", applyFilters);

    state.sortKey = "score";
    state.sortDir = "desc";
    applyFilters();
  }catch(e){
    status.textContent = "Failed to load leaderboard.";
    console.error(e);
  }
}

main();
